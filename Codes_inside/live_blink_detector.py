"""
live_blink_detector.py
----------------------
Turns your trained EOG_NET checkpoint into a live double-blink detector that
emits a SELECT event, ready to wire into the grid UI.

WHY THIS EXISTS
    Offline the model sees tidy, event-centred 3-second windows. Live, you slide
    a window over a continuous stream and a double-blink passes through every
    position in the window. So a single raw frame prediction is noisy near the
    edges. This module fixes that with three wrappers around the raw model:
        1. sliding window   - keep a rolling 768-sample buffer, infer every `hop`
        2. smoothing        - require the probability to stay high for a few
                              consecutive inferences before firing (kills the
                              off-centre wobble / edge false-positives)
        3. debounce         - after a SELECT, ignore input for `refractory_s`
                              so one double-blink fires exactly once

INFERENCE CONTRACT (must match training or the model silently degrades)
    - input shape (1, 2, 768) float32, channel order ch0 then ch1
    - normalise per channel with the TRAINING mean/std in norm_stats.json
    - 768 samples @ 256 Hz = 3.0 s window
    - model.eval() + torch.no_grad()

USAGE (live)
    det = LiveBlinkDetector(
        checkpoint="eog-cnn-epoch_03-val_f1_1_0000_ckpt.zip",
        norm_stats="norm_stats.json",
        on_select=lambda info: print("SELECT!", info),
    )
    # feed samples as they arrive from your ESP32/OpenBCI stream:
    for ch0, ch1 in stream:            # one (ch0, ch1) pair per sample @256Hz
        det.push(ch0, ch1)             # fires on_select() when a double-blink is confirmed

USAGE (offline test on a full recording)
    det.run_offline(signal_2xN)        # signal shape (2, N); prints every SELECT time
"""

import json
import time
from collections import deque

import numpy as np
import torch

from eog_model import EOG_NET


class LiveBlinkDetector:
    def __init__(
        self,
        checkpoint: str,
        norm_stats: str,
        on_select=None,
        *,
        threshold: float = 0.5,
        smooth_k: int = 2,          # need this many consecutive frames >= threshold to fire
        smooth_window: int = 3,     # look at the last N frames ...
        smooth_needed: int = 2,     # ... and require this many of them >= threshold
        hop: int = 64,              # run inference every `hop` samples (64 = ~0.25 s @256Hz)
        refractory_s: float = 1.5,  # blind period after a fire
        sample_rate: int = 256,
        window: int = 768,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.threshold = threshold
        self.smooth_window = smooth_window
        self.smooth_needed = smooth_needed
        self.hop = hop
        self.refractory_s = refractory_s
        self.sample_rate = sample_rate
        self.window = window
        self.on_select = on_select or (lambda info: None)

        # --- load normalization stats (training split) ---
        with open(norm_stats) as f:
            stats = json.load(f)
        self.ch_mean = np.array(stats["channel_mean"], dtype=np.float32).reshape(1, 2, 1)
        self.ch_std = np.array(stats["channel_std"], dtype=np.float32).reshape(1, 2, 1)
        if stats.get("window_samples", window) != window:
            print(f"[warn] norm_stats window={stats['window_samples']} != {window}")
        if stats.get("sample_rate_hz", sample_rate) != sample_rate:
            print(f"[warn] norm_stats sample_rate={stats['sample_rate_hz']} != {sample_rate}. "
                  "Resample your live stream so 768 samples == 3.0 s.")

        # --- load model ---
        self.model = self._load_model(checkpoint)

        # --- runtime state ---
        self.buf = deque(maxlen=window)          # rolling raw samples, each (ch0, ch1)
        self.prob_hist = deque(maxlen=smooth_window)
        self._since_hop = 0
        self._last_fire_t = -1e9
        self.last_prob = 0.0

    # ------------------------------------------------------------------ #
    def _load_model(self, checkpoint):
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
        # Lightning wraps EOG_NET as self.model.* -> strip the prefix
        sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
        net = EOG_NET(input_channels=2)
        missing, unexpected = net.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[warn] load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
        net.to(self.device).eval()
        return net

    def _infer(self, window_2x768: np.ndarray) -> float:
        """window_2x768: (2, 768) raw -> double-blink probability."""
        x = window_2x768.astype(np.float32)[None, :, :]      # (1,2,768)
        x = (x - self.ch_mean) / self.ch_std                 # train-stat normalization
        with torch.no_grad():
            logit = self.model(torch.from_numpy(x).to(self.device))
            prob = torch.sigmoid(logit).item()
        return prob

    # ------------------------------------------------------------------ #
    def push(self, ch0: float, ch1: float, t: float = None):
        """Feed ONE sample (both channels). Fires on_select() on a confirmed double-blink."""
        if t is None:
            t = time.time()
        self.buf.append((ch0, ch1))
        if len(self.buf) < self.window:
            return                                  # not enough data yet
        self._since_hop += 1
        if self._since_hop < self.hop:
            return                                  # only infer every `hop` samples
        self._since_hop = 0

        # in refractory? skip inference entirely
        if (t - self._last_fire_t) < self.refractory_s:
            return

        win = np.asarray(self.buf, dtype=np.float32).T   # (2, 768)
        prob = self._infer(win)
        self.last_prob = prob
        self.prob_hist.append(prob)

        # smoothing: N-of-M recent frames above threshold
        hits = sum(p >= self.threshold for p in self.prob_hist)
        if len(self.prob_hist) >= self.smooth_window and hits >= self.smooth_needed:
            self._fire(t, max(self.prob_hist))   # report the peak, not the trailing frame

    def _fire(self, t, prob):
        self._last_fire_t = t
        self.prob_hist.clear()
        info = {"time": t, "prob": round(prob, 3)}
        self.on_select(info)

    # ------------------------------------------------------------------ #
    def predict_window(self, window_2x768: np.ndarray) -> float:
        """One-shot probability for a single (2,768) window (for offline testing)."""
        return self._infer(np.asarray(window_2x768))

    def run_offline(self, signal_2xN: np.ndarray, verbose: bool = True):
        """
        Replay a recorded (2, N) signal through the same live logic.
        Returns list of SELECT sample-indices. Use this to sanity-check on a
        known recording before trusting the live stream.
        """
        fires = []
        orig = self.on_select
        self.on_select = lambda info: fires.append(info)
        # reset state
        self.buf.clear(); self.prob_hist.clear()
        self._since_hop = 0; self._last_fire_t = -1e9
        sig = np.asarray(signal_2xN, dtype=np.float32)
        for i in range(sig.shape[1]):
            # use sample index / sample_rate as the clock so refractory works in "signal time"
            self.push(sig[0, i], sig[1, i], t=i / self.sample_rate)
        self.on_select = orig
        if verbose:
            print(f"offline: {len(fires)} SELECT event(s)")
            for f in fires:
                print(f"  t={f['time']:.2f}s  prob={f['prob']}")
        return fires


# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "eog-cnn-epoch_03-val_f1_1_0000_ckpt.zip"
    det = LiveBlinkDetector(
        checkpoint=ckpt,
        norm_stats="norm_stats.json",
        on_select=lambda info: print(f">>> SELECT  prob={info['prob']}  t={info['time']:.2f}"),
        threshold=0.5,
    )
    print("Detector ready. Wire det.push(ch0, ch1) to your live stream.")
    print("The on_select callback is where you tell the grid UI to select the highlighted tile.")
