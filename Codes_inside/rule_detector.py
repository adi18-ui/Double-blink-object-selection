"""
rule_detector.py  —  rule-based double-blink detector (no ML)
=============================================================
Onset/offset blink detection + amplitude-ratio double guard. No model, no
normalization, no training data. Runs on ONE channel (default ch0), so it's
immune to a flaky second electrode.

Method:
  1. Robust baseline + noise scale from a rolling window:
       baseline = median(recent),  SD = 1.4826 * MAD(recent)
     MAD ignores blinks, so SD tracks rest noise.
  2. A BLINK is one contiguous excursion past `threshold_sd`*SD (onset->offset).
     Detecting the whole excursion as one event (not a threshold crossing)
     naturally avoids double-counting the biphasic blink shape. Each blink's
     PEAK amplitude is recorded.
  3. A DOUBLE-BLINK = two blinks whose gap is in [min_gap, max_gap] AND whose
     peaks are comparable (min/max >= `amp_ratio`). The amplitude test is what
     separates a real double (two similar blinks) from a single-blink + filter
     ring (one big blink + one small ring) — they overlap in timing but not in
     size. -> fire on_select, then debounce.

Same interface as before: push(ch0, ch1, t), on_select / on_blink callbacks,
run_offline(filtered_2xN).

Polarity: default 'neg' (abhi's ch0 blinks go negative). 'pos' or 'abs' if a
different montage flips it.
"""

import time
from collections import deque

import numpy as np


class RuleBasedBlinkDetector:
    def __init__(
        self,
        on_select=None,
        on_blink=None,
        *,
        fs: int = 256,
        channel: int = 0,
        threshold_sd: float = 7.0,
        min_gap_s: float = 0.25,       # a double's two blinks: at least this apart
        max_gap_s: float = 1.1,        # ...and at most this (abhi's run ~0.3-0.9 s)
        amp_ratio: float = 0.45,       # 2nd blink peak >= this * 1st (rejects rings)
        min_blink_sep_s: float = 0.12, # min quiet gap between two blink excursions
        fire_debounce_s: float = 1.0,
        sd_window_s: float = 10.0,
        warmup_s: float = 5.0,
        polarity: str = "neg",
        sd_update_every: int = 32,
        # accepted for backward-compat (ignored / mapped):
        blink_refractory_s: float = None,
        double_min_gap_s: float = None,
        double_max_gap_s: float = None,
    ):
        self.on_select = on_select or (lambda info: None)
        self.on_blink = on_blink or (lambda info: None)
        self.fs = fs
        self.channel = channel
        self.threshold_sd = threshold_sd
        self.min_gap_s = double_min_gap_s if double_min_gap_s is not None else min_gap_s
        self.max_gap_s = double_max_gap_s if double_max_gap_s is not None else max_gap_s
        self.amp_ratio = amp_ratio
        self.min_blink_sep_s = min_blink_sep_s
        self.fire_debounce_s = fire_debounce_s
        self.warmup_s = warmup_s
        self.polarity = polarity
        self.sd_update_every = sd_update_every

        self.buf = deque(maxlen=int(sd_window_s * fs))
        self.blinks = deque()          # (onset_t, peak_mag)
        self._t0 = None
        self._baseline = 0.0
        self._sd = 1.0
        self._since_sd = 0
        self._in_blink = False
        self._onset_t = 0.0
        self._peak = 0.0
        self._last_offset_t = -1e9
        self._last_fire_t = -1e9
        self.last_blink = None

    def push(self, ch0, ch1=None, t=None):
        x = ch0 if self.channel == 0 else ch1
        if t is None:
            t = time.time()
        if self._t0 is None:
            self._t0 = t
        self.buf.append(x)

        self._since_sd += 1
        if self._since_sd >= self.sd_update_every and len(self.buf) >= self.fs:
            arr = np.asarray(self.buf, dtype=np.float64)
            self._baseline = float(np.median(arr))
            mad = float(np.median(np.abs(arr - self._baseline)))
            self._sd = 1.4826 * mad if mad > 1e-9 else 1.0
            self._since_sd = 0

        if (t - self._t0) < self.warmup_s:
            return

        dev = x - self._baseline
        if self.polarity == "neg":
            mag = -dev
        elif self.polarity == "pos":
            mag = dev
        else:
            mag = abs(dev)

        thr = self.threshold_sd * self._sd
        if mag > thr:
            # inside a blink excursion
            if not self._in_blink:
                # require a small quiet gap since the last blink ended
                if (t - self._last_offset_t) >= self.min_blink_sep_s:
                    self._in_blink = True
                    self._onset_t = t
                    self._peak = mag
                else:
                    self._peak = max(self._peak, mag)  # merge with just-ended blink
            else:
                self._peak = max(self._peak, mag)
        else:
            if self._in_blink:
                self._in_blink = False
                self._last_offset_t = t
                self._register_blink(self._onset_t, self._peak)

    def _register_blink(self, onset_t, peak):
        self.last_blink = onset_t
        self.on_blink({"time": onset_t, "peak": round(peak, 1)})
        self.blinks.append((onset_t, peak))
        while self.blinks and (onset_t - self.blinks[0][0]) > self.max_gap_s:
            self.blinks.popleft()
        if len(self.blinks) >= 2:
            t1, p1 = self.blinks[-2]
            t2, p2 = self.blinks[-1]
            gap = t2 - t1
            ratio = min(p1, p2) / max(p1, p2) if max(p1, p2) > 0 else 0
            if (self.min_gap_s <= gap <= self.max_gap_s
                    and ratio >= self.amp_ratio
                    and (t2 - self._last_fire_t) >= self.fire_debounce_s):
                self._fire(t2, gap, ratio)

    def _fire(self, t, gap, ratio):
        self._last_fire_t = t
        self.blinks.clear()
        self.on_select({"time": t, "gap": round(gap, 3), "ratio": round(ratio, 2)})

    def run_offline(self, filtered_2xN, verbose=True):
        fires = []
        orig = self.on_select
        self.on_select = lambda info: fires.append(info)
        self.buf.clear(); self.blinks.clear()
        self._t0 = None; self._in_blink = False
        self._last_offset_t = -1e9; self._last_fire_t = -1e9
        sig = np.asarray(filtered_2xN, dtype=np.float64)
        c1 = sig[1] if sig.shape[0] > 1 else sig[0]
        for i in range(sig.shape[1]):
            self.push(sig[0, i], c1[i], t=i / self.fs)
        self.on_select = orig
        if verbose:
            print(f"offline: {len(fires)} SELECT event(s)")
        return fires