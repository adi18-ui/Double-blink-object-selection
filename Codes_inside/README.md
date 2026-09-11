# Live Double-Blink Detector 
Turns your trained `EOG_NET` checkpoint into a live `SELECT` event for the grid UI.

## Files
- `eog_model.py` — exact `EOG_NET` architecture from your notebook (CNN + 4-layer pre-LN transformer). Loads your checkpoint with **0 missing / 0 unexpected** keys.
- `norm_stats.json` — per-channel normalization from the **training split (seed=42)**. Live input must use these or the model silently degrades.
- `preprocessing.py` — real-time band-pass (~0.5–10 Hz) matching the signal `X.npy` was trained on. **Verify against your build script** (recipe below).
- `live_blink_detector.py` — the detector: sliding window + smoothing + threshold + debounce → `on_select`.
- `live_demo.py` — your collector's acquisition (reader + queue + worker) stripped of the experiment, with preprocessing + detector + logging wired in. Feeds ESP32 → SELECT.

## Preprocessing
`X.npy` was built from the collector's `filtered_chX` columns — confirmed by
reproducing an `X.npy` window from your `abhi_full_01` recording at correlation
**1.0000**. So the live filter is your collector's exact chain:
**0.5 Hz high-pass → 50 Hz notch → 45 Hz low-pass** (stateful, per channel).
`preprocessing.py` (`CollectorFilter`) reproduces the `filtered_` column from raw
with correlation 1.000000 (max diff 0.000). No tuning needed — it's exact.

Windowing used to build `X.npy`: 768 samples (3 s) starting at each marker;
DOUBLE_BLINK → label 1, single/rest → label 0. `X.npy` spans several of your
sessions (this one file supplied 20 of the 247 windows).


## The inference 
- input `(1, 2, 768)` float32, channel order **ch0 then ch1**
- normalize per channel: `x = (x - mean) / std` with the values in `norm_stats.json`
  - ch0: mean −1.1673, std 160.4709
  - ch1: mean −1.5997, std 196.7938
- 768 samples @ **256 Hz = 3.0 s** window. If your live stream isn't 256 Hz, resample so 768 samples == 3 s.
- `model.eval()` + `torch.no_grad()` (handled inside the module)

## Wire it up (live)
```python
from live_blink_detector import LiveBlinkDetector

det = LiveBlinkDetector(
    checkpoint="eog-cnn-epoch_03-val_f1_1_0000_ckpt.zip",
    norm_stats="norm_stats.json",
    on_select=lambda info: grid.select_highlighted(),
    threshold=0.5,
)
for ch0, ch1 in stream:      
    det.push(ch0, ch1)
```
`on_select` is the single event the grid consumes: it fires once per confirmed
double-blink. Everything else (which tile is highlighted) stays in the UI.


Run your **real** session recording (continuous, not pre-windowed) through the
same logic:
```python
det.run_offline(signal_2xN)  
```
Check that it fires once per real double-blink and stays silent during rest.
This is the true test — a synthetic concatenation of windows overstates the
edge-smear problem because the transitions aren't real.

## Tuning — validated on real recording
Replaying `abhi_full_01` (12 min, 190k raw samples) through this exact pipeline
(filter → model → fire logic), scored against your markers:

| threshold | smooth | refractory | doubles caught | false fires |
|-----------|--------|-----------|----------------|-------------|
| 0.5 | 2/3 | 1.5 s | 10/50 | 47 |
| 0.5 | 3/4 | 2.0 s | 17/50 | 34 |
| 0.6 | 3/4 | 2.0 s | 23/50 | 27 |
| 0.7 | 3/5 | 2.0 s | 27/50 | 18 |
| **0.7** | **4/6** | **2.0 s** | **41/50 (82%)** | **2** |

`live_demo.py` ships with the last row as default. Honest read: ~82% of
deliberate double-blinks register on the first try (repeat the rest), and about
one spurious selection per ~6 min. This is **lower than the 93% test-set F1**
because live uses a sliding window over continuous data (the model was trained
on windows centred on events), and most trials in this file were unseen. Tune on
your own live stream; push threshold/smoothing higher if false selections bother
you more than occasional repeats.


- **Single subject / single session.** This generalizes to the same person +
  setup it was trained on — fine for your demo, not yet proven across people.
- **Val F1 = 1.0000 is inflated** by ~8% near-duplicate windows leaking across a
  random split. Trust the held-out behavior (~0.9 F1), not the 1.0.
- **Edge smear** is the main live risk. The smoothing + refractory handle it;
  `run_offline` on a real recording is how you confirm the settings.
- The 2 hard false-positive windows are worth eyeballing — plotting them tells
  you whether it's fixable (threshold) or a genuine model limit.

If false selections are still too frequent live, your **rule-based detector had
0% rest false-fires** and is the safer fallback for the demo. This kit doesn't
close that comparison — you deferred it — but the fallback is there.

---

# Grid UI + gaze demo (wired)

Three pieces now connect the detector to your grid:
- `bci_bridge.py` — SSE server (stdlib, no deps). The detector pushes SELECT to
  the browser through it. Model-independent — a retrained model needs no change.
- `bci_integration.js` — added to `Open_BCI.html`. Real webcam gaze (webgazer)
  highlights the tile you look at; the bridge's SELECT selects it on a blink.
- `Open_BCI.html` — your grid, now loading webgazer + the integration script.


1. `pip install pyserial torch scipy pandas numpy`
2. `python live_demo.py`  (starts the detector AND the SSE bridge on :8765)
3. Open `Open_BCI.html` in Chrome. Allow the webcam. Click a few points to
   calibrate webgazer, then look at tiles to highlight them.
4. Double-blink to select the highlighted tile.

Keyboard fallbacks already in the UI: keys 1-4 focus tiles, Enter = select,
Esc = unfocus, D = dev panel. Handy if webcam/serial isn't ready.


The current model only fires reliably on "look up + double-blink". With real
gaze you look *around* the grid, so selecting bottom tiles by blinking while
looking down won't work yet — the blink won't register looking down. Options
until you retrain gaze-invariant:
- Demo the top tiles (where look-up-and-blink is natural), or
- Set `GAZE_MODE = 'mouse'` in `bci_integration.js` so the mouse drives the
  highlight while you keep looking at the screen to blink (selection just works
  for a scripted demo), or
- Retrain across gaze directions (the real fix) — then remove this caveat.

Swapping the model later touches nothing in the UI/bridge: retrain, replace the
checkpoint + `norm_stats.json`, done.

---

# Rule-based detector (no ML)

`rule_detector.py` is a threshold+refractory+count double-blink detector. No
model, no normalization, no training data, and it runs on **ch0 alone** — so a
flaky ch1 electrode can't break it. Same push()/on_select interface as the ML
detector, so it's a one-line swap.

## Validated head-to-head
| detector | recall | rest false-fires | needs |
|----------|--------|------------------|-------|
| ML (EOG_NET) | 41/50 (82%) | 2 | model + norm stats + both channels |
| **rule-based** | **50/50 (100%)** | **0** | ch0 only |

The rule detector caught every double-blink with zero rest false-fires (the few
extra fires were near single-blink cues, not during rest). It's the stronger,
simpler choice for the live demo right now.

## Switch detectors
In `live_demo.py`: `DETECTOR_MODE = "rule"` (default) or `"ml"`. Nothing else
changes — both feed the same SELECT bridge to the grid UI.

## How it works / knobs
- baseline = median(recent), SD = 1.4826*MAD(recent) — MAD ignores blinks so SD
  tracks rest noise; auto-adapts, no calibration step.
- a blink = deflection > `threshold_sd` * SD (default 7). `polarity="neg"` because
  your ch0 blinks deflect negative (validated); if it won't fire live, try "pos"
  or "abs".
- `blink_refractory_s` (0.3) collapses one biphasic blink into one event.
- a double = two blinks with gap in [`double_min_gap_s`, `double_max_gap_s`]
  (0.3–1.0 s; yours sit ~0.76 s). `fire_debounce_s` (2.0) = one double fires once.

Tune on your own stream with `run_offline(filtered_2xN)`. If you get false fires
live, raise `threshold_sd`; if it misses doubles, lower it or widen the gap.

---

# Live signal monitor 
A small ch0 trace with blink/select markers renders bottom-left in the browser
while you present. It reads the same SSE bridge the grid uses — nothing is added
to the reader/worker hot path beyond a throttled, non-blocking push.

Measured cost: ~0.05 us/sample when no browser is watching (guarded), ~1 us per
push at 32 Hz when it is — about 0.003% CPU. Detection timing is unaffected.

Toggles:
- `MONITOR = True` in `live_demo.py` (and `MONITOR_DECIM = 8` -> 32 Hz push rate).
  Set `MONITOR = False` for the leanest possible loop.
- `SHOW_MONITOR` in `bci_integration.js` hides/shows the canvas.

Colors: blue = ch0 trace, green flash = a detected blink (rule mode),
yellow flash = a SELECT. The blink flashes let you see the detector reacting in
real time even between selections.
