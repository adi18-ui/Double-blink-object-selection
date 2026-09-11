"""
preprocessing.py  —  EXACT reproduction of the collector's on-board filter.
==========================================================================
VERIFIED: X.npy was built from the collector's `filtered_chX` columns
(correlation 1.0000 against your abhi recording). So the live filter must be
the SAME chain the collector applied, sample-by-sample, stateful:

    0.5 Hz high-pass (butter order 2)
    50 Hz notch      (iirnotch, Q=30)
    45 Hz low-pass   (butter order 2)

This is identical to the collector's apply_filters(); reproduced here so the
live stream is filtered exactly as the training data was. Do NOT substitute a
different band (an earlier 0.5-10 Hz guess did not match).
"""

import numpy as np
from scipy import signal

FS = 256


class CollectorFilter:
    """Stateful, causal, per-channel — matches the collector's filtered_ output."""

    def __init__(self, fs=FS, n_channels=2):
        self.n = n_channels
        self.hp_sos = signal.butter(2, 0.5, btype="highpass", fs=fs, output="sos")
        self.notch_b, self.notch_a = signal.iirnotch(50, 30, fs)
        self.lp_sos = signal.butter(2, 45, btype="lowpass", fs=fs, output="sos")
        # per-channel filter state, initialised exactly like the collector
        self.hp_zi = [signal.sosfilt_zi(self.hp_sos) for _ in range(n_channels)]
        self.notch_zi = [signal.lfilter_zi(self.notch_b, self.notch_a) for _ in range(n_channels)]
        self.lp_zi = [signal.sosfilt_zi(self.lp_sos) for _ in range(n_channels)]

    def process(self, ch, x):
        """Filter ONE raw sample for channel `ch`. Returns the filtered scalar."""
        y, self.hp_zi[ch] = signal.sosfilt(self.hp_sos, [x], zi=self.hp_zi[ch]); x = y[0]
        y, self.notch_zi[ch] = signal.lfilter(self.notch_b, self.notch_a, [x], zi=self.notch_zi[ch]); x = y[0]
        y, self.lp_zi[ch] = signal.sosfilt(self.lp_sos, [x], zi=self.lp_zi[ch]); x = y[0]
        return float(x)

    def reset(self):
        self.hp_zi = [signal.sosfilt_zi(self.hp_sos) for _ in range(self.n)]
        self.notch_zi = [signal.lfilter_zi(self.notch_b, self.notch_a) for _ in range(self.n)]
        self.lp_zi = [signal.sosfilt_zi(self.lp_sos) for _ in range(self.n)]


# backwards-compat alias (live_demo imported BandpassPreprocessor before)
BandpassPreprocessor = CollectorFilter
