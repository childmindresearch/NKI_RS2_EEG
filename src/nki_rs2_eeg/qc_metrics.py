#%%
"""Quality control metrics for NKI RS2 EEG data."""

import logging

logger = logging.getLogger(__name__)
import numpy as np
import pandas as pd
from scipy import signal, stats
from pathlib import Path
import logging
from timeit import default_timer

from config import (
    FS, N_CHANNELS, CHANNEL_NAMES,
    VARIANCE_MIN, VARIANCE_MAX,
    PEAK_TO_PEAK_MAX,
    KURTOSIS_MAX,
    FLAT_THRESHOLD, FLAT_MIN_DURATION_SEC,
    MUSCLE_FREQ_BAND, MUSCLE_POWER_MAX,
    DRIFT_FREQ_BAND, DRIFT_POWER_MAX,
    LINE_NOISE_HZ, LINE_NOISE_POWER_MAX,
    WINDOW_SIZE_SAMPLES, WINDOW_STEP_SAMPLES,
    MIN_CLEAN_DATA_PCT, MAX_BAD_CHANNELS,
    RESULTS_DIR
)

logger = logging.getLogger(__name__)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def compute_band_power(data, freq_band, fs):
    """
    Compute average power in a frequency band for a single channel.

    Parameters
    ----------
    data : np.array
        1D array of EEG samples for one channel
    freq_band : tuple
        (low_hz, high_hz) frequency band of interest
    fs : int
        Sampling frequency

    Returns
    -------
    float
        Mean power in the frequency band
    """
    freqs, psd = signal.welch(data, fs=fs, nperseg=fs*2)
    band_mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
    return np.mean(psd[band_mask])


def find_flat_segments(data, threshold, min_duration_samples):
    """
    Detect stretches of signal that are not meaningfully changing.

    Parameters
    ----------
    data : np.array
        1D array of EEG samples for one channel
    threshold : float
        Maximum difference between consecutive samples to count as flat
    min_duration_samples : int
        Minimum number of consecutive flat samples to count as a segment

    Returns
    -------
    flat_mask : np.array
        Boolean array, True where signal is flat
    n_flat_segments : int
        Number of distinct flat segments found
    """
    # Find samples where the signal barely moves
    diff = np.abs(np.diff(data))
    is_flat = np.concatenate([[False], diff < threshold])

    # Only count stretches that last long enough
    flat_mask = np.zeros(len(data), dtype=bool)
    n_flat_segments = 0
    segment_start = None

    for i, flat in enumerate(is_flat):
        if flat and segment_start is None:
            segment_start = i
        elif not flat and segment_start is not None:
            duration = i - segment_start
            if duration >= min_duration_samples:
                flat_mask[segment_start:i] = True
                n_flat_segments += 1
            segment_start = None

    return flat_mask, n_flat_segments

#%%