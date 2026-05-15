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

# ============================================================
# PER-CHANNEL METRICS (single participant)
# ============================================================

def compute_channel_metrics(eeg_data: np.ndarray) -> pd.DataFrame:
    """Compute quality metrics for each channel for a single participant.

    Parameters
    ----------
    eeg_data : np.ndarray
        2D array shaped (channels, samples)

    Returns
    -------
    pd.DataFrame
        One row per channel with all quality metrics as columns
    """
    n_channels, n_samples = eeg_data.shape
    min_flat_samples = int(FLAT_MIN_DURATION_SEC * FS)
    records = []

    for ch_idx in range(n_channels):
        ch_data = eeg_data[ch_idx, :]
        ch_name = CHANNEL_NAMES[ch_idx] if ch_idx < len(CHANNEL_NAMES) else f"ch_{ch_idx}"

        # Basic stats
        variance = np.var(ch_data)
        peak_to_peak = np.ptp(ch_data)
        kurtosis = stats.kurtosis(ch_data)

        # Flat segment detection
        flat_mask, n_flat_segments = find_flat_segments(
            ch_data, FLAT_THRESHOLD, min_flat_samples
        )
        pct_flat = 100 * np.sum(flat_mask) / n_samples

        # Frequency-band power
        muscle_power = compute_band_power(ch_data, MUSCLE_FREQ_BAND, FS)
        drift_power = compute_band_power(ch_data, DRIFT_FREQ_BAND, FS)
        line_noise_power = compute_band_power(
            ch_data, (LINE_NOISE_HZ - 1, LINE_NOISE_HZ + 1), FS
        )

        # Flag as bad channel based on thresholds from config
        is_bad = (
            variance < VARIANCE_MIN or
            variance > VARIANCE_MAX or
            peak_to_peak > PEAK_TO_PEAK_MAX or
            kurtosis > KURTOSIS_MAX or
            pct_flat > 10 or
            muscle_power > MUSCLE_POWER_MAX or
            line_noise_power > LINE_NOISE_POWER_MAX
        )

        records.append({
            "channel": ch_name,
            "channel_idx": ch_idx,
            "variance": variance,
            "peak_to_peak": peak_to_peak,
            "kurtosis": kurtosis,
            "pct_flat": pct_flat,
            "n_flat_segments": n_flat_segments,
            "muscle_power": muscle_power,
            "drift_power": drift_power,
            "line_noise_power": line_noise_power,
            "is_bad_channel": is_bad
        })

    return pd.DataFrame(records)


# ============================================================
# PER-PARTICIPANT METRICS
# ============================================================

def compute_participant_metrics(participant_id, eeg_data):
    """
    Compute a single-row summary of data quality for one participant.

    Parameters
    ----------
    participant_id : str
        Participant identifier
    eeg_data : np.array
        2D array shaped (channels, samples)

    Returns
    -------
    dict
        Summary metrics for this participant
    """
    n_channels, n_samples = eeg_data.shape
    channel_df = compute_channel_metrics(eeg_data)

    n_bad_channels = channel_df["is_bad_channel"].sum()

    # Build an artifact mask: a sample is bad if ANY channel is bad at that moment
    # Here we use variance in sliding windows as a simple artifact detector
    artifact_mask = compute_artifact_mask(eeg_data)
    n_clean_samples = np.sum(~artifact_mask)
    pct_clean = 100 * n_clean_samples / n_samples

    is_excluded = (
        n_bad_channels > MAX_BAD_CHANNELS or
        pct_clean < MIN_CLEAN_DATA_PCT
    )

    return {
        "participant_id": participant_id,
        "n_samples": n_samples,
        "n_channels": n_channels,
        "n_bad_channels": n_bad_channels,
        "pct_bad_channels": 100 * n_bad_channels / n_channels,
        "pct_clean_data": pct_clean,
        "mean_variance": channel_df["variance"].mean(),
        "mean_kurtosis": channel_df["kurtosis"].mean(),
        "mean_line_noise": channel_df["line_noise_power"].mean(),
        "mean_muscle_power": channel_df["muscle_power"].mean(),
        "is_excluded": is_excluded,
        "exclusion_reason": _get_exclusion_reason(n_bad_channels, pct_clean)
    }


def _get_exclusion_reason(n_bad_channels, pct_clean):
    """
    Return a human-readable string explaining why a participant was excluded,
    or None if they were not excluded.
    """
    reasons = []
    if n_bad_channels > MAX_BAD_CHANNELS:
        reasons.append(f"too many bad channels ({n_bad_channels})")
    if pct_clean < MIN_CLEAN_DATA_PCT:
        reasons.append(f"insufficient clean data ({pct_clean:.1f}%)")
    return "; ".join(reasons) if reasons else None


# ============================================================
# SAMPLE-LEVEL METRICS ACROSS PARTICIPANTS
# ============================================================

def compute_artifact_mask(eeg_data: np.ndarray) -> np.ndarray:
    """Produce a 1D boolean array the length of the recording where
    True = artifact present at that sample (any channel flagged).

    Parameters:
    ----------
    eeg_data : np.ndarray
        2D array shaped (channels, samples)

    Returns:
    ------- 
    np.ndarray
        Boolean array of shape (samples,)
    """
    n_channels, n_samples = eeg_data.shape
    artifact_mask = np.zeros(n_samples, dtype=bool)

    for ch_idx in range(n_channels):
        ch_data = eeg_data[ch_idx, :]

        # Flag samples inside windows where variance exceeds threshold
        for start in range(0, n_samples - WINDOW_SIZE_SAMPLES, WINDOW_STEP_SAMPLES):
            end = start + WINDOW_SIZE_SAMPLES
            window_variance = np.var(ch_data[start:end])
            if window_variance > VARIANCE_MAX or window_variance < VARIANCE_MIN:
                artifact_mask[start:end] = True

        # Also flag flat samples directly
        flat_mask, _ = find_flat_segments(
            ch_data, FLAT_THRESHOLD, int(FLAT_MIN_DURATION_SEC * FS)
        )
        artifact_mask |= flat_mask

    return artifact_mask


#%%