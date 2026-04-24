"""Compute and store EEG quality metrics for each participant.

This script loads raw EEG data from NWB files in ``data/raw/``, computes a
set of standard quality metrics using MNE-Python, and saves the results as
a JSON summary and an annotated FIF file for each participant in
``data/derivatives/``.

Typical usage::

    python -m nki_rs2_eeg.quality_metrics

or, to process a single subject::

"""

from __future__ import annotations

import argparse
from curses import raw
import json
import logging
import os
import pathlib
from typing import Any

import mne
import numpy as np
import pandas as pd
import pynwb
import pyprep

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (resolved relative to the repository root)
# ---------------------------------------------------------------------------


def _find_repo_root(start: pathlib.Path) -> pathlib.Path:
    """Walk up from *start* until a ``pyproject.toml`` is found.

    Args:
        start: Starting directory for the upward search.

    Returns:
        The first directory that contains ``pyproject.toml``.

    Raises:
        FileNotFoundError: If no ``pyproject.toml`` is found up to the
            filesystem root.
    """
    current = start.resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise FileNotFoundError(
        "Could not find repository root (no pyproject.toml found)."
    )


_REPO_ROOT = _find_repo_root(pathlib.Path(__file__).parent)
RAW_DIR = _REPO_ROOT / "data" / "raw"
DERIVATIVES_DIR = _REPO_ROOT / "data" / "derivatives"
CAP_DIR = _REPO_ROOT / "data" / "caps"
SESSION_ID = "MOBI2C"
TASK_ID = "passivepresent"
RUN_ID = "01"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def custom_serializer(obj):
    if hasattr(obj, '__dict__'):
        return obj.__dict__    # Convert custom classes to dicts
    return str(obj)            # Fallback to string for everything else

def annotate_blinks(raw: mne.io.Raw, ch_name: list[str] = ["Fp1", "Fp2"]) -> mne.Annotations:
    """Annotate the blinks in the EEG signal.

    Args:
        raw (mne.io.Raw): The raw EEG data in mne format.
        ch_name (list[str]): The channels to use for the EOG. Default is
                            ["Fp1", "Fp2"]. I would suggest to use the
                            channels that are the most frontal (just above
                            the eyes). In the case of an EGI system the
                            channels would be "E25" and "E8".

    Returns:
        mne.Annotations: The annotations object containing the blink events.
    """
    eog_epochs = mne.preprocessing.create_eog_epochs(raw, ch_name=ch_name)
    blink_annotations = mne.annotations_from_events(
        eog_epochs.events,
        raw.info["sfreq"],
        event_desc={eog_epochs.events[0, 2]: "blink"},
    )
    return blink_annotations


def annotate_muscle(raw: mne.io.Raw) -> mne.Annotations:
    """Annotate muscle artifacts in the EEG signal.

    Args:        raw (mne.io.Raw): The raw EEG data in mne format.
    Returns:        mne.Annotations: The annotations object containing the muscle artifact events.
    """
    muscle_annotations, _ = mne.preprocessing.annotate_muscle_zscore(raw, 
        threshold=6, # this needs to be calibrated for the entire dataset
        ch_type='eeg', 
        min_length_good=0.1, 
        filter_freq=(95, 120), 
        )

    return muscle_annotations


def compute_autocorrelation(data: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute autocorrelation for a 1D signal.
    
    Parameters:
    -----------
    data : array, shape (n_samples,)
        EEG data for single channel
    max_lag : int
        Maximum lag to compute (in samples)
    
    Returns:
    --------
    autocorr : array, shape (max_lag+1,)
        Autocorrelation values from lag 0 to max_lag
    """
    # Normalize the signal
    data_normalized = (data - np.mean(data)) / np.std(data)
    
    # Compute autocorrelation
    autocorr = np.correlate(data_normalized, data_normalized, mode='full')
    autocorr = autocorr / autocorr[len(autocorr)//2]  # Normalize by lag 0
    
    # Take only positive lags
    center = len(autocorr) // 2
    autocorr = autocorr[center:center + max_lag + 1]
    
    return autocorr


def sliding_window_autocorr(channel_data: np.ndarray, window_size: int, hop_size: int, max_lag: int) -> np.ndarray:
    """Compute autocorrelation with sliding windows.
    
    Parameters:
    -----------
    channel_data : array, shape (n_samples,)
        EEG data for single channel
    window_size : int
        Window size in samples
    hop_size : int
        How many samples to slide the window
    max_lag : int
        Maximum lag to compute
    
    Returns:
    --------
    autocorr_windows : array, shape (n_windows, max_lag+1)
        Autocorrelation for each window
    """
    n_samples = len(channel_data)
    n_windows = (n_samples - window_size) // hop_size + 1
    
    autocorr_windows = []
    
    for i in range(n_windows):
        start = i * hop_size
        end = start + window_size
        
        if end > n_samples:
            break
            
        window = channel_data[start:end]
        autocorr = compute_autocorrelation(window, max_lag)
        autocorr_windows.append(autocorr)
    
    return np.array(autocorr_windows)


def assess_channel_quality(
    autocorr_windows: np.ndarray,
    threshold: float = 0.3,
    lag_start: int = 10,
) -> tuple[bool, float]:
    """Assess signal quality based on autocorrelation.
    
    Parameters:
    -----------
    autocorr_windows : array, shape (n_windows, max_lag+1)
        Autocorrelation values
    threshold : float
        Threshold for "bad" autocorrelation
    lag_start : int
        Start checking from this lag (skip lag 0 which is always 1.0)
    
    Returns:
    --------
    is_clean : bool
        True if channel is clean
    max_autocorr : float
        Maximum autocorrelation value (excluding early lags)
    """
    # Average across windows
    mean_autocorr = np.mean(autocorr_windows, axis=0)
    
    # Check autocorrelation after lag_start (ignore very short lags)
    max_autocorr = np.max(np.abs(mean_autocorr[lag_start:]))
    
    is_clean = max_autocorr < threshold
    
    return is_clean, max_autocorr


def analyze_autocorr_quality(eeg_data: np.ndarray, fs: float, window_seconds: float = 2, hop_seconds: float = 1, 
                        max_lag_ms: float = 500, threshold: float = 0.3) -> dict[str, Any]:
    """Analyze EEG signal quality across all channels.
    
    Parameters:
    -----------
    eeg_data : array, shape (n_channels, n_samples)
        EEG data
    fs : float
        Sampling frequency in Hz
    window_seconds : float
        Window size in seconds
    hop_seconds : float
        Hop size in seconds
    max_lag_ms : float
        Maximum lag in milliseconds
    threshold : float
        Autocorrelation threshold for quality
    
    Returns:
    --------
    results : dict
        Dictionary with quality metrics
    """
    n_channels, n_samples = eeg_data.shape
    
    # Convert to samples
    window_size = int(window_seconds * fs)
    hop_size = int(hop_seconds * fs)
    max_lag = int(max_lag_ms * fs / 1000)
    
    # Store results
    channel_quality = []
    max_autocorrs = []
    all_autocorrs = []
    
    print(f"Analyzing {n_channels} channels...")
    
    for ch in range(n_channels):
        # Compute sliding window autocorrelation
        autocorr_windows = sliding_window_autocorr(
            eeg_data[ch, :], window_size, hop_size, max_lag
        )
        
        # Assess quality
        is_clean, max_ac = assess_channel_quality(
            autocorr_windows, threshold=threshold
        )
        
        channel_quality.append(is_clean)
        max_autocorrs.append(max_ac)
        
        # Store mean autocorr for this channel
        mean_autocorr = np.mean(autocorr_windows, axis=0)
        all_autocorrs.append(mean_autocorr)
    
    # Summarize across channels
    results = {
        'channel_quality': str(channel_quality),
        'max_autocorrs': max_autocorrs,
        'all_autocorrs': all_autocorrs,
        'bad_channels_ac': np.where(~np.array(channel_quality))[0].tolist(),
        'percent_clean': (np.mean(channel_quality) * 100).tolist(),
        'n_bad_channels_ac': int(np.sum(~np.array(channel_quality))),
        'worst_autocorr': float(np.max(max_autocorrs)),
        'lags_ms': (np.arange(max_lag + 1) * 1000 / fs).tolist()
    }
    
    return results



def compute_simple_metrics(raw: mne.io.BaseRaw) -> dict[str, Any]:
    """Compute quality metrics for a continuous EEG recording.

    Args:
        raw: A preloaded :class:`mne.io.BaseRaw` instance.

    Returns:
        A dictionary with the following keys:

        - ``n_channels``: number of EEG channels.
        - ``duration_s``: recording duration in seconds.
        - ``sampling_freq_hz``: sampling frequency in Hz.
        - ``channel_variance``: per-channel variance (µV²) as a list.
        - ``channel_names``: EEG channel names corresponding to
          ``channel_variance``.
        - ``psd_band_power``: mean power in standard frequency bands (µV²/Hz).
        - ``n_annotations``: number of annotations already present in the
          recording (e.g. bad segments).
        - ``bad_channels``: channels already marked as bad.
    """
    raw.load_data()
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    data, _ = raw[eeg_picks]

    # Per-channel variance (µV²)
    channel_variance: list[float] = (np.var(data, axis=1) * 1e12).tolist()

    # Band power via Welch PSD
    bands = {
        "delta": (1.0, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 80.0),
    }
    spectrum = raw.compute_psd(picks="eeg", fmin=1.0, fmax=80.0)
    psds, freqs = spectrum.get_data(return_freqs=True)

    psd_band_power: dict[str, float] = {}
    for band_name, (fmin, fmax) in bands.items():
        idx = np.where((freqs >= fmin) & (freqs < fmax))[0]
        psd_band_power[band_name] = float(np.mean(psds[:, idx]))

    return {
        "channel_variance": channel_variance,
        "psd_band_power": psd_band_power,
        "n_annotations": len(raw.annotations),
        "bad_channels": list(raw.info["bads"]),
    }

def compute_all_qc_matrics(raw: mne.io.BaseRaw) -> dict[str, Any]:
    """Compute all quality metrics for a continuous EEG recording.

    Args:
        raw: A preloaded :class:`mne.io.BaseRaw` instance.
    Returns:
        A dictionary with all quality metrics.
    """
    metrics = compute_simple_metrics(raw)
    ac_metrics = analyze_autocorr_quality(
        raw.get_data(picks="eeg"), raw.info["sfreq"])
    metrics.update(ac_metrics)
    return metrics



def clean_data(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Apply basic cleaning steps to the raw EEG data.

    This function performs the following steps:
    1. Apply a bandpass filter to remove slow drifts and high-frequency noise.
    2. Use the PREP pipeline to identify and interpolate bad channels, and to
       re-reference the data robustly.
    3. Annotate blinks and muscle artifacts using MNE's built-in functions.

    Args:
        raw: A preloaded :class:`mne.io.BaseRaw` instance.
    """
    clean_data_path = DERIVATIVES_DIR / f"{raw.subject_info['subject_id']}_ses-{raw.subject_info['session_id']}_task-{raw.subject_info['task_id']}_run-{raw.subject_info['run_id']}_cleaned.fif"
    if clean_data_path.exists():
        logger.info("Cleaned data already exists at %s, loading it.", clean_data_path)
        return mne.io.read_raw_fif(clean_data_path, preload=True)
    
    else:
        
        # add a high frequency bandpass filter to remove muscle artifacts
        raw.filter(l_freq=1.0, h_freq=50.0) # only keeping frequencies between 1-50 Hz
        
        # Preprocessing the EEG data
        prep_params = {
                "ref_chs": "eeg",
                "reref_chs": "eeg",
                "line_freqs": np.arange(60, raw.info["sfreq"] / 2, 60),
            }
        # these params set up the robust reference  - i.e. median of all channels and interpolate bad channels
        prep = pyprep.PrepPipeline(raw, montage=raw.get_montage(), channel_wise=True, prep_params=prep_params)
        print("STARTING preprocessing")
        prep_output = prep.fit()
        raw_cleaned = prep_output.raw_eeg

        print("DONE with preprocessing")
        # Save the cleaned data for future use
        raw_cleaned.save(clean_data_path, overwrite=True)
        logger.info("Saved cleaned raw data to %s", clean_data_path)

        return raw_cleaned, prep_output
    

def process_subject(subject_id: str, session_id: str = SESSION_ID , task_id: str = TASK_ID, run_id: str = RUN_ID) -> None:
    """Load raw EEG data for one participant and save quality metrics.

    The function expects a file named ``<subject_id>_task-rest_eeg.nwb``
    (or a ``.fif`` file with the same stem) inside ``data/raw/``.  Metrics
    are written to ``data/derivatives/<subject_id>_quality_metrics.json``.

    Args:
        subject_id: Subject identifier string (e.g. ``"sub-0001"``).
        session_id: Session identifier string (e.g. ``"MOBI2C"``).
        task_id: Task identifier string (e.g. ``"passivepresent"``).
        run_id: Run identifier string (e.g. ``"01"``).

    Raises:
        FileNotFoundError: If no supported EEG file is found for the subject.
    """
    DERIVATIVES_DIR.mkdir(parents=True, exist_ok=True)

    # Prefer NWB, fall back to FIF
    nwb_path = (
        RAW_DIR
        / f"{subject_id}"
        / f"ses-{session_id}"
        / f"{subject_id}_ses-{session_id}_task-{task_id}_run-{run_id}_MoBI.nwb"
    )

    if nwb_path.exists():
        logger.info("Loading NWB file: %s", nwb_path)

        with pynwb.NWBHDF5IO(str(nwb_path), "r") as io:
            nwb = io.read()
            e_series = nwb.acquisition["ElectricalSeries"]
            stim_series = nwb.acquisition["StimLabels"]
            electrode_info = nwb.electrodes.to_dataframe().copy()

            # Load 
            df = pd.DataFrame(e_series.data[()], columns=e_series.description.split(","))
            df["timestamps"] = e_series.timestamps[()]

            # Stim mapping (vectorized-ish, no list comp)
            stim_keys = stim_series.data[()].astype(str).flatten()
            stim_times = stim_series.timestamps[()]
        stim = dict(zip(stim_keys, stim_times))

        # Event filtering (use .between for clarity + speed)
        event_df = df[df['timestamps'].between(
            stim['Onset Movie'], stim['Offset Movie'])].copy()
        # CReate MNE Raw object
        info = mne.create_info(
            ch_names=list(event_df.columns[:-1]),
            sfreq=1 / event_df['timestamps'].diff().mean(),
            ch_types='eeg'
        )
        event_df = event_df.drop(columns=['timestamps'])

        # Get montage file based on cap type
        cap_types = pd.read_csv(os.path.join(CAP_DIR, 'captypes_clean.csv'))
        subject_cap_type = cap_types.loc[
            cap_types['a_number'] == subject_id[4:], 'cap_type'
        ].values[0]
        if subject_cap_type.startswith("RNP"):
            montage_file = os.path.join(CAP_DIR, 'R-Net for BrainAmp_RNP-BA', subject_cap_type)
        elif subject_cap_type.startswith("BC-MR"):
            montage_file = os.path.join(CAP_DIR, subject_cap_type)
        else:
            raise ValueError(f"Unknown cap type: {subject_cap_type}")
        montage = mne.channels.read_custom_montage(montage_file)
        info.set_montage(montage, on_missing='ignore')

        raw = mne.io.RawArray(
            event_df.T * 1e-6, info=info
        )  # multiplying by 1e-6 converts to volts
        raw.subject_info = {"subject_id": subject_id, "session_id": session_id, "task_id": task_id, "run_id": run_id}

        

        imp_vars = {}
        #expand the allImpedances column to get 3 variables for each channel
        #imp_vars['impedance1'] = {ch: int(imp1) for ch, imp1 in zip(raw.info['ch_names'], [electrode_info.allImpedances[i][0] for i in range(len(electrode_info))])}
        #imp_vars['impedance2'] = {ch: int(imp2) for ch, imp2 in zip(raw.info['ch_names'], [electrode_info.allImpedances[i][1] for i in range(len(electrode_info))])}
        #imp_vars['impedance3'] = {ch: int(imp3) for ch, imp3 in zip(raw.info['ch_names'], [electrode_info.allImpedances[i][2] for i in range(len(electrode_info))])}
        imp_vars['mean_impedance'] = {ch: np.nanmean(electrode_info.allImpedances[i]) for i, ch in enumerate(raw.info['ch_names'])}

    else:
        raise FileNotFoundError(
            f"No EEG file found for {subject_id} in {RAW_DIR}. "
            "Expected a .nwb or .fif file."
        )

    logger.info("Computing quality metrics for %s …", subject_id)
    
    all_metrics = compute_all_qc_matrics(raw)
    all_metrics["subject_id"] = subject_id
    all_metrics = {"pre_" + k: v for k, v in all_metrics.items()}

    raw_cleaned, prep_output = clean_data(raw)

    all_metrics['noisy_channels_before_reref'] = prep_output.noisy_channels_original
    all_metrics['noisy_channels_before_interp'] = prep_output.noisy_channels_before_interpolation
    all_metrics['bad_after_reref_before_interp'] = prep_output.bad_before_interpolation
    all_metrics['noisy_channels_after_interp'] = prep_output.noisy_channels_after_interpolation
    all_metrics['interpolated_channels'] = prep_output.interpolated_channels
    all_metrics['noisy_after_interp'] = prep_output.still_noisy_channels


    post_metrics = compute_all_qc_matrics(raw_cleaned)
    post_metrics = {"post_" + k: v for k, v in post_metrics.items()}

    all_metrics.update(post_metrics)
    all_metrics.update(imp_vars)

    #return all_metrics, prep_output
    json_out_path = DERIVATIVES_DIR / f"{subject_id}_ses-{session_id}_task-{task_id}_run-{run_id}_qc_metrics.json"
    with json_out_path.open("w") as fh:
        json.dump(all_metrics, fh, indent=2, default=custom_serializer)
    logger.info("Saved metrics to %s", json_out_path)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute EEG quality metrics for NKI RS2 participants.",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help=(
            "Process a single subject (e.g. sub-0001). "
            "If omitted, all subjects found in data/raw/ are processed."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging verbosity (default: INFO).",
    )
    return parser


def main() -> None:
    """Run the quality-metrics pipeline from the command line."""
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.subject:
        process_subject(args.subject)
    else:
        # Discover all subjects from available raw files
        glob_pattern = (
            f'sub-*/ses-{SESSION_ID}/'
            f'sub-*_ses-{SESSION_ID}_task-{TASK_ID}_run-{RUN_ID}_MoBI.nwb'
        )
        nwb_files = list(RAW_DIR.glob(glob_pattern))

        subject_ids = sorted(
            {p.name.split("_")[0] for p in nwb_files}
        )
        if not subject_ids:
            logger.warning(
                "No EEG files found in %s. Nothing to process.", RAW_DIR
            )
            return
        for subject_id in subject_ids:
            try:
                process_subject(subject_id, SESSION_ID, TASK_ID, RUN_ID)
            except Exception:
                logger.exception("Failed to process %s", subject_id)


if __name__ == "__main__":
    main()
