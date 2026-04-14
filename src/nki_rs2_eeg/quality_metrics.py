"""Compute and store EEG quality metrics for each participant.

This script loads raw EEG data from NWB files in ``data/raw/``, computes a
set of standard quality metrics using MNE-Python, and saves the results as
a JSON summary and an annotated FIF file for each participant in
``data/derivatives/``.

Typical usage::

    python -m nki_rs2_eeg.quality_metrics

or, to process a single subject::

    python -m nki_rs2_eeg.quality_metrics --subject sub-0001
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
from typing import Any

import mne
import numpy as np

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_metrics(raw: mne.io.BaseRaw) -> dict[str, Any]:
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
        "n_channels": len(eeg_picks),
        "duration_s": raw.times[-1],
        "sampling_freq_hz": raw.info["sfreq"],
        "channel_names": [raw.ch_names[i] for i in eeg_picks],
        "channel_variance": channel_variance,
        "psd_band_power": psd_band_power,
        "n_annotations": len(raw.annotations),
        "bad_channels": list(raw.info["bads"]),
    }


def process_subject(subject_id: str) -> None:
    """Load raw EEG data for one participant and save quality metrics.

    The function expects a file named ``<subject_id>_task-rest_eeg.nwb``
    (or a ``.fif`` file with the same stem) inside ``data/raw/``.  Metrics
    are written to ``data/derivatives/<subject_id>_quality_metrics.json``.

    Args:
        subject_id: Subject identifier string (e.g. ``"sub-0001"``).

    Raises:
        FileNotFoundError: If no supported EEG file is found for the subject.
    """
    DERIVATIVES_DIR.mkdir(parents=True, exist_ok=True)

    # Prefer NWB, fall back to FIF
    nwb_path = RAW_DIR / f"{subject_id}_task-rest_eeg.nwb"
    fif_path = RAW_DIR / f"{subject_id}_task-rest_eeg_raw.fif"

    if nwb_path.exists():
        logger.info("Loading NWB file: %s", nwb_path)
        raw = mne.io.read_raw(str(nwb_path), preload=False)
    elif fif_path.exists():
        logger.info("Loading FIF file: %s", fif_path)
        raw = mne.io.read_raw_fif(str(fif_path), preload=False)
    else:
        raise FileNotFoundError(
            f"No EEG file found for {subject_id} in {RAW_DIR}. "
            "Expected a .nwb or .fif file."
        )

    logger.info("Computing quality metrics for %s …", subject_id)
    metrics = compute_metrics(raw)
    metrics["subject_id"] = subject_id

    out_path = DERIVATIVES_DIR / f"{subject_id}_quality_metrics.json"
    with out_path.open("w") as fh:
        json.dump(metrics, fh, indent=2)

    logger.info("Saved metrics to %s", out_path)


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
        nwb_files = list(RAW_DIR.glob("sub-*_task-rest_eeg.nwb"))
        fif_files = list(RAW_DIR.glob("sub-*_task-rest_eeg_raw.fif"))
        subject_ids = sorted(
            {p.name.split("_")[0] for p in nwb_files + fif_files}
        )
        if not subject_ids:
            logger.warning(
                "No EEG files found in %s. Nothing to process.", RAW_DIR
            )
            return
        for subject_id in subject_ids:
            try:
                process_subject(subject_id)
            except Exception:
                logger.exception("Failed to process %s", subject_id)


if __name__ == "__main__":
    main()
