"""Generate figures to visualise EEG data quality metrics.

This script reads the JSON metrics files produced by
:mod:`nki_rs2_eeg.quality_metrics` from ``data/derivatives/`` and writes
summary figures (PNG) to the ``figures/`` directory.

Typical usage::

    python -m nki_rs2_eeg.create_figures

or, to plot a single subject::

    python -m nki_rs2_eeg.create_figures --subject sub-0001
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")  # Non-interactive backend suitable for headless runs

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

FIGURES_DIR = _REPO_ROOT / "figures"

# Canonical band order for consistent bar-chart plotting
_BAND_ORDER = ["delta", "theta", "alpha", "beta", "gamma"]
# SUBJECT_LIST is derived from the derivatives directory structure


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_channel_variance(
    metrics: dict,
    out_path: pathlib.Path,
) -> None:
    """Plot per-channel variance for a single participant.

    Args:
        metrics: Quality-metrics dictionary as returned by
            :func:`nki_rs2_eeg.quality_metrics.compute_metrics`.
        out_path: Destination path for the PNG figure.
    """
    channel_names: list[str] = metrics["channel_names"]
    variances: list[float] = metrics["channel_variance"]
    subject_id: str = metrics.get("subject_id", "unknown")
    bad_channels: list[str] = metrics.get("bad_channels", [])

    colors = [
        "firebrick" if ch in bad_channels else "steelblue"
        for ch in channel_names
    ]

    fig, ax = plt.subplots(figsize=(max(8, len(channel_names) * 0.3), 4))
    ax.bar(range(len(channel_names)), variances, color=colors)
    ax.set_xticks(range(len(channel_names)))
    ax.set_xticklabels(channel_names, rotation=90, fontsize=6)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Variance (µV²)")
    ax.set_title(f"Per-channel variance — {subject_id}")
    if bad_channels:
        ax.legend(
            handles=[
                matplotlib.patches.Patch(color="firebrick", label="Bad channel"),
                matplotlib.patches.Patch(color="steelblue", label="Good channel"),
            ]
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved channel-variance figure: %s", out_path)


def plot_band_power(
    metrics: dict,
    out_path: pathlib.Path,
) -> None:
    """Plot mean PSD band power for a single participant.

    Args:
        metrics: Quality-metrics dictionary as returned by
            :func:`nki_rs2_eeg.quality_metrics.compute_metrics`.
        out_path: Destination path for the PNG figure.
    """
    subject_id: str = metrics.get("subject_id", "unknown")
    band_power: dict[str, float] = metrics.get("psd_band_power", {})

    bands = [b for b in _BAND_ORDER if b in band_power]
    powers = [band_power[b] for b in bands]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(bands, powers, color="mediumseagreen")
    ax.set_xlabel("Frequency band")
    ax.set_ylabel("Mean power (µV²/Hz)")
    ax.set_title(f"PSD band power — {subject_id}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved band-power figure: %s", out_path)


def plot_group_summary(
    all_metrics: list[dict],
    out_path: pathlib.Path,
) -> None:
    """Plot a group-level summary of mean alpha power across participants.

    Args:
        all_metrics: List of per-subject metrics dictionaries.
        out_path: Destination path for the PNG figure.
    """
    subject_ids = [metrics.get("subject_id", f"sub-{i:04d}") for i, metrics in enumerate(all_metrics)]
    alpha_powers = [
        metrics.get("psd_band_power", {}).get("alpha", float("nan"))
        for metrics in all_metrics
    ]

    fig, ax = plt.subplots(figsize=(max(6, len(subject_ids) * 0.5), 4))
    x = np.arange(len(subject_ids))
    ax.bar(x, alpha_powers, color="mediumpurple")
    ax.set_xticks(x)
    ax.set_xticklabels(subject_ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean alpha power (µV²/Hz)")
    ax.set_title("Group summary — mean alpha power")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved group-summary figure: %s", out_path)


def create_figures_for_subject(subject_id: str) -> None:
    """Read metrics and produce figures for a single participant.

    Args:
        subject_id: Subject identifier string (e.g. ``"sub-0001"``).

    Raises:
        FileNotFoundError: If the metrics JSON file does not exist.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = DERIVATIVES_DIR / f"{subject_id}_quality_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {metrics_path}. "
            "Run quality_metrics.py first."
        )

    with metrics_path.open() as fh:
        metrics = json.load(fh)

    plot_channel_variance(
        metrics,
        FIGURES_DIR / f"{subject_id}_channel_variance.png",
    )
    plot_band_power(
        metrics,
        FIGURES_DIR / f"{subject_id}_band_power.png",
    )


def load_all_metrics() -> pd.DataFrame:
    """Load all per-subject metrics JSON files into a DataFrame."""
    metrics_files = sorted([s for s in DERIVATIVES_DIR.glob("sub-*.json")])

    qc_results = []
    for qc_file in metrics_files:
        if os.path.exists(qc_file):
            with open(qc_file, 'r') as f:
                qc_data = json.load(f)
                qc_results.append(qc_data)

    # Convert the list of QC results to a DataFrame
    qc_df = pd.DataFrame(qc_results)
    qc_df.set_index('pre_subject_id', inplace=True)
    return qc_df


def create_group_impedance_heatmap(qc_df: pd.DataFrame, out_path: pathlib.Path) -> None:
    """Create a heatmap of mean impedance values across channels and subjects."""
    
    logger.info("Saved impedance heatmap figure: %s", out_path)
    # make pre_subject_id the index
    qc_df.set_index('pre_subject_id', inplace=True)
    imps = qc_df.mean_impedance
    imps.apply(pd.Series)
    imps_df = pd.DataFrame(imps.tolist(), index = imps.index)
    plt.figure(figsize=(10, 6))
    sns.heatmap(imps_df, cmap ="viridis", annot=False)
    plt.title("Mean Impedance (kOhms): Subject x Channel")
    plt.tight_layout()
    plt.show()
    # Save the figure
    out_path = FIGURES_DIR / "group_impedance_heatmap.png"
    plt.savefig(out_path, dpi=150)

    # Add 30 rows of synthetic data for demonstration
    #synthetic_data = pd.DataFrame(
    #    np.random.rand(30, imps_df.shape[1]) * 80 + 10,  # Random values between 5 and 15
    #    columns=imps_df.columns,
    #    index=[f'sub-{i:04d}' for i in range(1001, 1031)]
    #)
    #imps_df = pd.concat([imps_df, synthetic_data])


qc_df = load_all_metrics()
pre_psd = qc_df.pre_psd_band_power
pre_psd.apply(pd.Series)
pre_psd_df = pd.DataFrame(pre_psd.to_list(), index=pre_psd.index)
post_psd = qc_df.post_psd_band_power
post_psd.apply(pd.Series)
post_psd_df = pd.DataFrame(post_psd.to_list(), index=post_psd.index)
# plot pre and post psd curves for all subjects in the same figure. 

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
for subject_id, row in pre_psd_df.iterrows():
    freqs = row['freqs']
    psd = row['psd']
    ax[0].plot(freqs, psd, label=subject_id)
for subject_id, row in post_psd_df.iterrows():
    freqs = row['freqs']
    psd = row['psd']
    ax[1].plot(freqs, psd, label=subject_id, linestyle='--', ax=ax[1])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Mean Power (µV²/Hz)')
plt.title('Mean PSD Curves for All Subjects (Pre: solid, Post: dashed)')


# Plot each subject's PSD curve
plt.figure(figsize=(10, 6))
for subject_id, row in pre_psd_df.iterrows():
    freqs = row['freqs']
    psd = row['psd']
    plt.plot(freqs, psd, label=subject_id)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Mean Power (µV²/Hz)')
plt.title('Mean PSD Curves for All Subjects')
plt.xlim(1, 80)
plt.ylim(0, pre_psd_df['psd'].apply(max).max() * 1.1)  # Set y-axis limit slightly above max for better visualization
plt


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate EEG quality figures for NKI RS2 participants.",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help=(
            "Generate figures for a single subject (e.g. sub-0001). "
            "If omitted, all subjects with a metrics file are processed and "
            "a group summary figure is also created."
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
    """Run the figure-generation pipeline from the command line."""
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.subject:
        create_figures_for_subject(args.subject)
    else:
        metrics_files = sorted(DERIVATIVES_DIR.glob("sub-*_quality_metrics.json"))
        if not metrics_files:
            logger.warning(
                "No metrics files found in %s. "
                "Run quality_metrics.py first.",
                DERIVATIVES_DIR,
            )
            return

        all_metrics: list[dict] = []
        for mf in metrics_files:
            subject_id = mf.name.replace("_quality_metrics.json", "")
            try:
                create_figures_for_subject(subject_id)
                with mf.open() as fh:
                    all_metrics.append(json.load(fh))
            except Exception:
                logger.exception("Failed to create figures for %s", subject_id)

        if len(all_metrics) > 1:
            plot_group_summary(
                all_metrics,
                FIGURES_DIR / "group_alpha_power_summary.png",
            )


if __name__ == "__main__":
    main()
