"""Tests for nki_rs2_eeg.create_figures."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from nki_rs2_eeg import create_figures


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_METRICS = {
    "subject_id": "sub-0001",
    "n_channels": 4,
    "duration_s": 5.0,
    "sampling_freq_hz": 256.0,
    "channel_names": ["EEG000", "EEG001", "EEG002", "EEG003"],
    "channel_variance": [1.0, 2.0, 0.5, 3.0],
    "psd_band_power": {
        "delta": 0.1,
        "theta": 0.05,
        "alpha": 0.08,
        "beta": 0.03,
        "gamma": 0.01,
    },
    "n_annotations": 0,
    "bad_channels": ["EEG001"],
}


# ---------------------------------------------------------------------------
# plot_channel_variance
# ---------------------------------------------------------------------------


def test_plot_channel_variance_creates_file(tmp_path: pathlib.Path) -> None:
    """plot_channel_variance should write a PNG to the provided path."""
    out = tmp_path / "ch_var.png"
    create_figures.plot_channel_variance(_SAMPLE_METRICS, out)
    assert out.exists()
    assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# plot_band_power
# ---------------------------------------------------------------------------


def test_plot_band_power_creates_file(tmp_path: pathlib.Path) -> None:
    """plot_band_power should write a PNG to the provided path."""
    out = tmp_path / "band_power.png"
    create_figures.plot_band_power(_SAMPLE_METRICS, out)
    assert out.exists()
    assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# plot_group_summary
# ---------------------------------------------------------------------------


def test_plot_group_summary_creates_file(tmp_path: pathlib.Path) -> None:
    """plot_group_summary should write a PNG to the provided path."""
    all_metrics = [
        {**_SAMPLE_METRICS, "subject_id": f"sub-{i:04d}"}
        for i in range(3)
    ]
    out = tmp_path / "group_summary.png"
    create_figures.plot_group_summary(all_metrics, out)
    assert out.exists()
    assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# create_figures_for_subject (integration-style)
# ---------------------------------------------------------------------------


def test_create_figures_for_subject(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """create_figures_for_subject should produce both per-subject PNGs."""
    deriv_dir = tmp_path / "derivatives"
    figures_dir = tmp_path / "figures"
    deriv_dir.mkdir()

    mf = deriv_dir / "sub-0001_quality_metrics.json"
    with mf.open("w") as fh:
        json.dump(_SAMPLE_METRICS, fh)

    monkeypatch.setattr(create_figures, "DERIVATIVES_DIR", deriv_dir)
    monkeypatch.setattr(create_figures, "FIGURES_DIR", figures_dir)

    create_figures.create_figures_for_subject("sub-0001")

    assert (figures_dir / "sub-0001_channel_variance.png").exists()
    assert (figures_dir / "sub-0001_band_power.png").exists()


def test_create_figures_for_subject_missing_metrics(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """create_figures_for_subject should raise FileNotFoundError if metrics are absent."""
    monkeypatch.setattr(create_figures, "DERIVATIVES_DIR", tmp_path / "derivatives")
    monkeypatch.setattr(create_figures, "FIGURES_DIR", tmp_path / "figures")

    with pytest.raises(FileNotFoundError):
        create_figures.create_figures_for_subject("sub-9999")
