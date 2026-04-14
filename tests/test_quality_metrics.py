"""Tests for nki_rs2_eeg.quality_metrics."""

from __future__ import annotations

import json
import pathlib

import mne
import numpy as np
import pytest

from nki_rs2_eeg import quality_metrics


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_raw(
    n_channels: int = 4,
    sfreq: float = 256.0,
    duration: float = 5.0,
) -> mne.io.RawArray:
    """Create a simple in-memory RawArray for testing."""
    n_samples = int(sfreq * duration)
    rng = np.random.default_rng(0)
    data = rng.standard_normal((n_channels, n_samples)) * 1e-6  # µV scale
    ch_names = [f"EEG{i:03d}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose=False)


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


def test_compute_metrics_keys() -> None:
    """compute_metrics should return all expected keys."""
    raw = _make_raw()
    result = quality_metrics.compute_metrics(raw)

    expected_keys = {
        "n_channels",
        "duration_s",
        "sampling_freq_hz",
        "channel_names",
        "channel_variance",
        "psd_band_power",
        "n_annotations",
        "bad_channels",
    }
    assert expected_keys <= result.keys()


def test_compute_metrics_n_channels() -> None:
    """n_channels should match the number of EEG channels in the recording."""
    n = 6
    raw = _make_raw(n_channels=n)
    result = quality_metrics.compute_metrics(raw)
    assert result["n_channels"] == n


def test_compute_metrics_sampling_freq() -> None:
    """sampling_freq_hz should match the info sfreq."""
    sfreq = 512.0
    raw = _make_raw(sfreq=sfreq)
    result = quality_metrics.compute_metrics(raw)
    assert result["sampling_freq_hz"] == sfreq


def test_compute_metrics_channel_variance_length() -> None:
    """channel_variance should have one entry per EEG channel."""
    n = 4
    raw = _make_raw(n_channels=n)
    result = quality_metrics.compute_metrics(raw)
    assert len(result["channel_variance"]) == n


def test_compute_metrics_channel_variance_positive() -> None:
    """All channel variances should be non-negative."""
    raw = _make_raw()
    result = quality_metrics.compute_metrics(raw)
    assert all(v >= 0.0 for v in result["channel_variance"])


def test_compute_metrics_psd_bands() -> None:
    """psd_band_power should contain all five standard bands."""
    raw = _make_raw()
    result = quality_metrics.compute_metrics(raw)
    assert set(result["psd_band_power"].keys()) == {
        "delta",
        "theta",
        "alpha",
        "beta",
        "gamma",
    }


def test_compute_metrics_no_annotations() -> None:
    """n_annotations should be 0 for a clean recording."""
    raw = _make_raw()
    result = quality_metrics.compute_metrics(raw)
    assert result["n_annotations"] == 0


def test_compute_metrics_bad_channels() -> None:
    """bad_channels should reflect channels marked bad in the raw object."""
    raw = _make_raw(n_channels=4)
    raw.info["bads"] = ["EEG001"]
    result = quality_metrics.compute_metrics(raw)
    assert result["bad_channels"] == ["EEG001"]


# ---------------------------------------------------------------------------
# process_subject (integration-style, using tmp_path)
# ---------------------------------------------------------------------------


def test_process_subject_saves_json(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """process_subject should write a JSON metrics file to the derivatives dir."""
    raw_dir = tmp_path / "raw"
    deriv_dir = tmp_path / "derivatives"
    raw_dir.mkdir()

    # Write a synthetic FIF file that process_subject can load
    raw = _make_raw()
    fif_path = raw_dir / "sub-9999_task-rest_eeg_raw.fif"
    raw.save(str(fif_path), overwrite=True, verbose=False)

    monkeypatch.setattr(quality_metrics, "RAW_DIR", raw_dir)
    monkeypatch.setattr(quality_metrics, "DERIVATIVES_DIR", deriv_dir)

    quality_metrics.process_subject("sub-9999")

    out = deriv_dir / "sub-9999_quality_metrics.json"
    assert out.exists(), "Expected metrics JSON file to be created."

    with out.open() as fh:
        data = json.load(fh)
    assert data["subject_id"] == "sub-9999"


def test_process_subject_missing_file(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """process_subject should raise FileNotFoundError when no EEG file exists."""
    monkeypatch.setattr(quality_metrics, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(quality_metrics, "DERIVATIVES_DIR", tmp_path / "derivatives")

    with pytest.raises(FileNotFoundError):
        quality_metrics.process_subject("sub-0000")
