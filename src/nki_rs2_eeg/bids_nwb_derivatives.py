"""Write preprocessed EEG as NWB files inside a BIDS-Derivatives dataset.

Layout produced::

    <bids_root>/derivatives/<pipeline>/
        dataset_description.json
        sub-XX/ses-YY/eeg/
            sub-XX_ses-YY_task-ZZ_desc-preproc_eeg.nwb
            sub-XX_ses-YY_task-ZZ_desc-preproc_eeg.json
            sub-XX_ses-YY_task-ZZ_desc-preproc_channels.tsv
            sub-XX_ses-YY_task-ZZ_desc-preproc_events.tsv

The raw BIDS dataset is left untouched; each derivative file points back to
its raw source through a ``bids:raw:`` URI in its sidecar.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from hdmf.backends.hdf5.h5_utils import H5DataIO
from mne_bids import BIDSPath, get_bids_path_from_fname
from pynwb import NWBHDF5IO
from pynwb.ecephys import ElectricalSeries

BIDS_VERSION = "1.10.0"


# ---------------------------------------------------------------------------
# Step 1: the derivative dataset (once per pipeline)
# ---------------------------------------------------------------------------
def init_derivative_dataset(
    bids_root: os.PathLike,
    pipeline: str,
    version: str,
    description: str,
    code_url: str | None = None,
) -> Path:
    """Create derivatives/<pipeline>/ with a derivative dataset_description.json.

    ``DatasetType: derivative`` and ``GeneratedBy`` are what make this a BIDS
    derivative dataset. ``DatasetLinks`` defines the ``raw`` alias used by the
    ``bids:raw:...`` URIs in each file's ``Sources`` field.
    """
    bids_root = Path(bids_root).resolve()
    deriv_root = bids_root / "derivatives" / pipeline
    deriv_root.mkdir(parents=True, exist_ok=True)

    generated_by = {"Name": pipeline, "Version": version, "Description": description}
    if code_url:
        generated_by["CodeURL"] = code_url

    dataset_description = {
        "Name": f"{pipeline} preprocessed EEG",
        "BIDSVersion": BIDS_VERSION,
        "DatasetType": "derivative",
        "GeneratedBy": [generated_by],
        "DatasetLinks": {"raw": bids_root.as_uri()},
    }
    with open(deriv_root / "dataset_description.json", "w") as f:
        json.dump(dataset_description, f, indent=4)
    return deriv_root


# ---------------------------------------------------------------------------
# Step 2: map a raw file to its derivative path
# ---------------------------------------------------------------------------
def derivative_path(source_fname: os.PathLike, deriv_root: os.PathLike,
                    desc: str = "preproc") -> BIDSPath:
    """Same entities as the raw file, new root, plus ``desc-<desc>``.

    ``check=False`` because .nwb is not a BIDS-allowed extension for scalp EEG;
    the derivative is still named by BIDS rules.
    """
    src = get_bids_path_from_fname(source_fname, check=False)
    return src.copy().update(
        root=deriv_root, description=desc, datatype="eeg",
        suffix="eeg", extension=".nwb", check=False,
    )


# ---------------------------------------------------------------------------
# Step 3: the NWB file (same layout as the raw NWB files)
# ---------------------------------------------------------------------------
def write_clean_nwb(clean: mne.io.BaseRaw, raw_filename: os.PathLike,
                    out_filename: os.PathLike, provenance: str) -> None:
    """Copy the raw NWB file, swapping in the cleaned ElectricalSeries."""
    with NWBHDF5IO(raw_filename, "r") as read_io:
        nwbfile = read_io.read()
        old = nwbfile.acquisition["ElectricalSeries"]

        old_names = old.description.split(",")
        name_to_row = dict(zip(old_names, old.electrodes.data[:]))

        picks = mne.pick_types(clean.info, eeg=True, exclude=[])
        ch_names = [clean.ch_names[i] for i in picks]
        missing = set(ch_names) - name_to_row.keys()
        if missing:
            raise ValueError(f"Channels not in raw electrode table: {missing}")
        rows = [int(name_to_row[ch]) for ch in ch_names]

        nwbfile.acquisition.pop("ElectricalSeries")
        region = nwbfile.create_electrode_table_region(
            region=rows, description="Channels retained after preprocessing"
        )

        t0 = clean.info["meas_date"].timestamp() + clean.first_time
        timestamps = t0 + clean.times  # float64 absolute Unix time

        data = (clean.get_data(picks=picks).T * 1e6).astype(np.float32)  # µV

        es = ElectricalSeries(
            name="ElectricalSeries",
            data=H5DataIO(data, compression="gzip", compression_opts=4, chunks=True),
            timestamps=H5DataIO(timestamps, compression="gzip"),
            electrodes=region,
            description=",".join(ch_names),
            conversion=1e-6,
            comments=provenance,
            filtering=(f"highpass {clean.info['highpass']} Hz, "
                       f"lowpass {clean.info['lowpass']} Hz"),
        )
        nwbfile.add_acquisition(es)
        nwbfile.generate_new_id()

        with NWBHDF5IO(out_filename, "w") as export_io:
            export_io.export(src_io=read_io, nwbfile=nwbfile)


# ---------------------------------------------------------------------------
# Step 4: BIDS sidecars describing the derivative
# ---------------------------------------------------------------------------
def _write_sidecar_json(clean, deriv_path, source_uri, reference, description):
    sidecar = {
        "Description": description,
        "Sources": [source_uri],
        "TaskName": deriv_path.task,
        "SamplingFrequency": float(clean.info["sfreq"]),
        "PowerLineFrequency": clean.info.get("line_freq") or "n/a",
        "EEGReference": reference,
        "SoftwareFilters": {
            "HighPass": {"Cutoff (Hz)": clean.info["highpass"]},
            "LowPass": {"Cutoff (Hz)": clean.info["lowpass"]},
        },
        "EEGChannelCount": len(mne.pick_types(clean.info, eeg=True)),
        "RecordingDuration": float(clean.times[-1] + 1 / clean.info["sfreq"]),
        "RecordingType": "continuous",
    }
    fpath = deriv_path.copy().update(extension=".json", check=False).fpath
    with open(fpath, "w") as f:
        json.dump(sidecar, f, indent=4)


def _write_channels_tsv(clean, deriv_path, interpolated):
    """One row per channel stored in the NWB file (EEG only)."""
    picks = mne.pick_types(clean.info, eeg=True, exclude=[])
    names = [clean.ch_names[i] for i in picks]
    interpolated = set(interpolated or [])
    df = pd.DataFrame({
        "name": names,
        "type": "EEG",
        "units": "µV",
        "low_cutoff": clean.info["highpass"],
        "high_cutoff": clean.info["lowpass"],
        "status": "good",
        "status_description": ["interpolated" if n in interpolated else "n/a"
                               for n in names],
    })
    fpath = deriv_path.copy().update(suffix="channels", extension=".tsv",
                                     check=False).fpath
    df.to_csv(fpath, sep="\t", index=False)


def _write_events_tsv(clean, deriv_path):
    """Onsets are seconds from the first sample *of this file*.

    MNE stores annotation onsets relative to meas_date when orig_time is set,
    which includes first_time; subtracting it re-expresses them relative to
    the (possibly cropped) start of the derivative.
    """
    ann = clean.annotations
    if len(ann) == 0:
        return
    offset = clean.first_time if ann.orig_time is not None else 0.0
    df = pd.DataFrame({
        "onset": np.round(ann.onset - offset, 6),
        "duration": ann.duration,
        "trial_type": ann.description,
    })
    fpath = deriv_path.copy().update(suffix="events", extension=".tsv",
                                     check=False).fpath
    df.to_csv(fpath, sep="\t", index=False)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def write_derivative_nwb(
    clean: mne.io.BaseRaw,
    source_fname: os.PathLike,
    bids_root: os.PathLike,
    deriv_root: os.PathLike,
    provenance: str,
    reference: str = "average",
    interpolated: list[str] | None = None,
    desc: str = "preproc",
    overwrite: bool = False,
) -> BIDSPath:
    """Write one preprocessed recording as a BIDS derivative in NWB format.

    Parameters
    ----------
    clean : the preprocessed Raw.
    source_fname : the raw NWB file inside ``bids_root`` it came from.
    bids_root, deriv_root : the raw dataset and its derivative dataset.
    provenance : free-text processing record (stored in NWB and sidecar).
    reference : EEGReference value for the sidecar.
    interpolated : channels interpolated by PREP (``prep.interpolated_channels``).
    """
    deriv_path = derivative_path(source_fname, deriv_root, desc=desc)
    if deriv_path.fpath.exists() and not overwrite:
        raise FileExistsError(f"{deriv_path.fpath} exists; pass overwrite=True.")
    deriv_path.mkdir()

    rel = Path(source_fname).resolve().relative_to(Path(bids_root).resolve())
    source_uri = f"bids:raw:{rel.as_posix()}"

    write_clean_nwb(clean, source_fname, deriv_path.fpath, provenance)
    _write_sidecar_json(clean, deriv_path, source_uri, reference, provenance)
    _write_channels_tsv(clean, deriv_path, interpolated)
    _write_events_tsv(clean, deriv_path)
    return deriv_path