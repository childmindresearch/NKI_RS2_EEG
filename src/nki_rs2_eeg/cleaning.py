#%%
from __future__ import annotations

import argparse
import json
import logging
import os

nthreads = "12" 
os.environ["OMP_NUM_THREADS"] = nthreads
os.environ["OPENBLAS_NUM_THREADS"] = nthreads
os.environ["MKL_NUM_THREADS"] = nthreads
os.environ["VECLIB_MAXIMUM_THREADS"] = nthreads
os.environ["NUMEXPR_NUM_THREADS"] = nthreads

import pathlib
from typing import Any
from pathlib import Path
import mne
import mne_bids
import numpy as np
import pandas as pd
import pyprep
from nki_rs2_eeg import read_file
from nki_rs2_eeg.read_file import read_raw_nwb

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
LOG_PATH = DERIVATIVES_DIR / "cleaning.log"
CAP_DIR = _REPO_ROOT / "data" / "caps"
SESSION_ID = "MOBI2C"
TASK_ID = "passivepresent"
RUN_ID = "01"


#%%
#raw = read_raw_nwb(nwb_path=nwb_path, cap_dir=CAP_DIR)

# %%

def run_prep(raw: mne.io.Raw) -> pyprep.PrepPipeline:
    """Run the PREP pipeline for EEG preprocessing.
    
    Applies the PREP pipeline which includes:
    1. Setting channel montage
    2. Filtering (0-125 Hz) and resampling to 250 Hz
    3. Reference correction
    4. Line noise removal (60 Hz and harmonics)
    
    Args:
        raw (mne.io.Raw): The raw EEG data in MNE format.

    Returns:
        pyprep.PrepPipeline: The fitted PREP pipeline object containing
                             the cleaned data and preprocessing information.
    """
    raw.filter(l_freq=0, h_freq=125).resample(250)
    prep_params = {
        "ref_chs": "eeg",
        "reref_chs": "eeg",
        "line_freqs": np.arange(60, raw.info["sfreq"] / 2, 60),
    }
    prep = pyprep.PrepPipeline(
        raw, 
        montage=raw.get_montage(), 
        channel_wise=True, 
        prep_params=prep_params
    )

    return prep.fit()
#%%

def annotate_blinks(raw: mne.io.Raw) -> mne.Annotations:
    """Detect and annotate eye blinks in EEG data.
    
    Uses frontal electrodes (Fp1, Fp2) to detect eye blinks and
    creates annotations marking their occurrences.
    
    Args:
        raw (mne.io.Raw): The EEG data in MNE format.
        
    Returns:
        mne.Annotations: Annotations marking detected eye blinks.
    """
    eog_epochs = mne.preprocessing.create_eog_epochs(raw, ch_name=["Fp1", "Fp2"])
    blink_annotations = mne.annotations_from_events(
        eog_epochs.events,
        raw.info["sfreq"],
        event_desc={eog_epochs.events[0, 2]: "blink"},
        orig_time=raw.info["meas_date"]
    )
    return blink_annotations

def annotate_muscle(raw: mne.io.Raw) -> mne.Annotations:
    """Detect and annotate muscle artifacts in EEG data.
    
    Uses z-score thresholding in the high-frequency band (95-120 Hz)
    to identify muscle artifacts in the EEG data.
    
    Args:
        raw (mne.io.Raw): The EEG data in MNE format.
        
    Returns:
        mne.Annotations: Annotations marking detected muscle artifacts.
    """
    muscle_annotations, _ = mne.preprocessing.annotate_muscle_zscore(
        raw, 
        threshold=3, 
        ch_type='eeg', 
        min_length_good=0.1, 
        filter_freq=(95, 120),
        )

    return muscle_annotations

def combine_annotations(
    annotations_list: list[mne.Annotations]
                        ) -> mne.Annotations:
    """Combine multiple MNE Annotations objects into a single object.
    
    Takes a list of annotation objects and combines them into a single
    annotations object. Handles empty annotation lists gracefully.
    
    Args:
        annotations_list (list[mne.Annotations]): List of annotation objects to combine.
        
    Returns:
        mne.Annotations: Combined annotations object.
    """
    return sum(annotations_list, start=mne.Annotations([],[],[]))


def save_bids_tree(raw_cleaned: mne.io.Raw, bids_path: mne_bids.BIDSPath) -> None:
    """Save preprocessed EEG data in BIDS-compliant format.
    
    Writes the cleaned EEG data to a BIDS-compliant EDF file using the
    specified BIDS path.
    
    Args:
        raw_cleaned (mne.io.Raw): Cleaned EEG data to save.
        bids_path (mne_bids.BIDSPath): BIDS path object specifying where to save the data.
        
    Returns:
        None: The function saves the EEG data to disk and doesn't return anything.
    """
    mne_bids.write_raw_bids(
        raw_cleaned.pick_types(eeg = True),
        bids_path=bids_path,
        allow_preload=True,
        format="EDF",
        overwrite=True,
    )
#%%
def set_nwb_channel_dataframe(
    channels: pd.DataFrame, 
    prep_output
    ) -> pd.DataFrame:
    """Create a DataFrame with channel information and quality metrics.
    
    Constructs a DataFrame that contains channel names, flags for different
    types of noisy channels, and impedance values.

    Args:
        raw_bv (mne.io.Raw): Raw MNE object containing the EEG data and impedance values.
        prep_output: Object containing preprocessing results, with attributes:
            - noisy_channels_original (dict): Dictionary mapping noise labels to channel lists
            - still_noisy_channels (list): List of channels that remain noisy after preprocessing

    Returns:
        pd.DataFrame: DataFrame with channel names as index and columns for:
            - Various noise flags from prep_output.noisy_channels_original
            - 'still_noisy' flag for channels that remain noisy
            - 'impedances' values for each channel
    """
    df_dict = {"name": [s.split(" ")[-1] 
                        for s in channels["group_name"].values]}
    for bad_label, bad_ch in prep_output.noisy_channels_original.items():
        if bad_label != "bad_all":
            df_dict[bad_label] = np.isin(df_dict["name"], bad_ch)
    df_dict["still_noisy"] = np.isin(df_dict["name"], prep_output.still_noisy_channels)
    df_dict["impedances"] = channels["allImpedances"].values

    return pd.DataFrame(df_dict).set_index("name")
#%%
def save_nwb_channels_info(
    channels: pd.DataFrame,
    prep_output,
    saving_bids_path
    ):
    """Save channel information to a BIDS-compliant TSV file.
    
    Creates a channel information DataFrame using set_channel_dataframe(),
    then merges it with any existing channel information and saves it
    to a TSV file following BIDS formatting conventions.

    Args:
        raw_bv (mne.io.Raw): Raw MNE object containing the EEG data.
        prep_output: Object containing preprocessing results with noisy channel information.
        saving_bids_path: Object with attributes:
            - fpath (pathlib.Path): Path object for the parent directory
            - basename (str): Base filename to use for the output file

    Returns:
        None: The function saves the channel information to a TSV file and doesn't return anything.
    """
    channel_dataframe = set_nwb_channel_dataframe(channels, prep_output)

    

    channel_info_fname = saving_bids_path.fpath.parent / (
        os.fspath(saving_bids_path.basename) + "_channels.tsv"
    )

    channel_info_dataframe = pd.read_csv(
        channel_info_fname, sep="\t", index_col=["name"]
    )

    result = channel_info_dataframe.join(channel_dataframe, how="outer")
    values_to_replace = result.loc[~result["type"].isna().values]
    columns_to_fill = [
        "type",
        "units",
        "low_cutoff",
        "high_cutoff",
        "description",
        "sampling_frequency",
    ]
    for col in columns_to_fill:
        result.loc[result[col].isna().values,col] = values_to_replace[col].iloc[0]

    result.loc[result["still_noisy"], "status"] = "bad"
    result.to_csv(channel_info_fname, sep="\t")


#%%    
def full_pipeline(file: str, saving_bids_path: os.PathLike, overwrite = False) -> str:
    """Execute the complete EEG preprocessing pipeline for a given file.
    
    The pipeline includes:
    1. Loading BrainVision and XDF files
    2. Setting up BIDS path for saving
    3. Running PREP preprocessing
    4. Detecting and annotating artifacts (blinks, muscle)
    5. Extracting and adding experimental markers
    6. Saving the processed data in BIDS format
    
    Args:
        file (dict): Dictionary containing file information with keys:
            - 'root': Root directory path
            - 'subject': Subject ID
            - 'session': Session ID
            - 'run': Run number
            - 'task': Task name
            - 'filename': Path to the XDF file
            
    Returns:
        str: Status message indicating success ("OK"), "Already Done", or error message.
    """
    theoretical_fname = Path(os.fspath(saving_bids_path.fpath))
    
    print(f"Theoretical fname:{theoretical_fname}")
    saving_bids_path.mkdir()
    try:
        if theoretical_fname.is_file() and not overwrite:
            return "Already Done"
        else:
            saving_bids_path.mkdir()

    except Exception as e:
        saving_bids_path.mkdir()
        return str(e)

    mne.set_log_level(verbose="ERROR")
    raw, channels = read_file.read_raw_nwb(file)
    
    try:
        prep_output = run_prep(raw)
        raw_cleaned = prep_output.raw
    except Exception as e:
        return str(e)

    try:
        blinks_annotations = annotate_blinks(raw_cleaned)
        muscle_annotations = annotate_muscle(raw_cleaned)

        annotations = combine_annotations([
            blinks_annotations,
            muscle_annotations,
            raw.annotations,
            ])

        raw_cleaned.set_annotations(annotations)
        save_bids_tree(raw_cleaned, saving_bids_path)
        save_nwb_channels_info(channels, prep_output, saving_bids_path)
    except Exception as e:
        return str(e)
    
    return "OK"

#%%
if __name__ == "__main__":
    DERIVATIVES_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH),
            logging.StreamHandler(),
        ],
    )

    logger.info("Logging to %s", LOG_PATH)


    report = {"subject":[],
              "session":[],
              "task":[],
              "run":[],
              "message":[]
    }
    filesofinsterest = list(RAW_DIR.rglob(rf'sub-*_ses-{SESSION_ID}_task-{TASK_ID}_run-{RUN_ID}_MoBI.nwb'))

    for file in filesofinsterest[:5]:

        if not file.suffix == ".nwb":
            continue
        print(f"=== PROCESSING {file.name} ===")
        subject = file.parts[-3]
        report["subject"].append(subject)
        report["session"].append(SESSION_ID)
        report["task"].append(TASK_ID)
        report["run"].append(RUN_ID)
        
        try:
            saving_bids_path = mne_bids.BIDSPath(
                root=DERIVATIVES_DIR,
                subject=subject[4:],
                session=SESSION_ID,
                datatype="eeg",
                task=TASK_ID,
                run=int(RUN_ID),
            )
            message = full_pipeline(file, saving_bids_path, overwrite = True)
            report["message"].append(message)

        except Exception as e:
            report["message"].append(str(e))
            continue

    report_df = pd.DataFrame(report)
    report_df.to_csv(DERIVATIVES_DIR / "cleaning_report_nwb.tsv", sep = "\t", index = False)

#%%