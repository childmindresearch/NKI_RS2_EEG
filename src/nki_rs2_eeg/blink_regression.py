#%%

import argparse
import json
import logging
import os


nthreads = "8" 
os.environ["OMP_NUM_THREADS"] = nthreads
os.environ["OPENBLAS_NUM_THREADS"] = nthreads
os.environ["MKL_NUM_THREADS"] = nthreads
os.environ["VECLIB_MAXIMUM_THREADS"] = nthreads
os.environ["NUMEXPR_NUM_THREADS"] = nthreads

import pathlib
from pathlib import Path
import mne
import mne_bids
import numpy as np
import pandas as pd
from nki_rs2_eeg.read_file import read_processed_edf
from nki_rs2_eeg.config import (
    RAW_DATA_DIR,
    DERIVATIVES_DIR,
    SESSION_ID,
    TASK_ID,
    RUN_ID,
    CHANNEL_NAMES,
    FS,
    LINE_NOISE_HZ,
)



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
        raw_cleaned, #.pick_types(eeg = True),
        bids_path=bids_path,
        allow_preload=True,
        format="EDF",
        overwrite=True,
    )


def regress_blinks(raw: mne.io.Raw) -> mne.io.Raw:
    """Regress out blink artifacts from EEG data using linear regression.
    
    This function identifies blink-related activity in the EEG data and removes
    it by performing linear regression. The resulting cleaned EEG data is returned.
    
    Args:
        raw (mne.io.Raw): The raw EEG data containing blink artifacts.
            
    Returns:
        mne.io.Raw: The cleaned EEG data with blink artifacts regressed out.
    """
    # Create a copy of the raw data to avoid modifying the original
    raw.set_channel_types({"Fp1": "eog", "Fp2": "eog"})
    raw.set_eeg_reference("average")
    events, event_id = mne.events_from_annotations(raw)
    evid = event_id["blink"]
    blink_epochs = mne.Epochs(raw=raw, events=events, event_id=evid, tmin=-0.5, tmax=0.5, preload=True)
    model = mne.preprocessing.EOGRegression(picks="eeg", picks_artifact="eog").fit(blink_epochs)
    return model.apply(raw)
#%%

if __name__ == "__main__":
    
    filesofinsterest = list(DERIVATIVES_DIR.rglob(rf'sub-*_ses-{SESSION_ID}_task-{TASK_ID}_run-{RUN_ID}_eeg.edf'))

    for i, file in enumerate(filesofinsterest):
        print(f"Percent complete: {i/len(filesofinsterest)*100:.2f}% - Processing file: {file}")
        subject = file.stem.split('_')[0]
        # Define the BIDS path for the cleaned data
        saving_bids_path = mne_bids.BIDSPath(
                    root=DERIVATIVES_DIR,
                subject=subject[4:],
                session=SESSION_ID,
                datatype="eeg",
                task=TASK_ID,
                run=int(RUN_ID),
            )
        try:
            raw = read_processed_edf(file)
            raw_cleaned = regress_blinks(raw)
            save_bids_tree(raw_cleaned, saving_bids_path)
        except Exception as e:
            logging.error(f"Error processing file {file}: {e}")