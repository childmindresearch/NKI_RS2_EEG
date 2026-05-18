#%%
"""Module for saving processed EEG data in a structured format.

This module provides functions to save processed EEG data in a structured
format for later analysis. It includes utilities to trim raw data to specific
event windows and create condition-specific dictionaries of EEG data for use
in analyses like CCA and CorrCA.
"""
import argparse

import mne
import numpy as np

from config import DERIVATIVES_DIR

#%%
def trim_data_to_event(
    raw: mne.io.BaseRaw, onset_label: str, offset_label: str
) -> mne.io.BaseRaw:
    """Trim the raw data to the time window defined by the first occurrence of the specified onset and offset labels.

    Args:
        raw (mne.io.BaseRaw): The raw EEG data in MNE format.
        onset_label (str): The label of the onset event.
        offset_label (str): The label of the offset event.

    Returns:
        mne.io.BaseRaw: The trimmed raw EEG data.
    """
    onsets = raw.annotations.onset
    descriptions = raw.annotations.description

    # Example: Get the onset time for the first occurrence of each
    t_start = onsets[descriptions == onset_label][0]
    t_stop = onsets[descriptions == offset_label][0]
    return raw.copy().crop(tmin=t_start, tmax=t_stop)
#%%

def create_condition_array(processed_files: list, onset_label: str, offset_label: str) -> np.ndarray:
    """Create an array of shape (subjects, channels, samples).

    Args:
        processed_files (list): List of file paths to the processed EEG data.
        onset_label (str): The label of the onset event.
        offset_label (str): The label of the offset event.

    Returns:
        np.ndarray: A numpy array of shape (subjects, channels, samples).
    """
    # First get a list of raw objects trimmed to event of interest
    raws = []
    subject_order = []
    for f in processed_files:
        raw = mne.io.read_raw_edf(f, preload=True)
        raw = trim_data_to_event(raw, onset_label, offset_label)
        raws.append(raw)
        subject_order.append(f.stem)
    # They all must have the same number of samples and channels, so we can stack them into a numpy array
    min_samples = min([r.get_data().shape[1] for r in raws])
    raws_dat = [r.copy().get_data()[:, :min_samples] for r in raws]
    # mean center the data across time for each channel*subject
    #raws_dat = [r - np.mean(r, axis=1, keepdims=True) for r in raws_dat]
    return np.stack(raws_dat), subject_order
#%%

def save_collated_condition_data(
    session_id: str,
    task_id: str,
    run_id: str,
    onset_label: str,
    offset_label: str,
) -> None:
    """Save the condition dictionary to a specified path.

    Args:
        session_id (str): The session ID.
        task_id (str): The task ID.
        run_id (str): The run ID.
        onset_label (str): The label of the onset event.
        offset_label (str): The label of the offset event.

    """
    processed_files = list(
        DERIVATIVES_DIR.rglob(f'sub-*{session_id}*{task_id}*run-{run_id}_eeg.edf')
    )
    condition, subject_order = create_condition_array(
        processed_files, onset_label, offset_label
    )
    save_path = DERIVATIVES_DIR / f'sub-ALL_ses-{session_id}_task-{task_id}_run-{run_id}_eeg.npy'

    np.save(
        DERIVATIVES_DIR
        / f'sub-ALL_ses-{session_id}_task-{task_id}_run-{run_id}_isc_sub_order.npy',
        subject_order,
    )
    np.save(save_path, condition)
    print(f"{len(processed_files)} subjects collated and saved to {save_path}.")

# %%

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save collated condition data.")
    parser.add_argument("--session_id", type=str, default="MOBI1A", help="Session ID")
    parser.add_argument("--task_id", type=str, default="passivepresent", help="Task ID")
    parser.add_argument("--run_id", type=str, default="ALL", help="Run ID")
    parser.add_argument("--onset_label", type=str, default="S  1", help="Onset event label")
    parser.add_argument("--offset_label", type=str, default="S  2", help="Offset event label")

    args = parser.parse_args()

    save_collated_condition_data(
        session_id=args.session_id,
        task_id=args.task_id,
        run_id=args.run_id,
        onset_label=args.onset_label,
        offset_label=args.offset_label,
    )