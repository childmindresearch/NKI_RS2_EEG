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

from nki_rs2_eeg.config import DERIVATIVES_DIR

#%%
def trim_data_to_event(
    raw: mne.io.BaseRaw, onset_label: str, offset_label: str
) -> tuple[mne.io.BaseRaw, str]:
    """Trim the raw data to the time window defined by the first occurrence of the specified onset and offset labels.

    Args:
        raw (mne.io.BaseRaw): The raw EEG data in MNE format.
        onset_label (str): The label of the onset event.
        offset_label (str): The label of the offset event.

    Returns:
        tuple[mne.io.BaseRaw, str]: The trimmed raw EEG data and a status message.
    """
    try:
        onsets = raw.annotations.onset
        descriptions = raw.annotations.description

        # Example: Get the onset time for the first occurrence of each
        t_start = onsets[descriptions == onset_label][0]
        t_stop = onsets[descriptions == offset_label][0]
        return raw.copy().crop(tmin=t_start, tmax=t_stop), "all good"
    except Exception as e:
        print(f"Error trimming data: {e}")
        return raw.copy(), "Not Good..."
#%%

def create_condition_array(processed_files: list, onset_label: str, offset_label: str) -> tuple[np.ndarray, list]:
    """Create an array of shape (subjects, channels, samples).

    Args:
        processed_files (list): List of file paths to the processed EEG data.
        onset_label (str): The label of the onset event.
        offset_label (str): The label of the offset event.

    Returns:
        tuple[np.ndarray, list]: A numpy array of shape (subjects, channels, samples) and a list of subject order.
    """
    # First get a list of raw objects trimmed to event of interest
    raws = []
    subject_order = []
    for i, f in enumerate(processed_files):
        print(f"{i/len(processed_files)*100:.2f}% done")
        try:
            raw = mne.io.read_raw_edf(f, preload=True)
            raw, status = trim_data_to_event(raw, onset_label, offset_label)
            if status != "all good":
                print(f"Warning: {status} for file {f}")
                continue
    
        except Exception as e:
            print(f"Error processing file {f}: {e}")
            continue

        raws.append(raw)
        subject_order.append(f.stem)  

    min_samples = min(r.n_times for r in raws)
    n_channels = raws[0].info['nchan']
    out = np.empty((len(raws), n_channels, min_samples))
    for i, r in enumerate(raws):
        out[i] = r.get_data()[:, :min_samples]
        del raws[i]
    
    return out, subject_order
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