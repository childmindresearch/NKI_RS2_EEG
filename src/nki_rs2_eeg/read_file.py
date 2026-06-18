#%%
from pathlib import Path
import numpy as np
import mne
from datetime import datetime
from pynwb import NWBHDF5IO
import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)



def convert_signal(eeg_stream:dict) -> np.ndarray:
    units = {
        "microvolts": 10e-6,
        "millivolts": 10e-3,
        "volts": 1,
    }
    unit_matrix = list()
    chan_dict = eeg_stream['info']['desc'][0]['channels'][0]['channel']
    signals = eeg_stream['time_series'].T
    unit_matrix = np.array([[units.get(chan['unit'][0].lower(),1) 
                   for chan in chan_dict]]).T
    return np.multiply(signals,unit_matrix)


def read_raw_nwb(filename: str | os.PathLike) -> tuple[mne.io.Raw, pd.DataFrame]:
    '''Read a raw NWB file and return an MNE Raw object.
    
    Parameters
    ----------
    filename : str or os.PathLike
        Path to the NWB file.           
    Returns
    -------
    raw : mne.io.Raw
        The loaded Raw object containing the EEG data.
    electrodes : pd.DataFrame
        The electrode information from the NWB file.

    '''
    with NWBHDF5IO(filename, 'r') as io:
        nwbfile = io.read()
        eeg_data = nwbfile.acquisition["ElectricalSeries"].data[:]
        eeg_time = nwbfile.acquisition["ElectricalSeries"].timestamps[:]
        electrodes = nwbfile.acquisition["ElectricalSeries"].electrodes[:]
        elec_names = nwbfile.acquisition["ElectricalSeries"].description[:]
        events_name = nwbfile.acquisition["StimLabels"].data[:]
        events_onset = nwbfile.acquisition["StimLabels"].timestamps[:]
        
    datetime_times = [
        datetime.fromtimestamp(t) for t in eeg_time
    ]
    first_time_step = datetime_times[1] - datetime_times[0]
    sfreq = 1e6 / first_time_step.microseconds

    info = mne.create_info(
        ch_names = elec_names.split(","),
        sfreq = sfreq,
        ch_types = "eeg"
    )

    raw = mne.io.RawArray(eeg_data.T*1e-6, info)
    # TODO: TRY LOADING .bvef file for montage 
    # TODO: check out LPA, RPA, Nz coordinates in the standardcoordinates.txt file on nki github
    coord = {
    item["group_name"].split(" ")[-1]: (
        item["x"]*1e-3,
        item["y"]*1e-3,
        item["z"]*1e-3 
        )
    for _, item in electrodes.iterrows()
    }
    
    
    raw.set_meas_date(eeg_time[0])
    #montage = mne.channels.make_dig_montage(
     #   ch_pos = coord)
    montage = mne.channels.read_custom_montage('/home/bgonzalez/NKI_RS2_EEG/data/caps/R-Net for BrainAmp_RNP-BA/RNP-BA-64.bvef')
    raw.set_montage(montage)
    

    eeg_start = datetime.fromtimestamp(eeg_time[0])
    delta = [datetime.fromtimestamp(t) - eeg_start for t in events_onset]
    onset = [d.total_seconds() for d in delta]
    duration = np.zeros_like(onset)

    annotations = mne.Annotations(
        onset=onset,
        duration=duration,
        description=[name[0] for name in events_name],
        orig_time=raw.info["meas_date"]
    )
    
    raw.set_annotations(annotations)
    onsets = raw.annotations.onset
    descriptions = raw.annotations.description

    try:
        # Example: Get the onset time for the first occurrence of each
        t_start = onsets[descriptions == "Onset Movie"][0]
        t_stop = onsets[descriptions == "Offset Movie"][0]
        return raw.copy().crop(tmin=t_start, tmax=t_stop, reset_first_samp=True), electrodes
    except Exception as e:
        print(f"Error trimming data, event markers not found: {e}")


    
# %%

def read_processed_edf(filename: str | os.PathLike) -> mne.io.Raw:
    """Read a processed EDF file and return an MNE Raw object.

    Parameters
    ----------
    filename : str or os.PathLike
        Path to the EDF file.
    
    Returns
    -------
    raw : mne.io.Raw
        The loaded Raw object containing the EEG data.
    """
    return mne.io.read_raw_edf(filename, preload=True)


#%%
def trim_data_to_event(raw: mne.io.BaseRaw, onset_label: str, offset_label: str) -> mne.io.BaseRaw:
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

def create_condition_dict(processed_files: list, conditions: list) -> dict:
    """Create a dictionary with keys as condition names and values as numpy arrays of shape (subjects, channels, samples).

    Args:
        processed_files (list): List of file paths to the processed EEG data.
        conditions (list): List of condition names to be included in the dictionary.

    Returns:
        dict: A dictionary where keys are condition names and values are numpy arrays of shape (subjects, channels, samples).
    """
    data = {}
    # First get a list of raw objects trimmed to event of interest
    for condition in conditions:
        raws = []
        for f in processed_files:
            raw = read_processed_edf(f, preload=True)
            raw = trim_data_to_event(raw, 'Onset Movie', 'Offset Movie')
            raws.append(raw)
        # They all must have the same number of samples and channels, so we can stack them into a numpy array
        min_samples = min([r.get_data().shape[1] for r in raws])
        raws_dat = [r.copy().get_data()[:, :min_samples] for r in raws]
        # mean center the data across time for each channel*subject
        raws_dat = [r - np.mean(r, axis=1, keepdims=True) for r in raws_dat]
        data[condition] = np.stack(raws_dat)
    return data
# %%

