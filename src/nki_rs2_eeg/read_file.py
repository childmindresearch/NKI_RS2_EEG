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

def read_raw_nwb_old(nwb_path: str | os.PathLike, cap_dir: str | os.PathLike) -> mne.io.Raw:   
    subject_id = nwb_path.parts[-3]
    if nwb_path.exists():
        logger.info("Loading NWB file: %s", nwb_path)

        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwb = io.read()
            e_series = nwb.acquisition["ElectricalSeries"]
            stim_series = nwb.acquisition["StimLabels"]
            #electrode_info = nwb.electrodes.to_dataframe().copy()

            # Load 
            df = pd.DataFrame(e_series.data[()], columns=e_series.description.split(","))
            df["timestamps"] = e_series.timestamps[()]

            # Stim mapping (vectorized-ish, no list comp)
            stim_keys = stim_series.data[()].astype(str).flatten()
            stim_times = stim_series.timestamps[()]
        stim = dict(zip(stim_keys, stim_times))

        # Event filtering (use .between for clarity + speed)
        event_df = df[df['timestamps'].between(
            stim['Onset Movie'], stim['Offset Movie'])].copy()
        # CReate MNE Raw object
        info = mne.create_info(
            ch_names=list(event_df.columns[:-1]),
            sfreq=1 / event_df['timestamps'].diff().mean(),
            ch_types='eeg'
        )
        event_df = event_df.drop(columns=['timestamps'])

        # Get montage file based on cap type
        cap_types = pd.read_csv(os.path.join(cap_dir, 'captypes_clean.csv'))
        subject_cap_type = cap_types.loc[
            cap_types['a_number'] == subject_id[4:], 'cap_type'
        ].values[0]
        if subject_cap_type.startswith("RNP"):
            montage_file = os.path.join(cap_dir, 'R-Net for BrainAmp_RNP-BA', subject_cap_type)
        elif subject_cap_type.startswith("BC-MR"):
            montage_file = os.path.join(cap_dir, subject_cap_type)
        else:
            raise ValueError(f"Unknown cap type: {subject_cap_type}")
        montage = mne.channels.read_custom_montage(montage_file)
        info.set_montage(montage, on_missing='ignore')

        raw = mne.io.RawArray(
            event_df.T * 1e-6, info=info
        )  # multiplying by 1e-6 converts to volts
        #raw.subject_info = {"subject_id": subject_id, "session_id": session_id, "task_id": task_id, "run_id": run_id}
        return raw


def read_raw_nwb(filename: str | os.PathLike) -> mne.io.Raw:

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

    coord = {
    item["group_name"].split(" ")[-1]: (
        item["x"]*1e-3,
        item["y"]*1e-3,
        item["z"]*1e-3 
        )
    for _, item in electrodes.iterrows()
    }
    
    
    raw.set_meas_date(eeg_time[0])
    montage = mne.channels.make_dig_montage(
        ch_pos = coord)
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

    return raw, electrodes
    
# %%