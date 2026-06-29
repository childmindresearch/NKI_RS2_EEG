#%%
import logging
import os
import re
import pathlib
from pathlib import Path
import mne
import mne_bids
import numpy as np
import pandas as pd
import pyprep
from nki_rs2_eeg import read_file
from nki_rs2_eeg.config import ( RAW_DATA_DIR, DERIVATIVES_DIR, LOG_PATH)



nthreads = "8" 
os.environ["OMP_NUM_THREADS"] = nthreads
os.environ["OPENBLAS_NUM_THREADS"] = nthreads
os.environ["MKL_NUM_THREADS"] = nthreads
os.environ["VECLIB_MAXIMUM_THREADS"] = nthreads
os.environ["NUMEXPR_NUM_THREADS"] = nthreads




logger = logging.getLogger(__name__)
#%%
# Get only present and sherlock .nwb files

#%%

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
    raw.filter(l_freq=0, h_freq=125).resample(500)
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
    model = mne.preprocessing.EOGRegression(picks="eeg", picks_artifact=["Fp1", "Fp2"]).fit(blink_epochs)
    return model.apply(raw)
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

#%%


def full_pipeline(nwb_path, saving_bids_path,  overwrite=False):
     

    theoretical_fname = Path(os.fspath(saving_bids_path.fpath))
        
    print(f"Theoretical fname:{theoretical_fname}")
    saving_bids_path.mkdir()
    try:
        if theoretical_fname.is_file() and not overwrite:
            return "Already Done - Not Overwritten"
        else:
            saving_bids_path.mkdir()

    except Exception as e:
        saving_bids_path.mkdir()
        return str(e)

    mne.set_log_level(verbose="ERROR")
    raw, channels = read_file.read_raw_nwb(nwb_path)
    try: 
        prep_output = run_prep(raw=raw)
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
    except Exception as e:
        return str(e)

    try:
        raw_cleaned = regress_blinks(raw_cleaned)
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
    movies = ["passivepresent", "passivesherlock"]
    pattern = re.compile("|".join(map(re.escape, movies)))
    nwb_files = [
        Path(os.path.join(dirpath, name))
        for dirpath, _, filenames in os.walk(RAW_DATA_DIR)
        for name in filenames
        if pattern.search(name) and name.endswith("_MoBI.nwb")
    ]

    for i, nwb_path in enumerate(nwb_files[:2]):
        print("========================")
        print(" ")
        print(f"processing complete: {(i/len(nwb_files))*100:.02f}%")
        print(" ")
        print("========================")
        file_parts = nwb_path.parts[-1].split('_')
        for k, v in zip(list(report.keys())[:-1], file_parts[:-1]):
            report[k].append(v.split('-')[1])
        
        try:
            saving_bids_path = mne_bids.BIDSPath(
                        root=DERIVATIVES_DIR,
                        subject=report['subject'][-1],
                        session=report['session'][-1],
                        datatype="eeg",
                        task=report['task'][-1],
                        run='0'+str(int(report['run'][-1])), #bc inconsistent file names (e.g. 001, 1, 01)
                        suffix='eeg'
                    ) 
            message = full_pipeline(nwb_path, saving_bids_path)
            report["message"].append(message)
        except Exception as e:
            report["message"].append(str(e))
            continue
    
    if (DERIVATIVES_DIR / "cleaning_report_nwb.tsv").is_file():
        existing_report = pd.read_csv(DERIVATIVES_DIR / "cleaning_report_nwb.tsv", sep="\t")
        report_df = pd.DataFrame(report)
        combined_report = pd.concat([existing_report, report_df], ignore_index=True)
        combined_report.to_csv(DERIVATIVES_DIR / "cleaning_report_nwb.tsv", sep="\t", index=False)
    else:
        report_df = pd.DataFrame(report)
        report_df.to_csv(DERIVATIVES_DIR / "cleaning_report_nwb.tsv", sep = "\t", index = False)

    print("======Finally Done!=======")



