#%%
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from scipy.linalg import eigh, inv
import pandas as pd
from nki_rs2_eeg.config import (SESSION_ID, TASK_ID, RUN_ID, DERIVATIVES_DIR, CONCAT_DATA_DIR, RAW_DATA_DIR)
#%%
channel_files = list(DERIVATIVES_DIR.rglob(f"sub*/ses-{SESSION_ID}/eeg/sub-*_ses-{SESSION_ID}_task-{TASK_ID}_run-{RUN_ID}_channels.tsv"))
# Combine all channel files into a single DataFrame
channels_df = pd.concat([pd.read_csv(file, sep='\t') for file in channel_files])
sub_ids = [file.parts[-4] for file in channel_files]
channels_df['subject'] = np.repeat(sub_ids, 64)
channels_df['is_bad'] = channels_df['status'] == 'bad'
# for each subject get the sum of the boolean columns
channels = channels_df.groupby('subject')[channels_df.select_dtypes(include='bool').columns].sum()
subject_exclusions = channels[channels.is_bad > 3].index.tolist()
excluded = [s.split('-')[1] for s in subject_exclusions]


#%%
subject_isc = np.load(DERIVATIVES_DIR / f"sub-ALL_ses-{SESSION_ID}_task-{TASK_ID}_run-{RUN_ID}_isc_per_subject.npy")
subject_order = np.load(DERIVATIVES_DIR / f"sub-ALL_ses-{SESSION_ID}_task-{TASK_ID}_run-{RUN_ID}_isc_sub_order.npy")
subject_order = [s.split('-')[1].split('_')[0] for s in subject_order]
subject_age = pd.read_csv("/home/bgonzalez/NKI_RS2_EEG/data/RS2_age.csv", sep=',')
subject_age['subject'] = subject_age['mri_age_yr']#%%



sub_isc_df = pd.DataFrame({
    'subject': subject_order,
    'ISC': subject_isc
})
sub_isc_df = sub_isc_df.merge(subject_age, on='subject')
#%%
# get only the subject not excluded
sub_isc_df = sub_isc_df[~sub_isc_df['subject'].isin(excluded)]

#%%
