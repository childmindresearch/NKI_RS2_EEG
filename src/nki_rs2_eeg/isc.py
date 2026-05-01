#%%
from __future__ import annotations

from mne import data
import numpy as np
from scipy.linalg import eigh
from timeit import default_timer
import mne

import argparse
import json
import logging
import os
import pathlib
from typing import Any

import mne
import numpy as np
import pandas as pd
import pynwb
import pyprep
import seaborn as sns
import matplotlib.pyplot as plt

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
#RAW_DIR = _REPO_ROOT / "data" / "raw"
DERIVATIVES_DIR = _REPO_ROOT / "data" / "derivatives"
SESSION_ID = "MOBI1A"
TASK_ID = "passivepresent"
RUN_ID = "01"
PROCESSED_FILES = list(DERIVATIVES_DIR.rglob(rf'sub-*_ses-{SESSION_ID}_task-{TASK_ID}_run-*_eeg.edf'))
FIGURES_DIR = _REPO_ROOT / "figures"

#%%
def trim_data_to_event(raw: mne.io.BaseRaw, onset_lable: str, offset_label: str) -> mne.io.BaseRaw:
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
    t_start = onsets[descriptions == onset_lable][0]
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
            raw = mne.io.read_raw_edf(f, preload=True)
            raw = trim_data_to_event(raw, 'Onset Movie', 'Offset Movie')
            raws.append(raw)
        # They all must have the same number of samples and channels, so we can stack them into a numpy array
        min_samples = min([r.get_data().shape[1] for r in raws])
        raws_dat = [r.copy().get_data()[:, :min_samples] for r in raws]
        data[condition] = np.stack(raws_dat)
    return data

#%%

def train_cca(data):
    """Run Correlated Component Analysis on your training data.

        Parameters:
        ----------
        data : dict
            Dictionary with keys are names of conditions and values are numpy
            arrays structured like (subjects, channels, samples).
            The number of channels must be the same between all conditions!

        Returns:
        -------
        W : np.array
            Columns are spatial filters. They are sorted in descending order, it means that first column-vector maximize
            correlation the most.
        ISC : np.array
            Inter-subject correlation sorted in descending order

    """

    start = default_timer()

    C = len(data.keys())
    print(f'train_cca - calculations started. There are {C} conditions')

    gamma = 0.1
    Rw, Rb = 0, 0
    for cond in data.values():
        N, D, T, = cond.shape
        print(f'Condition has {N} subjects, {D} sensors and {T} samples')
        cond = cond.reshape(D * N, T)

        # Rij
        Rij = np.swapaxes(np.reshape(np.cov(cond), (N, D, N, D)), 1, 2)

        # Rw
        Rw = Rw + np.mean([Rij[i, i, :, :]
                           for i in range(0, N)], axis=0)

        # Rb
        Rb = Rb + np.mean([Rij[i, j, :, :]
                           for i in range(0, N)
                           for j in range(0, N) if i != j], axis=0)

    # Divide by number of condition
    Rw, Rb = Rw/C, Rb/C

    # Regularization
    Rw_reg = (1 - gamma) * Rw + gamma * np.mean(eigh(Rw)[0]) * np.identity(Rw.shape[0])

    # ISCs and Ws
    [ISC, W] = eigh(Rb, Rw_reg)

    # Make descending order
    ISC, W = ISC[::-1], W[:, ::-1]

    stop = default_timer()

    print(f'Elapsed time: {round(stop - start)} seconds.')
    return W, ISC

# %%

def apply_cca(X, W, fs):
    """Applying precomputed spatial filters to your data.

        Parameters:
        ----------
        X : ndarray
            3-D numpy array structured like (subject, channel, sample)
        W : ndarray
            Spatial filters.
        fs : int
            Frequency sampling.
        Returns:
        -------
        ISC : ndarray
            Inter-subject correlations values are sorted in descending order.
        ISC_persecond : ndarray
            Inter-subject correlations values per second where first row is the most correlated.
        ISC_bysubject : ndarray
            Description goes here.
        A : ndarray
            Scalp projections of ISC.
    """

    start = default_timer()
    print('apply_cca - calculations started')

    N, D, T = X.shape
    # gamma = 0.1
    window_sec = 5
    X = X.reshape(D * N, T)

    # Rij
    Rij = np.swapaxes(np.reshape(np.cov(X), (N, D, N, D)), 1, 2)

    # Rw
    Rw = np.mean([Rij[i, i, :, :]
                  for i in range(0, N)], axis=0)
    # Rw_reg = (1 - gamma) * Rw + gamma * np.mean(eigh(Rw)[0]) * np.identity(Rw.shape[0])

    # Rb
    Rb = np.mean([Rij[i, j, :, :]
                  for i in range(0, N)
                  for j in range(0, N) if i != j], axis=0)

    # ISCs
    ISC = np.sort(np.diag(np.transpose(W) @ Rb @ W) / np.diag(np.transpose(W) @ Rw @ W))[::-1]

    # Scalp projections
    A = np.linalg.solve(Rw @ W, np.transpose(W) @ Rw @ W)

    # ISC by subject
    print('by subject is calculating')
    ISC_bysubject = np.empty((D, N))

    for subj_k in range(0, N):
        Rw, Rb = 0, 0
        Rw = np.mean([Rw + 1 / (N - 1) * (Rij[subj_k, subj_k, :, :] + Rij[subj_l, subj_l, :, :])
                      for subj_l in range(0, N) if subj_k != subj_l], axis=0)
        Rb = np.mean([Rb + 1 / (N - 1) * (Rij[subj_k, subj_l, :, :] + Rij[subj_l, subj_k, :, :])
                      for subj_l in range(0, N) if subj_k != subj_l], axis=0)

        ISC_bysubject[:, subj_k] = np.diag(np.transpose(W) @ Rb @ W) / np.diag(np.transpose(W) @ Rw @ W)

    # ISC per second
    print('by persecond is calculating')
    ISC_persecond = np.empty((D, int(T / fs) + 1))
    window_i = 0

    for t in range(0, T, fs):

        Xt = X[:, t:t+window_sec*fs]
        Rij = np.cov(Xt)
        Rw = np.mean([Rij[i:i + D, i:i + D]
                      for i in range(0, D * N, D)], axis=0)
        Rb = np.mean([Rij[i:i + D, j:j + D]
                      for i in range(0, D * N, D)
                      for j in range(0, D * N, D) if i != j], axis=0)

        ISC_persecond[:, window_i] = np.diag(np.transpose(W) @ Rb @ W) / np.diag(np.transpose(W) @ Rw @ W)
        window_i += 1

    stop = default_timer()
    print(f'Elapsed time: {round(stop - start)} seconds.')

    return ISC, ISC_persecond, ISC_bysubject, A



# %%

if __name__ == "__main__":

    # Get cleaning report
    cleaning_report = pd.read_csv(DERIVATIVES_DIR / "cleaning_report_nwb.tsv", sep="\t")
    cleaning_report = cleaning_report[cleaning_report['session'] == SESSION_ID]
    cleaned_subs = sorted(list(set(cleaning_report[(cleaning_report['message'] == 'OK') | (cleaning_report['message']== "Already Done")]['subject'].tolist())) )
    # find the .edf files for the cleaned subjects
    processed_files = sorted([f for f in PROCESSED_FILES if f.stem.split('_')[0] in cleaned_subs])
    data = create_condition_dict(processed_files, [TASK_ID])
    
    W, ISC_overall = train_cca(data)

    isc_results = dict()
    for cond_key, cond_values in data.items():
        isc_results[str(cond_key)] = dict(zip(['ISC', 'ISC_persecond', 'ISC_bysubject', 'A'], apply_cca(cond_values, W, 1000)))

    subj_isc = isc_results['passivepresent']['ISC_bysubject'].mean(axis=0)
    ages = pd.read_csv(_REPO_ROOT / "data" / "RS2_age.csv", sep=",")
    x = {
    'subject_id': cleaned_subs, 
    'subject_ISC': subj_isc.tolist(),
    'age': [float(ages.loc[ages['a_number']==s[4:], 'mri_age_yr'].values[0]) 
            for s in cleaned_subs]}

    df = pd.DataFrame(x)

    sns.lmplot(data=df, x="age", y="subject_ISC")
    # change y axis to be between 0 and 1
    #plt.ylim(0, 1)
    plt.title("ISC vs Age")
    plt.xlabel("Age (years)")
    plt.ylabel("Mean ISC across components")
    # save the figure
    plt.savefig(FIGURES_DIR / "isc_vs_age.png")
    print("ISC vs Age plot saved to figures directory.")