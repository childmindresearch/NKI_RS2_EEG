from __future__ import annotations

import argparse
import json
import logging
import os

import pathlib
from typing import Any
from pathlib import Path
import mne
import mne_bids
import numpy as np
import pandas as pd
import pyprep
from nki_rs2_eeg import read_file , write_file
from nki_rs2_eeg.read_file import read_raw_nwb
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

mne.viz.set_3d_backend("pyvistaqt")

file = "/data3/cdb/MOBI_LAB/NKI_RS2/ET_DATA/sub-A00064418/ses-MOBI1A/sub-A00064418_ses-MOBI1A_task-passivepresent_run-01_MoBI.nwb"
raw, channels = read_file.read_raw_nwb(file)
raw.filter(l_freq=1, h_freq=125).resample(250)

onsets = raw.annotations.onset
descriptions = raw.annotations.description

# Example: Get the onset time for the first occurrence of each
t_start = onsets[descriptions == 'Onset Movie'][0]
raw_cropped = raw.copy().crop(tmin=t_start, tmax=t_start + 60)

import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.time_frequency import psd_array_multitaper

def plot_parietal_psd_before_after_notch(raw, channel=None, fmin=1, fmax=120, bandwidth=2.0):
    """
    Plot power spectrum of a parietal channel before and after 60Hz notch filtering.
    Uses multitaper spectral decomposition.
    
    Parameters:
    -----------
    raw : mne.io.Raw
        MNE Raw object
    channel : str or None
        Specific channel name. If None, auto-selects first parietal channel.
    fmin : float
        Minimum frequency for PSD (default: 1 Hz)
    fmax : float
        Maximum frequency for PSD (default: 120 Hz)
    bandwidth : float
        Multitaper bandwidth in Hz (default: 2.0)
    
    Returns:
    --------
    fig : matplotlib.Figure
    channel_name : str
        The channel used
    """
    
    # ── 1. Select parietal channel ─────────────────────────────────────────────
    if channel is None:
        parietal_picks = mne.pick_channels_regexp(raw.ch_names, regexp=r'P[z0-9]|CP[z0-9]')
        if len(parietal_picks) == 0:
            raise ValueError("No parietal channels found. Specify a channel name manually.")
        channel = raw.ch_names[parietal_picks[0]]
        print(f"Auto-selected parietal channel: {channel}")
    
    ch_idx = raw.ch_names.index(channel)
    sfreq = raw.info['sfreq']
    
    # ── 2. Extract raw signal (before filtering) ───────────────────────────────
    data_before, _ = raw[ch_idx, :]          # shape: (1, n_times)
    data_before = data_before[0]             # flatten to (n_times,)
    print("Applying the multi-taper notch filter to the raw data to remove noise........")
    # ── 3. Apply notch filter at 60 Hz (and harmonics) ────────────────────────
    notch_freqs = np.arange(60, min(sfreq / 2, fmax + 1), 60)
    raw_clean = raw.copy().notch_filter(
        freqs=notch_freqs,
        method='spectrum_fit',      # <-- multitaper spectral decomposition
        mt_bandwidth=bandwidth,     # multitaper half-bandwidth (Hz)
        p_value=0.05,               # significance threshold for line detection
        picks=[ch_idx],
        verbose=False
    )
    data_after = raw_clean[ch_idx, :][0][0]
    print("Multi-taper notch filter applied successfully.")
    print("creating the power spectrum plot before and after notch filtering........")

    # ── 4. Multitaper PSD ──────────────────────────────────────────────────────
    psd_before, freqs = psd_array_multitaper(
        data_before[np.newaxis, :],
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        bandwidth=bandwidth,
        normalization='full',
        verbose=False
    )
    psd_before = psd_before[0]   # (n_freqs,)

    psd_after, _ = psd_array_multitaper(
        data_after[np.newaxis, :],
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        bandwidth=bandwidth,
        normalization='full',
        verbose=False
    )
    psd_after = psd_after[0]

    # Convert to dB
    psd_before_db = 10 * np.log10(psd_before)
    psd_after_db  = 10 * np.log10(psd_after)
    diff_db       = psd_after_db - psd_before_db
    print("Multitaper PSD computed for both before and after notch filtering.")

    print("Plotting the PSD before and after notch filtering with band shading and annotations........")
    # ── 5. Plot ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.patch.set_facecolor('#0f1117')
    for ax in axes:
        ax.set_facecolor('#1a1d27')
        ax.tick_params(colors='#c9d1d9')
        ax.yaxis.label.set_color('#c9d1d9')
        ax.xaxis.label.set_color('#c9d1d9')
        ax.title.set_color('#e6edf3')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')

    # EEG band shading
    bands = [
        ('Delta',  0.5,  4,  '#3b4a6b'),
        ('Theta',  4,    8,  '#3b5e4a'),
        ('Alpha',  8,    13, '#5e5a2e'),
        ('Beta',   13,   30, '#5e3b2e'),
        ('Gamma',  30,   80, '#4a2e5e'),
    ]
    for ax in axes[:1]:
        for bname, b1, b2, bcolor in bands:
            ax.axvspan(b1, b2, alpha=0.25, color=bcolor, zorder=0)
            mid = (b1 + b2) / 2
            ypos = ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -1
            ax.text(mid, ax.get_ylim()[0], bname, color='#8b949e',
                    fontsize=7, ha='center', va='bottom', style='italic')

    # Top panel: before & after PSD
    ax0 = axes[0]
    ax0.plot(freqs, psd_before_db, color='#58a6ff', lw=1.5,
             label='Before (original)', alpha=0.9)
    ax0.plot(freqs, psd_after_db,  color='#3fb950', lw=1.5,
             label='After notch filter', alpha=0.9)

    # Mark 60 Hz harmonics
    notch_freqs = np.arange(60, fmax + 1, 60)
    for nf in notch_freqs:
        ax0.axvline(nf, color='#f85149', lw=0.8, ls='--', alpha=0.7)
        ax0.text(nf + 0.5, ax0.get_ylim()[1] if ax0.get_ylim()[1] != 1 else 0,
                 f'{int(nf)} Hz', color='#f85149', fontsize=7, va='top')

    # Band shading on top panel
    for bname, b1, b2, bcolor in bands:
        ax0.axvspan(b1, b2, alpha=0.18, color=bcolor, zorder=0)

    ax0.set_ylabel('Power (dB)', fontsize=11)
    ax0.set_title(f'Multitaper PSD — Channel {channel}  |  Bandwidth = {bandwidth} Hz',
                  fontsize=12, fontweight='bold', pad=10)
    ax0.legend(facecolor='#1a1d27', edgecolor='#30363d',
               labelcolor='#c9d1d9', fontsize=9, loc='upper right')
    ax0.grid(True, alpha=0.15, color='#8b949e')

    # Bottom panel: difference
    ax1 = axes[1]
    ax1.fill_between(freqs, diff_db, 0,
                     where=(diff_db < 0), color='#3fb950', alpha=0.6,
                     label='Power removed')
    ax1.fill_between(freqs, diff_db, 0,
                     where=(diff_db >= 0), color='#f0883e', alpha=0.6,
                     label='Power added (artifact)')
    ax1.axhline(0, color='#8b949e', lw=0.8, ls='-')
    for nf in notch_freqs:
        ax1.axvline(nf, color='#f85149', lw=0.8, ls='--', alpha=0.7)

    ax1.set_xlabel('Frequency (Hz)', fontsize=11)
    ax1.set_ylabel('Δ Power (dB)', fontsize=11)
    ax1.set_title('Difference (After − Before)', fontsize=11, pad=8)
    ax1.legend(facecolor='#1a1d27', edgecolor='#30363d',
               labelcolor='#c9d1d9', fontsize=9, loc='lower right')
    ax1.grid(True, alpha=0.15, color='#8b949e')

    plt.tight_layout(h_pad=1.5)
    plt.savefig('/home/bgonzalez/NKI_RS2_EEG/figures/psd_before_after_notch.png',
                dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()
    print(f"Figure saved → /home/bgonzalez/NKI_RS2_EEG/figures/psd_before_after_notch.png")
    return fig, channel


if __name__ == "__main__":

    fig, ch = plot_parietal_psd_before_after_notch(raw_cropped, fmin=1, fmax=150, bandwidth=2.0)