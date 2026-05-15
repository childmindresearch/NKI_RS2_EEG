#%%

import os
from pathlib import Path


#=================
# Paths
#=================

# Root directory of the repository
ROOT_DIR = Path(__file__).parent.parent.parent


# Directory containing raw NWB files
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
# RAW_DATA_DIR = Path("/path/to/raw/data")  # Update this path as needed

# Directory to save results
RESULTS_DIR = ROOT_DIR / "results"
# Directory to save figures
FIGURES_DIR = ROOT_DIR / "figures"
# Directory to save Derivatives
DERIVATIVES_DIR = ROOT_DIR / "data" / "derivatives"
# DERIVATIVES_DIR = Path("/path/to/derivatives")  # Update this path as needed

CONCAT_DATA_DIR = DERIVATIVES_DIR / "sub-ALL_ses-MOBI1A_task-passivepresent_run-ALL_eeg.npy"


#%%
# ============================================================
# RECORDING PARAMETERS
# ============================================================

# Sampling frequency in Hz
FS = 250

# Expected number of EEG channels
N_CHANNELS = 64

# List of expected channel names (in order)
CHANNEL_NAMES = ['Fp1', 'Fz', 'F3', 'F7', 'F9','FC5',
                 'FC1','C3','T7','CP5','CP1','Pz',
                 'P3','P7','P9','O1','Oz','O2',
                 'P10','P8','P4','CP2','CP6','T8',
                 'C4','Cz','FC2','FC6','F10','F8',
                 'F4','Fp2','AF7','AF3','AFz','F1',
                 'F5','FT7','FC3','C1','C5','TP7',
                 'CP3','P1','P5','PO7','PO3','Iz',
                 'POz','PO4','PO8','P6','P2','CPz',
                 'CP4','TP8','C6','C2','FC4','FT8',
                 'F6','F2','AF4','AF8']



# Line noise frequency (50 Hz in Europe, 60 Hz in North America)
LINE_NOISE_HZ = 60

SESSION_ID = "MOBI1A"
TASK_ID = "passivepresent"

# ============================================================
# QUALITY METRIC THRESHOLDS
# ============================================================

# Variance
VARIANCE_MIN = 0.1          # Below this → likely flat channel
VARIANCE_MAX = 1000.0       # Above this → likely artifact

# Peak-to-peak amplitude (in microvolts)
PEAK_TO_PEAK_MAX = 500.0    # Above this → likely artifact

# Kurtosis
KURTOSIS_MAX = 5.0          # Above this → likely non-brain artifact

# Flat signal detection
FLAT_THRESHOLD = 1e-6       # Samples within this range of each other = flat
FLAT_MIN_DURATION_SEC = 1.0 # Minimum duration to count as a flat segment

# Muscle artifact: high-frequency power threshold
MUSCLE_FREQ_BAND = (40, 100)    # Hz range to check
MUSCLE_POWER_MAX = 50.0         # Above this → likely muscle artifact

# Low-frequency drift
DRIFT_FREQ_BAND = (0.01, 0.1)   # Hz range to check
DRIFT_POWER_MAX = 100.0         # Above this → likely drift

# Line noise
LINE_NOISE_POWER_MAX = 50.0     # Above this → too much electrical interference

# Minimum percent of clean data to keep a participant
MIN_CLEAN_DATA_PCT = 50.0

# Minimum percent of participants needed at a given sample to keep it
MIN_PARTICIPANT_COVERAGE_PCT = 70.0

# Maximum number of bad channels before excluding a participant
MAX_BAD_CHANNELS = 10


# ============================================================
# SLIDING WINDOW PARAMETERS
# ============================================================

# Window size for rolling metrics (in seconds)
WINDOW_SIZE_SEC = 5

# Step size between windows (in seconds)
WINDOW_STEP_SEC = 1

# Derived window sizes in samples (do not edit)
WINDOW_SIZE_SAMPLES = WINDOW_SIZE_SEC * FS
WINDOW_STEP_SAMPLES = WINDOW_STEP_SEC * FS


# ============================================================
# CCA / ISC PARAMETERS
# ============================================================

# Regularization parameter for CCA
GAMMA = 0.1

# Number of CCA components to retain
N_COMPONENTS = 1

# ISC threshold below which a participant is flagged as an outlier
ISC_OUTLIER_THRESHOLD = 0.01


# ============================================================
# FIGURE SETTINGS
# ============================================================

# Output format for saved figures
FIGURE_FORMAT = "png"       # "png", "svg", or "pdf"

# DPI for rasterized outputs
FIGURE_DPI = 300

# Default figure size in inches (width, height)
FIGURE_SIZE = (16, 8)

# Colormap for heatmaps
HEATMAP_COLORMAP = "RdYlGn"  # Red = bad, green = good

# Whether to show figures interactively or just save them
SHOW_FIGURES = False