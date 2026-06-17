#%%
import logging
import os
import re

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
import pyprep
from nki_rs2_eeg import read_file
from nki_rs2_eeg.config import ( RAW_DATA_DIR, DERIVATIVES_DIR)




logger = logging.getLogger(__name__)
#%%
# Get only present and sherlock .nwb files
movies = ["passivepresent", "passivesherlock"]
pattern = re.compile("|".join(map(re.escape, movies)))
nwb_files = [
    os.path.join(dirpath, name)
    for dirpath, _, filenames in os.walk(RAW_DATA_DIR)
    for name in filenames
    if pattern.search(name) and name.endswith("_MoBI.nwb")
]
#%%
