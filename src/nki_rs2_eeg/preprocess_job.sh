#!/bin/bash
#SBATCH --job-name=eeg_preprocess
#SBATCH --output=/home/bgonzalez/NKI_RS2_EEG/logs/preprocess%j.out
#SBATCH --error=/home/bgonzalez/NKI_RS2_EEG/logs/preprocess%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G

set -euo pipefail

export PATH=/usr/local/bin:/usr/bin:/bin:$PATH

# --- Configuration ──────────────────────────────
VENV_DIR="/home/bgonzalez/NKI_RS2_EEG/.venv"
SCRIPT="/home/bgonzalez/NKI_RS2_EEG/src/nki_rs2_eeg/noise_removal.py"
LOG_FILE="/home/bgonzalez/NKI_RS2_EEG/logs/job_$(date +%Y%m%d_%H%M%S).log"
# ─────────────────────────────────────────────────

log() {
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $*" >> "$LOG_FILE"
}

log "Starting job"
log "Host: ${SLURMD_NODENAME:-$(hostname)}"

# Activate virtual environment
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    log "ERROR: venv not found at $VENV_DIR"
    exit 1
fi

source "$VENV_DIR/bin/activate"
log "Activated venv: $VENV_DIR"
log "Python: $(which python)"

# Run the script
log "Running: $SCRIPT"
python "$SCRIPT"

log "Job complete"
