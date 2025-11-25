#!/bin/bash

# --- HPC Job Cleanup and Submission Script ---

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting cleanup and job submission sequence..."
echo "----------------------------------------------"

# 1. Clean the General User Cache
# This command recursively deletes all files and directories inside ~/.cache/.
# The -f flag forces the deletion and suppresses most error messages (e.g., if a file is protected).
echo "1. Cleaning general user cache (~/.cache/*)..."
rm -rf "$HOME/.cache/*"
echo "   -> General cache cleaned successfully."

# 2. Clean the SLURM Job Logs Directory (Assuming 'slurm_jobs' is in the current directory)
# We check if the directory exists before trying to clear its contents to prevent errors.
SLURM_LOG_DIR="slurm_jobs"
if [ -d "$SLURM_LOG_DIR" ]; then
    echo "2. Cleaning existing SLURM job log files in '$SLURM_LOG_DIR'..."
    # The -f flag suppresses errors if no files are matched by the wildcard.
    rm -f "$SLURM_LOG_DIR"/*
    echo "   -> SLURM log directory contents cleared."
else
    echo "2. SLURM log directory '$SLURM_LOG_DIR' not found. Skipping log cleanup."
fi

# 3. Submit the New SLURM Job
SBATCH_SCRIPT="run_math.sbash"
if [ -f "$SBATCH_SCRIPT" ]; then
    echo "3. Submitting the SLURM job using '$SBATCH_SCRIPT'..."
    # Use 'sbatch' to submit the job script.
    sbatch "$SBATCH_SCRIPT"
    echo "   -> Job submission command executed."
else
    echo "3. Error: SLURM submission script '$SBATCH_SCRIPT' not found."
    exit 1
fi

echo "----------------------------------------------"
echo "Cleanup and submission sequence complete!"
echo "Check 'squeue -u \$USER' to monitor your job."
