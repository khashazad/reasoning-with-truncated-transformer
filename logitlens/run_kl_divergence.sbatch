#!/bin/bash
#SBATCH --job-name=logitlens_kl
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a4500:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=27G
#SBATCH --time=02:00:00
#SBATCH --output=logitlens_%j.log
#SBATCH --error=logitlens_%j.err

# --- 1. Setup Paths ---
# SCRATCH: For heavy temporary files (Environments, Models)
SCRATCH_ROOT=$(ls -d /scratch/scratch-space/expires-* | head -n 1)

# HOME: For the final result plot (Permanent, Safe)
RESULT_DIR="$HOME/truncated-transformer-reasoning/logitlens/logitlens_results"
mkdir -p "$RESULT_DIR"

# CRITICAL FIX: Switch to the directory where you submitted the job
# This ensures the script finds 'logitlens-kl-divergence.py'
cd "$SLURM_SUBMIT_DIR"

echo ">>> CONFIGURATION <<<"
echo "Running in: $(pwd)"
echo "Temporary Scratch: $SCRATCH_ROOT"
echo "Permanent Results: $RESULT_DIR"

# --- 2. Environment Setup (In Scratch) ---
ENV_PATH="$SCRATCH_ROOT/$USER/logitlens2_env"

# Create environment if it doesn't exist (Self-Healing)
if [ ! -d "$ENV_PATH" ]; then
    echo "Creating virtual environment in scratch: $ENV_PATH"
    python3 -m venv "$ENV_PATH"
    source "$ENV_PATH/bin/activate"
    pip install --upgrade pip

    # INSTALLS: Mapping your imports to pip packages
    # torch, matplotlib, numpy, transformers, datasets, accelerate
    pip install torch transformers datasets matplotlib numpy accelerate
else
    echo "Activating existing environment: $ENV_PATH"
    source "$ENV_PATH/bin/activate"

    pip install torch transformers datasets matplotlib numpy accelerate

    # Optional: Ensure packages are present even in existing env
    # pip install --upgrade torch transformers datasets matplotlib numpy accelerate
fi

# --- 3. Set Model Cache (In Scratch) ---
# Keep the 1.5B model weights in scratch to save your Home quota
export HF_HOME="$SCRATCH_ROOT/$USER/hf_cache"
mkdir -p "$HF_HOME"

# --- 4. Run Script ---
echo "Starting script..."

# Note: Ensure the filename matches exactly what is on your disk (dashes vs underscores)
python logitlens-kl-divergence.py \
    --output_file "$RESULT_DIR/kl_divergence_plot_${SLURM_JOB_ID}.png" \
    --n_problems 500

echo ">>> JOB COMPLETE <<<"
echo "Your plot is saved at: $RESULT_DIR/kl_divergence_plot_${SLURM_JOB_ID}.png"
