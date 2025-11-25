#!/bin/bash --login
#SBATCH --job-name=trunc_ps_math_hf_2
#SBATCH --partition=gpunodes
#SBATCH -t 0-23:59
#SBATCH -c 4
#SBATCH --mem=24GB
#SBATCH --gres=gpu:rtx_a2000:1
#SBATCH --output=slurm_jobs/%a.out
#SBATCH --error=slurm_jobs/%a.err
#SBATCH --mail-user=khashayar1924@gmail.com
#SBATCH --mail-type=ALL

RESULT_DIR="./results"
mkdir -p "$RESULT_DIR"

echo ">>> CONFIGURATION <<<"
echo "Running in: $(pwd)"

SCRATCH_ROOT=$(ls -d /scratch/scratch-space/expires-* | head -n 1)

if [ -z "$SCRATCH_ROOT" ]; then
    echo "ERROR: Could not find a scratch space directory." >&2
    exit 1
fi
echo "Using scratch space: $SCRATCH_ROOT"

ENV_NAME="math_trunc_run_env_ganni"
ENV_PATH="$SCRATCH_ROOT/$ENV_NAME"
# --- ADD: Set PYTHONPATH ---
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# --- ADD: Set HF_HOME for model cache ---
export HF_HOME="$SCRATCH_ROOT/hf_cache"
mkdir -p "$HF_HOME"

if [ ! -d "$ENV_PATH" ]; then
    echo "Creating virtual environment at $ENV_PATH..."
    python3 -m venv "$ENV_PATH"
fi

# --- ADD: Activate the environment (CRITICAL FIX) ---
source "$ENV_PATH/bin/activate"

echo "Installing dependencies..."
pip install --upgrade pip
pip install --no-cache-dir torch transformers pandas tqdm accelerate datasets huggingface_hub

echo "Running script..."
python run_trunc_math.py

echo ">>> JOB COMPLETE <<<"