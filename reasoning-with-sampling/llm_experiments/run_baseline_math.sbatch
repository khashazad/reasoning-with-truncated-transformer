#!/bin/bash --login
#SBATCH --job-name=baseline_math
#SBATCH --partition=gpunodes
#SBATCH -t 0-12:00
#SBATCH -c 4
#SBATCH --mem=24GB
#SBATCH --gres=gpu:rtx_a2000:1
#SBATCH --array=0-3
#SBATCH --output=slurm_jobs/baseline_%A_%a.out
#SBATCH --error=slurm_jobs/baseline_%A_%a.err
#SBATCH --mail-user=khashayar1924@gmail.com
#SBATCH --mail-type=ALL

# Map array task ID to layer_idx:
# 0 -> layer 22
# 1 -> layer 25
# 2 -> layer 26
# 3 -> layer 100 (full model)
LAYER_OPTIONS=(22 25 26 100)
LAYER_IDX=${LAYER_OPTIONS[$SLURM_ARRAY_TASK_ID]}

RESULT_DIR="./results_baseline"
mkdir -p "$RESULT_DIR"

echo ">>> BASELINE CONFIGURATION <<<"
echo "Running in: $(pwd)"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Layer Index: $LAYER_IDX"

SCRATCH_ROOT=$(ls -d /scratch/scratch-space/expires-* | head -n 1)

if [ -z "$SCRATCH_ROOT" ]; then
    echo "ERROR: Could not find a scratch space directory." >&2
    exit 1
fi
echo "Using scratch space: $SCRATCH_ROOT"

ENV_NAME="math_trunc_run_env_ganni"
ENV_PATH="$SCRATCH_ROOT/$ENV_NAME"

# Set PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Set HF_HOME for model cache
export HF_HOME="$SCRATCH_ROOT/hf_cache"
mkdir -p "$HF_HOME"

if [ ! -d "$ENV_PATH" ]; then
    echo "Creating virtual environment at $ENV_PATH..."
    python3 -m venv "$ENV_PATH"
fi

# Activate the environment
source "$ENV_PATH/bin/activate"

echo "Installing dependencies..."
pip install --upgrade pip
pip install --no-cache-dir torch transformers pandas tqdm accelerate datasets huggingface_hub

echo "Running BASELINE script with layer_idx=$LAYER_IDX..."
python run_baseline_math.py \
    --layer_idx=$LAYER_IDX \
    --model="Qwen/Qwen2.5-Math-1.5B" \
    --temperature=0.5 \
    --batch_idx=0 \
    --save_str="results_baseline"

echo ">>> JOB COMPLETE <<<"

