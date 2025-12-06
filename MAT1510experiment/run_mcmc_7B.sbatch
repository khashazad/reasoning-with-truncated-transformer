#!/bin/bash --login
#SBATCH --job-name=mcmc_7B
#SBATCH --partition=gpunodes
#SBATCH -t 0-23:59
#SBATCH -c 4
#SBATCH --mem=48GB
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --array=0-29
#SBATCH --output=slurm_jobs/mcmc_7B_%A_%a.out
#SBATCH --error=slurm_jobs/mcmc_7B_%A_%a.err
#SBATCH --mail-user=email@email.com
#SBATCH --mail-type=ALL

# Qwen2.5-Math-7B: Layers 21-26, 5 batches each = 30 jobs
# Array index 0-29 maps to (layer_config, batch_idx):
#   layer_config_idx = task_id / 5
#   batch_idx = task_id % 5

NUM_BATCHES=5
LAYER_OPTIONS=(21 22 23 24 25 26)

LAYER_CONFIG_IDX=$((SLURM_ARRAY_TASK_ID / NUM_BATCHES))
BATCH_IDX=$((SLURM_ARRAY_TASK_ID % NUM_BATCHES))
LAYER_IDX=${LAYER_OPTIONS[$LAYER_CONFIG_IDX]}

MODEL="Qwen/Qwen2.5-Math-7B"
MODEL_SHORT="7B"
RESULT_DIR="./results_mcmc_${MODEL_SHORT}"
mkdir -p "$RESULT_DIR"

echo ">>> MCMC Configuration (${MODEL_SHORT}) <<<"
echo "Running in: $(pwd)"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL"
echo "Layer Index: $LAYER_IDX"
echo "Batch Index: $BATCH_IDX"

# Check if output file already exists - skip if so
OUTPUT_FILE="$RESULT_DIR/Qwen_Qwen2.5-Math-${MODEL_SHORT}/truncated_ps_results_layer${LAYER_IDX}_steps10_temp0.5_batch${BATCH_IDX}.csv"
if [ -f "$OUTPUT_FILE" ]; then
    echo ">>> OUTPUT FILE ALREADY EXISTS: $OUTPUT_FILE"
    echo ">>> SKIPPING THIS JOB"
    exit 0
fi

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

echo "Running MCMC script..."
python run_trunc_math.py \
    --model="$MODEL" \
    --layer_idx=$LAYER_IDX \
    --batch_idx=$BATCH_IDX \
    --mcmc_steps=10 \
    --max_new_tokens=1024 \
    --temperature=0.5 \
    --save_str="$RESULT_DIR"

echo ">>> JOB COMPLETE <<<"


