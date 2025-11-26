#!/bin/bash --login
#SBATCH --job-name=baseline_math_100q
#SBATCH --partition=gpunodes
#SBATCH -t 0-06:00
#SBATCH -c 4
#SBATCH --mem=24GB
#SBATCH --gres=gpu:rtx_a2000:1
#SBATCH --array=0-19
#SBATCH --output=slurm_jobs/baseline_%A_%a.out
#SBATCH --error=slurm_jobs/baseline_%A_%a.err
#SBATCH --mail-user=khashayar1924@gmail.com
#SBATCH --mail-type=ALL

# 100 questions per layer: 4 layer configs x 5 batches = 20 jobs
# Array index 0-19 maps to (layer_config, batch_idx):
#   task_id = layer_config_idx * 5 + batch_idx
#   layer_config_idx = task_id / 5
#   batch_idx = task_id % 5

NUM_BATCHES=5
LAYER_OPTIONS=(22 25 26 100)

LAYER_CONFIG_IDX=$((SLURM_ARRAY_TASK_ID / NUM_BATCHES))
BATCH_IDX=$((SLURM_ARRAY_TASK_ID % NUM_BATCHES))
LAYER_IDX=${LAYER_OPTIONS[$LAYER_CONFIG_IDX]}

RESULT_DIR="./results_baseline"
mkdir -p "$RESULT_DIR"

echo ">>> BASELINE CONFIGURATION (100 Questions/Layer) <<<"
echo "Running in: $(pwd)"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Layer Config Index: $LAYER_CONFIG_IDX"
echo "Layer Index: $LAYER_IDX"
echo "Batch Index: $BATCH_IDX"

# Check if output file already exists - skip if so
OUTPUT_FILE="$RESULT_DIR/Qwen_Qwen2.5-Math-1.5B/baseline_results_layer${LAYER_IDX}_temp0.5_batch${BATCH_IDX}.csv"
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

echo "Running BASELINE script with layer_idx=$LAYER_IDX, batch_idx=$BATCH_IDX..."
python run_baseline_math.py \
    --layer_idx=$LAYER_IDX \
    --model="Qwen/Qwen2.5-Math-1.5B" \
    --temperature=0.5 \
    --batch_idx=$BATCH_IDX \
    --save_str="results_baseline"

echo ">>> JOB COMPLETE <<<"

