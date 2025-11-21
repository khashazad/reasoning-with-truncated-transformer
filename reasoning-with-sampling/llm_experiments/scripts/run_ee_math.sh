#!/bin/bash
#SBATCH --job-name=ee_ps_math
#SBATCH -t 0-23:59
#SBATCH --mem=64000
#SBATCH --gres=gpu:1
#SBATCH --array=0-4

# --- map array id -> (batch_idx) ---
BATCH_IDX=${SLURM_ARRAY_TASK_ID}
SEED=42

# Setup Environment
module load python/3.12.5-fasrc01
module load cuda/12.4.1-fasrc01

# Adjust these paths to your setup
REPO_ROOT=$(pwd)/../..
export PYTHONPATH="$PYTHONPATH:$REPO_ROOT/llm_experiments"
export HF_HOME="$HOME/.cache/huggingface"

source activate psamp
cd $REPO_ROOT/llm_experiments

echo "Running EE-PS Benchmark Batch ${BATCH_IDX}"

# Optional: Run calibration first if not exists (only by task 0)
if [ ${BATCH_IDX} -eq 0 ]; then
    echo "Checking/Running Calibration..."
    python run_ee_math.py \
        --model="Qwen/Qwen2.5-Math-1.5B" \
        --layer_idx=14 \
        --calibrate \
        --batch_idx=0 \
        --save_str="results_ee" \
        --dataset="MATH" # Just to trigger loading
fi

# Ensure calibration finishes before others proceed (simple sleep hack, ideally use dependencies)
# sleep 60

# Run Benchmark
python run_ee_math.py \
  --batch_idx="${BATCH_IDX}" \
  --mcmc_steps=10 \
  --temperature=0.25 \
  --seed="${SEED}" \
  --model="Qwen/Qwen2.5-Math-1.5B" \
  --layer_idx=14 \
  --save_str="results_ee"

