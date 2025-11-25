#!/bin/bash --login
#SBATCH --job-name=trunc_ps_math_hf
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

ENV_PATH="$HOME/math_trunc_run_env"

PYTHON_INTERPRETER="$ENV_PATH/bin/python"

PIP_EXECUTABLE="$ENV_PATH/bin/pip"

SCRATCH_ROOT=$(ls -d /scratch/scratch-space/expires-* | head -n 1)

if [ ! -d "$ENV_PATH" ]; then
    echo "Creating virtual environment at $ENV_PATH..."
    python3 -m venv "$ENV_PATH"
else
    echo "Virtual environment already exists at $ENV_PATH."
fi

echo "Installing dependencies using $PIP_EXECUTABLE..."

"$PIP_EXECUTABLE" install torch transformers pandas tqdm accelerate datasets huggingface_hub

echo "Running script with $PYTHON_INTERPRETER..."

"$PYTHON_INTERPRETER" run_trunc_math.py
