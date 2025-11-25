#!/bin/bash

RESULT_DIR="./results"
mkdir -p "$RESULT_DIR"

echo ">>> CONFIGURATION <<<"
echo "Running in: $(pwd)"

ENV_PATH="$HOME/math_trunc_run_env"

if [ ! -d "$ENV_PATH" ]; then
    python3 -m venv "$ENV_PATH"
    source "$ENV_PATH/bin/activate"
    python3 -m pip install --upgrade pip
    python3 -m pip install torch transformers pandas accelerate datasets huggingface_hub
    python3 -m pip install "git+https://github.com/tqdm/tqdm.git@devel#egg=tqdm"
else
    source "$ENV_PATH/bin/activate"
    python3 -m pip install torch transformers pandas accelerate datasets huggingface_hub
    python3 -m pip install "git+https://github.com/tqdm/tqdm.git@devel#egg=tqdm"
fi

python3 run_trunc_math.py
