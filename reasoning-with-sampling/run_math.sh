#!/bin/bash
LAYER_START=20
LAYER_END=27
SEED=42
LAYER_IDX=20
HF_REPO_BASE="khashazad/Qwen_Qwen2.5-Math-1.5B-truncated"
MODEL_NAME="Qwen_Qwen2.5-Math-1.5B"
KEEP_LAST_LAYER=true
MODEL_NAME_PATH=$(echo "$MODEL_NAME" | tr '/' '_')

RESULT_DIR="./results"
mkdir -p "$RESULT_DIR"

if [ "$KEEP_LAST_LAYER" = true ]; then
    SUFFIX="keeplast"
else
    SUFFIX="no-last"
fi

HF_MODEL_PATH="${HF_REPO_BASE}/${MODEL_NAME_PATH}/layer${LAYER_IDX}_${SUFFIX}"

echo ">>> CONFIGURATION <<<"
echo "Running in: $(pwd)"
echo "Results Directory: $RESULT_DIR"
echo "Layer Index: ${LAYER_IDX}"
echo "HF Model Path: ${HF_MODEL_PATH}"
echo "Keep Last Layer: ${KEEP_LAST_LAYER}"

ENV_PATH="$HOME/math_run_env"

if [ ! -d "$ENV_PATH" ]; then
    python3 -m venv "$ENV_PATH"
    source "$ENV_PATH/bin/activate"
else
    source "$ENV_PATH/bin/activate"
fi

pip install torch transformers pandas tqdm accelerate datasets huggingface_hub

python run_math.py \
    --mcmc_steps=10 \
    --temperature=0.25 \
    --seed="$SEED" \
    --model="$MODEL_NAME" \
    --layer_idx="$LAYER_IDX" \
    --hf_model_path="$HF_MODEL_PATH" \
    --keep_last_layer \
    --save_str="$RESULT_DIR"
