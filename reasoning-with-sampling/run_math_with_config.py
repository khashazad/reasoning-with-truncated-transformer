@echo off
setlocal enabledelayedexpansion

set "LAYER_START=20"
set "LAYER_END=27"
set "SEED=42"
set "LAYER_IDX=20"
set "HF_REPO_BASE=khashazad/Qwen_Qwen2.5-Math-1.5B-truncated"
set "MODEL_NAME=Qwen_Qwen2.5-Math-1.5B"
set "KEEP_LAST_LAYER=true"

set "MODEL_NAME_PATH=%MODEL_NAME:/=_%"

set "RESULT_DIR=results"
if not exist "%RESULT_DIR%" mkdir "%RESULT_DIR%"

if /i "%KEEP_LAST_LAYER%"=="true" (
    set "SUFFIX=keeplast"
) else (
    set "SUFFIX=no-last"
)

set "HF_MODEL_PATH=%HF_REPO_BASE%/%MODEL_NAME_PATH%/layer%LAYER_IDX%_%SUFFIX%"

echo >>> CONFIGURATION <<<
echo Running in: %cd%
echo Results Directory: %RESULT_DIR%
echo Layer Index: %LAYER_IDX%
echo HF Model Path: %HF_MODEL_PATH%
echo Keep Last Layer: %KEEP_LAST_LAYER%

set "ENV_PATH=%USERPROFILE%\math_run_env"

if not exist "%ENV_PATH%" (
    python3 -m venv "%ENV_PATH%"
    call "%ENV_PATH%\Scripts\activate.bat"
    python3 -m pip install --upgrade pip
    python3 -m pip install torch transformers pandas tqdm accelerate datasets huggingface_hub
) else (
    call "%ENV_PATH%\Scripts\activate.bat"
    python3 -m pip install torch transformers pandas tqdm accelerate datasets huggingface_hub
)

python3 run_math.py ^
    --mcmc_steps=10 ^
    --temperature=0.25 ^
    --seed=%SEED% ^
    --model=%MODEL_NAME% ^
    --layer_idx=%LAYER_IDX% ^
    --hf_model_path=%HF_MODEL_PATH% ^
    --keep_last_layer ^
    --save_str=%RESULT_DIR%

endlocal
