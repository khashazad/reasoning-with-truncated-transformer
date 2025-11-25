HF_REPO_ID="khashazad/Qwen2.5-Math-1.5B-truncated"  # replace with your repo id

python truncate_models.py \
  --models Qwen/Qwen2.5-Math-1.5B \
  --layer_index_start 20 \
  --keep_last_layer \
  --output_dir reasoning-with-sampling/models \
  --device cpu
