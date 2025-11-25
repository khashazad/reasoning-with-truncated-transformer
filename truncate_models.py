import argparse
import os
import torch
import transformers
from huggingface_hub import HfApi, create_repo, get_token


def _get_layer_container(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model, "layers"
    if hasattr(model, "layers"):
        return model, "layers"
    return None, None


def _get_hf_token(hf_token=None):
    """
    Get Hugging Face token from argument, environment variable, or login cache.
    """
    if hf_token is not None:
        return hf_token
    # Try to get token from environment or login cache
    try:
        token = get_token()
        return token
    except Exception:
        # If get_token() fails or raises an exception, return None
        return None


def create_or_get_repo_id(model_str, hf_token=None):
    """
    Create a Hugging Face repo if it doesn't exist, or return the repo_id.
    The repo name is based on the model name with '-truncated' suffix.
    """
    token = _get_hf_token(hf_token)
    if token is None:
        raise RuntimeError(
            "No Hugging Face token found. Please provide one via --hf_token argument "
            "or set the HF_TOKEN environment variable. Alternatively, log in with "
            "'huggingface_hub.login()' or 'hf auth login'."
        )

    api = HfApi(token=token)

    # Get the current user's username
    try:
        user_info = api.whoami()
        username = user_info["name"]
    except Exception as exc:
        raise RuntimeError(f"Failed to get user info. Make sure you're logged in with a valid token: {exc}") from exc

    # Generate repo name from model name
    model_name = model_str.replace("/", "_")
    repo_name = f"{model_name}-truncated"
    repo_id = f"{username}/{repo_name}"

    # Create repo if it doesn't exist
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            token=token,
            exist_ok=True,
        )
        print(f"Created or verified repo: {repo_id}")
    except Exception as exc:
        raise RuntimeError(f"Failed to create repo {repo_id}: {exc}") from exc

    return repo_id


def upload_folder_to_hub(local_path, repo_id, path_in_repo, commit_message=None, hf_token=None):
    token = _get_hf_token(hf_token)
    api = HfApi(token=token)
    print(f"Uploading {local_path} to {repo_id} at '{path_in_repo}'")
    try:
        api.upload_folder(
            repo_id=repo_id,
            folder_path=local_path,
            path_in_repo=path_in_repo,
            commit_message=commit_message,
            repo_type="model",
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to upload {local_path} to {repo_id}/{path_in_repo}: {exc}") from exc
    print("Upload complete.")


def save_truncated_model(model_str, layer_idx, output_path, keep_last_layer=False, device="auto"):
    """
    Load a model, truncate it to the specified layer_idx (0-indexed),
    optionally keep the original last layer appended, and save the result.
    """
    print(f"Loading model: {model_str}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_str, trust_remote_code=True)

    device_map = "auto" if device == "auto" else {"": device}
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_str,
        torch_dtype="auto",
        device_map=device_map,
        trust_remote_code=True,
    )

    container, attr = _get_layer_container(model)
    if container is None:
        print("Warning: Could not find layer list to truncate.")
        return None

    layers = getattr(container, attr)
    total_layers = len(layers)
    print(f"Original layers: {total_layers}")

    if layer_idx >= total_layers:
        print(f"Layer index {layer_idx} >= total layers {total_layers}. No truncation performed.")
        return None

    print(f"Truncating model to first {layer_idx + 1} layers" + (" and keeping last layer." if keep_last_layer else "."))
    truncated_layers = list(layers[: layer_idx + 1])
    if keep_last_layer and layer_idx < total_layers - 1:
        truncated_layers.append(layers[-1])

    setattr(container, attr, torch.nn.ModuleList(truncated_layers))
    model.config.num_hidden_layers = len(truncated_layers)
    print(f"New Total Layers: {model.config.num_hidden_layers}")

    os.makedirs(output_path, exist_ok=True)
    print(f"Saving truncated model to {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    return model


def truncate_across_layers(
    model_str, layer_index_start, output_dir, keep_last_layer=False, device="auto", repo_id=None, hf_token=None
):
    config = transformers.AutoConfig.from_pretrained(model_str, trust_remote_code=True)
    total_layers = config.num_hidden_layers
    if layer_index_start >= total_layers:
        print(f"Start index {layer_index_start} >= total layers {total_layers}. Skipping {model_str}.")
        return

    # Create repo if repo_id is not provided
    if repo_id is None:
        print("No repo_id provided. Creating a new Hugging Face repo...")
        repo_id = create_or_get_repo_id(model_str, hf_token)

    model_prefix = model_str.replace("/", "_")
    for layer_idx in range(layer_index_start, total_layers):
        suffix = "keeplast" if keep_last_layer else "no-last"
        out_path = os.path.join(output_dir, model_prefix, f"layer{layer_idx}_{suffix}")
        model = save_truncated_model(model_str, layer_idx, out_path, keep_last_layer, device)
        if model is None:
            continue

        path_in_repo = os.path.relpath(out_path, output_dir).replace(os.sep, "/")
        commit_message = f"Add {path_in_repo} truncated from {model_str}"
        upload_folder_to_hub(out_path, repo_id, path_in_repo, commit_message, hf_token)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Truncate and save multiple model variants.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Qwen/Qwen2.5-Math-1.5B"],
        help="One or more model identifiers to truncate.",
    )
    parser.add_argument(
        "--layer_index_start",
        type=int,
        default=0,
        help="First layer index (0-indexed) to start producing truncated checkpoints.",
    )
    parser.add_argument(
        "--keep_last_layer",
        action="store_true",
        help="Keep the final transformer layer appended after truncation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/truncated",
        help="Base directory to save all truncated models.",
    )
    parser.add_argument(
        "--repo_ids",
        nargs="+",
        default=None,
        help="Optional Hugging Face repo ids to upload to (one per model in --models).",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Optional Hugging Face token; defaults to HF_TOKEN env var if not provided.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device placement for loading (e.g., auto, cpu, cuda).",
    )

    args = parser.parse_args()

    if args.repo_ids is not None and len(args.repo_ids) != len(args.models):
        raise ValueError("If provided, --repo_ids must have the same length as --models.")

    repo_ids = args.repo_ids or [None] * len(args.models)

    for model_name, repo_id in zip(args.models, repo_ids):
        truncate_across_layers(
            model_name,
            args.layer_index_start,
            args.output_dir,
            keep_last_layer=args.keep_last_layer,
            device=args.device,
            repo_id=repo_id,
            hf_token=args.hf_token,
        )
