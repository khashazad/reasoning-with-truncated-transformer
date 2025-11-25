import argparse
import os

import torch
import transformers

def layer_container(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise ValueError("layers not found")


def truncate_once(model_name, layer_idx, keep_last, device, out_dir):
    tok = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    device_map = "auto" if device == "auto" else {"": device}
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=device_map,
        trust_remote_code=True,
    )
    layers = layer_container(model)
    if layer_idx >= len(layers):
        return
    kept = list(layers[: layer_idx + 1])
    if keep_last and layer_idx < len(layers) - 1:
        kept.append(layers[-1])
    new_layers = torch.nn.ModuleList(kept)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        model.model.layers = new_layers
    elif hasattr(model, "layers"):
        model.layers = new_layers
    model.config.num_hidden_layers = len(new_layers)
    if getattr(model.config, "layer_types", None):
        model.config.layer_types = model.config.layer_types[: len(new_layers)]
    suffix = "keeplast" if keep_last else "no-last"
    path = os.path.join(
        out_dir, model_name.replace("/", "_"), f"layer{layer_idx}_{suffix}"
    )
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tok.save_pretrained(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["Qwen/Qwen2.5-Math-1.5B"])
    parser.add_argument("--layer_index_start", type=int, default=20)
    parser.add_argument("--keep_last_layer", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(
            "reasoning-with-sampling", "llm_experiments", "truncated_models"
        ),
    )
    args = parser.parse_args()

    for model_name in args.models:
        config = transformers.AutoConfig.from_pretrained(
            model_name, trust_remote_code=True
        )
        for idx in range(args.layer_index_start, config.num_hidden_layers):
            truncate_once(
                model_name, idx, args.keep_last_layer, args.device, args.output_dir
            )


if __name__ == "__main__":
    main()
