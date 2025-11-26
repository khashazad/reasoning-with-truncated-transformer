import os
import json
import random
from tqdm import tqdm
import argparse
import pandas as pd
import torch
import transformers

from grader_utils.parse_utils import parse_answer
from constants import *
from power_samp_utils import AutoregressiveSampler, format_prompt, mcmc_power_samp


def _get_layer_container(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model, "layers"
    if hasattr(model, "layers"):
        return model, "layers"
    return None, None


def truncate_model(model, layer_idx, keep_last_layer=True):
    container, attr = _get_layer_container(model)
    if container is None:
        print("Warning: Could not find layer list to truncate. Running full model.")
        return model

    layers = getattr(container, attr)
    total_layers = len(layers)
    if layer_idx >= total_layers:
        print(f"Layer index {layer_idx} >= total layers {total_layers}. Running full model.")
        return model

    print(
        f"Truncating model layers to first {layer_idx + 1} layers"
        + (" and keeping last layer..." if keep_last_layer else "...")
    )

    truncated_layers = list(layers[: layer_idx + 1])
    if keep_last_layer and layer_idx < total_layers - 1:
        truncated_layers.append(layers[-1])

    setattr(container, attr, torch.nn.ModuleList(truncated_layers))
    model.config.num_hidden_layers = len(truncated_layers)

    # CRITICAL FIX: Update layer_idx for each layer to match new position
    # This fixes the KV cache IndexError when using keep_last_layer=True
    for new_idx, layer in enumerate(truncated_layers):
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'layer_idx'):
            layer.self_attn.layer_idx = new_idx

    if getattr(model.config, "layer_types", None):
        layer_types = list(model.config.layer_types)
        new_layer_types = layer_types[: layer_idx + 1]
        if keep_last_layer and layer_idx < len(layer_types) - 1:
            new_layer_types.append(layer_types[-1])
        model.config.layer_types = new_layer_types

    print(f"New Total Layers: {model.config.num_hidden_layers}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_str", action = "store", type = str, default = "results/",  dest = "save_str")
    parser.add_argument("--model", action = "store", default = "Qwen/Qwen2.5-Math-1.5B", type = str)
    parser.add_argument("--temperature", action = "store", default = 0.5, type = float, dest = "temperature")
    parser.add_argument("--dataset", action = "store", default = "MATH", type = str)
    parser.add_argument("--cot", action = "store", type = bool, default = False)
    parser.add_argument("--mcmc_steps", action = "store", type = int, default = 10)
    parser.add_argument("--device", action = "store", type = str, dest = "device", default = "cuda" if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--batch_idx", action = "store", type = int, default = 0)
    parser.add_argument("--seed", action = "store", type = int, default = 0)
    parser.add_argument("--layer_idx", action = "store", type = int, default =24, help="Truncate model to this layer index (0-indexed)")

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_str = args.model
    device = args.device
    dataset_name = args.dataset
    cot = args.cot
    temp = args.temperature
    mcmc_steps = args.mcmc_steps
    layer_idx = args.layer_idx
    keep_last_layer = True

    save_str = os.path.join(args.save_str, model_str.replace("/", "_"))
    os.makedirs(save_str, exist_ok=True)

    print(f"Model: {model_str}")
    print(f"Device: {device}")
    print(f"MCMC Steps: {mcmc_steps}")
    print(f"Truncating at Layer: {layer_idx}")
    print(f"Keep Last Layer: {keep_last_layer}")

    if dataset_name == "MATH":
        json_file = 'data/MATH500.json'
        if not os.path.exists(json_file):
             json_file = '../data/MATH500.json'
             if not os.path.exists(json_file):
                 print("Warning: MATH500.json not found. Please check path.")

        dataset = json.load(open(json_file, "r"))

    print("Dataset loaded")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_str, trust_remote_code=True)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(model_str, torch_dtype="auto", device_map="auto", trust_remote_code=True).to(device)

    print(f"Original Total Layers: {hf_model.config.num_hidden_layers}")

    hf_model = truncate_model(hf_model, layer_idx, keep_last_layer)

    # Autoregressive Sampler with Truncated Model
    autoreg_sampler = AutoregressiveSampler(hf_model, tokenizer, device)

    results = []

    start = 20*args.batch_idx
    end = 20*(args.batch_idx+1)
    end = min(end, len(dataset))

    for i, data in tqdm(enumerate(dataset[start:end]), total=end-start, desc="Truncated-PS Benchmark"):
        question = data["prompt"]
        answer = data["answer"]

        input_text = format_prompt(question, "qwen_math", tokenizer, cot)
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        prefx = [idx.item() for idx in input_ids[0]]

        # Using standard mcmc_power_samp on the truncated model
        mcmc_out, _, _, accept_ratio = mcmc_power_samp(
            autoreg_sampler, prefx, temp, mcmc_steps, max_new_tokens=512, block_num=16
        )

        mcmc_completion = tokenizer.decode(torch.tensor(mcmc_out[len(prefx):], dtype=torch.long).cpu(), skip_special_tokens=True)
        mcmc_answer_parsed = parse_answer(mcmc_completion)

        print(f"Q: {question}")
        print(f"Completion: {mcmc_completion}")
        print(f"A: {mcmc_answer_parsed}")
        print(f"Ratio: {accept_ratio}")

        results.append({
            "question": question,
            "correct_answer": answer,
            "mcmc_completion": mcmc_completion,
            "mcmc_answer": mcmc_answer_parsed,
            "acceptance_ratio": accept_ratio
        })

    out_file = os.path.join(save_str, f"truncated_ps_results_layer{layer_idx}_steps{mcmc_steps}_temp{temp}_batch{args.batch_idx}.csv")
    df = pd.DataFrame(results)
    try:
        df.to_csv(out_file, index=False)
        print(f"Saved results to {out_file}")
    except Exception as e:
        print(f"ERROR: Failed to save CSV to {out_file}: {e}")
        print("DUMPING RESULTS TO STDOUT:")
        print(df.to_csv(index=False))
