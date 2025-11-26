"""
Baseline generation script (no MCMC power sampling).
Runs standard model.generate() for comparison with MCMC results.
"""

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
from power_samp_utils import format_prompt


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
    parser.add_argument("--save_str", type=str, default="results_baseline/")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--dataset", type=str, default="MATH")
    parser.add_argument("--cot", type=bool, default=False)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--batch_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layer_idx", type=int, default=100, help="Truncate model to this layer index (100+ = full model)")

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_str = args.model
    device = args.device
    dataset_name = args.dataset
    cot = args.cot
    temp = args.temperature
    layer_idx = args.layer_idx
    keep_last_layer = True

    save_str = os.path.join(args.save_str, model_str.replace("/", "_"))
    os.makedirs(save_str, exist_ok=True)

    print(f"=== BASELINE (No MCMC) ===")
    print(f"Model: {model_str}")
    print(f"Device: {device}")
    print(f"Temperature: {temp}")
    print(f"Layer Index: {layer_idx}")
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
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_str, torch_dtype="auto", device_map="auto", trust_remote_code=True
    ).to(device)

    print(f"Original Total Layers: {hf_model.config.num_hidden_layers}")

    hf_model = truncate_model(hf_model, layer_idx, keep_last_layer)

    results = []

    # Same batch as MCMC runs: 20 questions per batch
    start = 20 * args.batch_idx
    end = 20 * (args.batch_idx + 1)
    end = min(end, len(dataset))

    for i, data in tqdm(enumerate(dataset[start:end]), total=end-start, desc="Baseline Generation"):
        question = data["prompt"]
        answer = data["answer"]

        input_text = format_prompt(question, "qwen_math", tokenizer, cot)
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

        # Standard generation (no MCMC)
        with torch.no_grad():
            output = hf_model.generate(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=temp,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        completion = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        answer_parsed = parse_answer(completion)

        print(f"Q: {question}")
        print(f"Completion: {completion[:200]}...")
        print(f"Parsed Answer: {answer_parsed}")
        print(f"Correct: {answer}")
        print("-" * 50)

        results.append({
            "question": question,
            "correct_answer": answer,
            "mcmc_completion": completion,  # Keep same column name for compatibility
            "mcmc_answer": answer_parsed,   # Keep same column name for compatibility
            "acceptance_ratio": 1.0         # N/A for baseline, use 1.0
        })

    out_file = os.path.join(save_str, f"baseline_results_layer{layer_idx}_temp{temp}_batch{args.batch_idx}.csv")
    df = pd.DataFrame(results)
    try:
        df.to_csv(out_file, index=False)
        print(f"Saved results to {out_file}")
    except Exception as e:
        print(f"ERROR: Failed to save CSV to {out_file}: {e}")
        print("DUMPING RESULTS TO STDOUT:")
        print(df.to_csv(index=False))

