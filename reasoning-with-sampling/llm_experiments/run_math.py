import os
import json
import random
from tqdm import tqdm
import argparse
import pandas as pd
import torch
import transformers


def layer_container(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise ValueError("layers not found")


def truncate_model(model, layer_idx, keep_last):
    layers = layer_container(model)
    if layer_idx >= len(layers):
        return model
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
    return model


from grader_utils.parse_utils import parse_answer
from constants import *
from power_samp_utils import (
    AutoregressiveSampler,
    format_prompt,
    mcmc_power_samp,
)

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_str", action="store", type=str, default="results/", dest="save_str"
    )
    parser.add_argument(
        "--model",
        action="store",
        default="Qwen_Qwen2.5-Math-1.5Bb",
        type=str,
    )
    parser.add_argument(
        "--temperature", action="store", default=0.25, type=float, dest="temperature"
    )
    parser.add_argument("--dataset", action="store", default="MATH", type=str)
    parser.add_argument("--cot", action="store", type=bool, default=True)
    parser.add_argument("--mcmc_steps", action="store", type=int, default=10)
    parser.add_argument(
        "--device",
        action="store",
        type=str,
        dest="device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch_idx", action="store", type=int, default=0)
    parser.add_argument("--seed", action="store", type=int, default=0)
    parser.add_argument(
        "--layer_idx",
        action="store",
        type=int,
        default=20,
        help="Layer index for truncated model",
    )
    parser.add_argument(
        "--hf_model_path",
        action="store",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B",
        help="Base Hugging Face model to truncate locally before sampling.",
    )
    parser.add_argument(
        "--keep_last_layer",
        action="store_true",
        help="Use truncated model with last layer kept (layerX_keeplast), otherwise use layerX_no-last",
    )

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
    keep_last_layer = args.keep_last_layer

    save_str = os.path.join(args.save_str, model_str.replace("/", "_"))
    os.makedirs(save_str, exist_ok=True)

    print(f"Base Model: {model_str}")
    print(f"Device: {device}")
    print(f"MCMC Steps: {mcmc_steps}")
    print(f"Truncated Layer Index: {layer_idx}")
    print(f"Keep Last Layer: {keep_last_layer}")

    if dataset_name == "MATH":
        json_file = "data/MATH500.json"
        if not os.path.exists(json_file):
            json_file = "../data/MATH500.json"
            if not os.path.exists(json_file):
                print("Warning: MATH500.json not found. Please check path.")

        dataset = json.load(open(json_file, "r"))

    print("Dataset loaded")

    HF_TOKEN = "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"
    base_model_path = args.hf_model_path or model_str
    base_is_local = os.path.isdir(base_model_path)

    if not base_is_local and HF_TOKEN is None:
        raise ValueError(
            "HF_TOKEN is missing. Please set the environment variable or hardcode it."
        )

    hf_kwargs = {"trust_remote_code": True}
    if base_is_local:
        hf_kwargs["local_files_only"] = True
    else:
        hf_kwargs["token"] = HF_TOKEN

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model_path,
        **hf_kwargs,
    )

    device_map = "auto" if device == "auto" else {"": device}
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype="auto",
        device_map=device_map,
        **hf_kwargs,
    )
    hf_model = truncate_model(hf_model, layer_idx, keep_last_layer).to(device)

    autoreg_sampler = AutoregressiveSampler(hf_model, tokenizer, device)

    results = []

    start = 100 * args.batch_idx
    end = 100 * (args.batch_idx + 1)
    end = min(end, len(dataset))

    for i, data in tqdm(
        enumerate(dataset[start:end]), total=end - start, desc="Truncated-PS Benchmark"
    ):
        question = data["prompt"]
        answer = data["answer"]

        input_text = format_prompt(question, "qwen_math", tokenizer, cot)
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        prefx = [idx.item() for idx in input_ids[0]]

        # Using standard mcmc_power_samp on the truncated model
        mcmc_out, _, _, accept_ratio = mcmc_power_samp(
            autoreg_sampler, prefx, temp, mcmc_steps, max_new_tokens=3072, block_num=16
        )

        mcmc_completion = tokenizer.decode(
            torch.tensor(mcmc_out[len(prefx) :], dtype=torch.long).cpu(),
            skip_special_tokens=True,
        )
        mcmc_answer_parsed = parse_answer(mcmc_completion)

        print(f"Q: {question[:50]}...")
        print(f"A: {mcmc_answer_parsed}")
        print(f"Ratio: {accept_ratio}")

        results.append(
            {
                "question": question,
                "correct_answer": answer,
                "mcmc_completion": mcmc_completion,
                "mcmc_answer": mcmc_answer_parsed,
                "acceptance_ratio": accept_ratio,
            }
        )

    out_file = os.path.join(
        save_str,
        f"truncated_ps_results_layer{layer_idx}_steps{mcmc_steps}_temp{temp}_batch{args.batch_idx}.csv",
    )
    df = pd.DataFrame(results)
    df.to_csv(out_file, index=False)
    print(f"Saved results to {out_file}")
