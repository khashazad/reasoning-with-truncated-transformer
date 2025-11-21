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
from power_samp_utils import AutoregressiveSampler, format_prompt
from ee_utils import EarlyExitHead, calibrate_mid_layer, patch_model_with_early_exit
from power_samp_ee import mcmc_power_samp_ee

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_str", action = "store", type = str, default = "results/",  dest = "save_str")
    parser.add_argument("--model", action = "store", default = "Qwen/Qwen2.5-Math-1.5B", type = str)
    parser.add_argument("--temperature", action = "store", default = 0.25, type = float, dest = "temperature")
    parser.add_argument("--dataset", action = "store", default = "MATH", type = str)
    parser.add_argument("--cot", action = "store", type = bool, default = True)
    parser.add_argument("--mcmc_steps", action = "store", type = int, default = 10)
    parser.add_argument("--device", action = "store", type = str, dest = "device", default = "cuda" if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--batch_idx", action = "store", type = int, default = 0)
    parser.add_argument("--seed", action = "store", type = int, default = 0)
    parser.add_argument("--layer_idx", action = "store", type = int, default = 18, help="Exit layer index")
    parser.add_argument("--calibrate", action="store_true", help="Run calibration")
    
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

    save_str = os.path.join(args.save_str, model_str.replace("/", "_"))
    os.makedirs(save_str, exist_ok=True)

    print(f"Model: {model_str}")
    print(f"Device: {device}")
    print(f"MCMC Steps: {mcmc_steps}")
    print(f"Early Exit Layer: {layer_idx}")

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
    
    print(f"Total Layers: {hf_model.config.num_hidden_layers}")
    if layer_idx >= hf_model.config.num_hidden_layers:
        print(f"Warning: layer_idx {layer_idx} is out of bounds. Clamping to {hf_model.config.num_hidden_layers // 2}")
        layer_idx = hf_model.config.num_hidden_layers // 2

    ee_head = EarlyExitHead(hf_model, layer_idx, device)
    
    # Calibration (Linear Regression)
    calib_file = os.path.join(save_str, f"calibration_layer_{layer_idx}_linear.pt")
    if args.calibrate:
        print("Starting Calibration...")
        calib_data = []
        for item in dataset[:50]: 
            prompt = format_prompt(item["prompt"], "qwen_math", tokenizer, cot) 
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            calib_data.append(input_ids)
        
        weight, bias = calibrate_mid_layer(hf_model, calib_data, layer_idx, device)
        ee_head.set_calibration_params(weight, bias)
        
        torch.save({'weight': weight, 'bias': bias}, calib_file)
        print(f"Calibration saved to {calib_file}")
    elif os.path.exists(calib_file):
        print(f"Loading calibration from {calib_file}")
        calib_params = torch.load(calib_file)
        ee_head.set_calibration_params(calib_params['weight'], calib_params['bias'])
    else:
        print("No calibration found. Running uncalibrated (naive Logit Lens).")

    # Patch Model
    patch_model_with_early_exit(hf_model, ee_head)
    
    # Autoregressive Sampler
    autoreg_sampler = AutoregressiveSampler(hf_model, tokenizer, device)

    results = []
    
    start = 100*args.batch_idx
    end = 100*(args.batch_idx+1)
    end = min(end, len(dataset))

    for i, data in tqdm(enumerate(dataset[start:end]), total=end-start, desc="EE-PS Benchmark"):
        question = data["prompt"]
        answer = data["answer"]

        input_text = format_prompt(question, "qwen_math", tokenizer, cot)
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        prefx = [idx.item() for idx in input_ids[0]]

        mcmc_out, _, _, accept_ratio = mcmc_power_samp_ee(
            autoreg_sampler, prefx, temp, mcmc_steps, max_new_tokens=1024, block_num=16, debug=True
        )

        mcmc_completion = tokenizer.decode(torch.tensor(mcmc_out[len(prefx):], dtype=torch.long).cpu(), skip_special_tokens=True)
        mcmc_answer_parsed = parse_answer(mcmc_completion)
        
        print(f"Q: {question[:50]}...")
        print(f"A: {mcmc_answer_parsed}")
        print(f"Ratio: {accept_ratio}")

        results.append({
            "question": question,
            "correct_answer": answer,
            "mcmc_completion": mcmc_completion,
            "mcmc_answer": mcmc_answer_parsed,
            "acceptance_ratio": accept_ratio
        })

    out_file = os.path.join(save_str, f"ee_ps_results_steps{mcmc_steps}_temp{temp}_batch{args.batch_idx}.csv")
    df = pd.DataFrame(results)
    df.to_csv(out_file, index=False)
    print(f"Saved results to {out_file}")
