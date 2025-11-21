import torch
import transformers
import random
import numpy as np
from tqdm import tqdm
import argparse
import os
import sys

# Ensure we can import from the local files
sys.path.append(os.getcwd())

from power_samp_utils import AutoregressiveSampler, format_prompt, mcmc_power_samp

def run_interactive_test(model_name="Qwen/Qwen2.5-Math-1.5B", layers=[18], alphas=[2.0], question="What is 2+2?"):
    """
    Runs a local interactive test to visualize the Truncated-PS process.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and torch.backends.mps.is_available():
        device = "mps"
    
    print(f"=== Starting Local Diagnostic Test (Truncated Models) ===")
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Question: {question}")
    print("="*40)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Base Prompt
    input_text = format_prompt(question, "qwen_math", tokenizer, cot=True)
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    prefix = [idx.item() for idx in input_ids[0]]
    
    print(f"Prompt Tokens: {len(prefix)}")
    print("-" * 40)

    results = []

    for layer_idx in layers:
        print(f"\n>>> Testing Truncated Model at Layer: {layer_idx}")
        
        # Reload model fresh each time to perform clean truncation
        print(f"Reloading and truncating model to {layer_idx+1} layers...")
        # Avoid device_map="auto" when we want to manually move to device afterwards
        # to avoid "RuntimeError: You can't move a model that has some modules offloaded"
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True).to(device)
        
        # Truncate
        if layer_idx < model.config.num_hidden_layers:
             if hasattr(model, "model") and hasattr(model.model, "layers"):
                 model.model.layers = torch.nn.ModuleList(model.model.layers[:layer_idx + 1])
             elif hasattr(model, "layers"):
                  model.layers = torch.nn.ModuleList(model.layers[:layer_idx + 1])
             model.config.num_hidden_layers = layer_idx + 1
        
        autoreg_sampler = AutoregressiveSampler(model, tokenizer, device)
        
        for alpha in alphas:
            temp = 1.0 / alpha
            print(f"\n   [Alpha = {alpha} (Temp={temp:.2f})]")
            
            gen = prefix.copy()
            print("   Starting Generation...")
            
            # Run MCMC Generation
            mcmc_out, _, _, ratio = mcmc_power_samp(
                autoreg_sampler, 
                gen, 
                temp, 
                mcmc_steps=5, 
                max_new_tokens=20, 
                block_num=2
            )
            
            output_text = tokenizer.decode(torch.tensor(mcmc_out[len(prefix):]).cpu(), skip_special_tokens=True)
            print(f"   > Generated: {repr(output_text)}")
            print(f"   > Acceptance Ratio: {ratio:.2%}")
            
            results.append({
                "layer": layer_idx,
                "alpha": alpha,
                "ratio": ratio,
                "output": output_text
            })

    print("\n" + "="*40)
    print("SUMMARY COMPARISON")
    print(f"{'Layer':<8} | {'Alpha':<8} | {'Acc Ratio':<10} | {'Output Start'}")
    print("-" * 60)
    for r in results:
        print(f"{r['layer']:<8} | {r['alpha']:<8} | {r['ratio']:<10.1%} | {r['output'][:30]}...")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--layers", type=int, nargs="+", default=[ 18, 20, 21 , 22, 24, 27])
    parser.add_argument("--alphas", type=float, nargs="+", default=[2.0, 4.0])
    parser.add_argument("--question", type=str, default="Calculate 15 * 7 + 3.")
    
    args = parser.parse_args()
    
    run_interactive_test(args.model, args.layers, args.alphas, args.question)
