import torch
import transformers
import random
import numpy as np
from tqdm import tqdm
import argparse
import os

# Import necessary components
# Ensure we can import from the local files
import sys
sys.path.append(os.getcwd())

from ee_utils import EarlyExitHead, patch_model_with_early_exit
from power_samp_utils import AutoregressiveSampler, format_prompt
from power_samp_ee import mcmc_power_samp_ee

def run_interactive_test(model_name="Qwen/Qwen2.5-Math-1.5B", layers=[18], alphas=[2.0], question="What is 2+2?"):
    """
    Runs a local interactive test to visualize the EE-PS process step-by-step.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and torch.backends.mps.is_available():
        device = "mps"
    
    print(f"=== Starting Local Diagnostic Test ===")
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Question: {question}")
    print("="*40)

    print("Loading Model (once)...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True).to(device)
    
    # Base Prompt
    input_text = format_prompt(question, "qwen_math", tokenizer, cot=True)
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    prefix = [idx.item() for idx in input_ids[0]]
    
    print(f"Prompt Tokens: {len(prefix)}")
    print("-" * 40)

    results = []

    for layer_idx in layers:
        print(f"\n>>> Testing Layer: {layer_idx}")
        
        # 1. Patch Model
        # We need to create a new head and patch
        # Note: If model is already patched, we should unpatch first or just overwrite
        if hasattr(model, 'original_forward'):
            # Simple unpatch logic or just re-patch
            model.forward = model.original_forward
            del model.original_forward
            del model.ee_head
            
        ee_head = EarlyExitHead(model, layer_idx, device)
        patch_model_with_early_exit(model, ee_head)
        
        # 2. Sampler Wrapper
        # We pass the patched model to the sampler
        # Note: AutoregressiveSampler expects a 'model' that has a .generate method
        # HuggingFace models have .generate. Our patching only affects .forward()
        # naive_temp calls .generate(). .generate() calls .forward().
        # So this works!
        autoreg_sampler = AutoregressiveSampler(model, tokenizer, device)
        
        for alpha in alphas:
            temp = 1.0 / alpha
            print(f"\n   [Alpha = {alpha} (Temp={temp:.2f})]")
            
            # Run MCMC Generation (Short run)
            # We turn on 'debug=True' in mcmc_power_samp_ee to see stats
            # We'll implement a custom verbose version here to see *step-by-step*
            
            # Re-implementing a mini-loop here for visualization
            gen = prefix.copy()
            
            print("   Starting Generation (showing first 2 steps details)...")
            
            # Let's run the actual function but capture output
            # We limit to small number of tokens for test
            mcmc_out, _, _, ratio = mcmc_power_samp_ee(
                autoreg_sampler, 
                gen, 
                temp, 
                mcmc_steps=5, # Small steps for test
                max_new_tokens=20, # Generate ~20 tokens
                block_num=2, # 2 blocks of 10 tokens
                debug=True
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

