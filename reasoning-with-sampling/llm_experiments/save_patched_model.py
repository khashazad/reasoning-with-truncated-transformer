import os
import torch
import transformers
import argparse
from ee_utils import EarlyExitHead, patch_model_with_early_exit

def save_patched_model(model_str, layer_idx, output_path, device="cuda"):
    """
    Loads a model, patches it with a direct early-exit connection at `layer_idx`,
    and saves the resulting model structure.
    
    Note: Since we are just monkey-patching the forward pass and not changing weights,
    saving the 'model' typically just saves the weights (which are unchanged).
    
    If you want to save a 'model object' that automatically has this behavior when loaded,
    PyTorch serialization (torch.save(model)) handles the object structure, including
    the monkey-patched method, *if* the environment is consistent.
    
    However, a safer and more portable way for HuggingFace models is usually to save
    the configuration or a wrapper. But since the user requested saving 'the model done in this way':
    
    We will save the entire patched model object using torch.save.
    """
    print(f"Loading model: {model_str}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_str, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_str, torch_dtype="auto", device_map="auto", trust_remote_code=True).to(device)

    print(f"Patching model at layer {layer_idx}...")
    ee_head = EarlyExitHead(model, layer_idx, device)
    patch_model_with_early_exit(model, ee_head)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Saving patched model to {output_path}")
    # Saving the entire model object (pickled)
    # This includes the patched 'forward' method bound to the instance.
    torch.save(model, output_path)
    
    print("Done! You can load this model using 'model = torch.load(path)'")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save Patched Early-Exit Model")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B", help="Model identifier")
    parser.add_argument("--layer_idx", type=int, default=18, help="Layer to attach early exit")
    parser.add_argument("--output", type=str, default="patched_models/qwen_ee_layer18.pt", help="Path to save .pt file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    save_patched_model(
        args.model, 
        args.layer_idx, 
        args.output, 
        args.device
    )

