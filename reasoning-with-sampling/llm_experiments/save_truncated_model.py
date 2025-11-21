import os
import torch
import transformers
import argparse

def save_truncated_model(model_str, layer_idx, output_path, device="cuda"):
    """
    Loads a model, physically removes layers after layer_idx,
    and saves the resulting model structure.
    """
    print(f"Loading model: {model_str}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_str, trust_remote_code=True)
    # Load on CPU first to avoid OOM if we are just saving
    model = transformers.AutoModelForCausalLM.from_pretrained(model_str, torch_dtype="auto", device_map="auto", trust_remote_code=True)

    print(f"Original layers: {model.config.num_hidden_layers}")
    
    if layer_idx < model.config.num_hidden_layers:
        print(f"Truncating model to first {layer_idx + 1} layers...")
        
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            model.model.layers = torch.nn.ModuleList(model.model.layers[:layer_idx + 1])
        elif hasattr(model, "layers"):
             model.layers = torch.nn.ModuleList(model.layers[:layer_idx + 1])
        else:
            print("Warning: Could not find layer list to truncate.")
            return

        # Update config
        model.config.num_hidden_layers = len(model.model.layers)
        print(f"New Total Layers: {model.config.num_hidden_layers}")
    else:
        print("Layer index >= total layers. No truncation performed.")

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Saving truncated model to {output_path}")
    
    # Save as a pretrained model (safetensors/config) so it can be loaded with .from_pretrained
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"Done! You can load this model using 'AutoModelForCausalLM.from_pretrained(\"{output_path}\")'")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save Truncated Model")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B", help="Model identifier")
    parser.add_argument("--layer_idx", type=int, default=18, help="Last layer index to keep (0-indexed)")
    parser.add_argument("--output", type=str, default="models/qwen_truncated_layer18", help="Directory to save model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    save_truncated_model(
        args.model, 
        args.layer_idx, 
        args.output, 
        args.device
    )

