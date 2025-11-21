import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os
from transformers.modeling_outputs import CausalLMOutputWithPast
from types import MethodType

class EarlyExitHead:
    def __init__(self, model, layer_idx, device):
        self.model = model
        self.layer_idx = layer_idx
        self.device = device
        
        # Linear Calibration parameters: h_final = h_mid @ weight.T + bias
        self.weight = None
        self.bias = None
        
        self.is_calibrated = False

    def set_calibration_params(self, weight, bias):
        self.weight = weight.to(self.device)
        self.bias = bias.to(self.device)
        self.is_calibrated = True

    def get_mid_logits(self, hidden_states_L):
        """
        Computes logits from mid-layer hidden states using the calibration map and the model's head.
        hidden_states_L: [batch_size, seq_len, hidden_dim]
        """
        if not self.is_calibrated:
            h_mapped = hidden_states_L
        else:
            # Apply linear map: h @ W.T + b
            h_mapped = F.linear(hidden_states_L, self.weight, self.bias)
            
        # Apply final norm and head
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'norm'):
             norm_layer = self.model.model.norm
        elif hasattr(self.model, 'norm'):
             norm_layer = self.model.norm
        else:
             # Fallback 
             raise ValueError("Could not find final normalization layer in model")

        h_normed = norm_layer(h_mapped)
        logits = self.model.lm_head(h_normed)
        
        return logits

def calibrate_mid_layer(model, dataloader, layer_idx, device, num_batches=10):
    """
    Computes Linear Regression parameters W, b such that h_L @ W.T + b approx h_final.
    """
    model.eval()
    mid_states = []
    final_states = []
    
    print(f"Calibrating layer {layer_idx} using Linear Regression...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, total=num_batches)):
            if i >= num_batches:
                break
            
            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(device)
            else:
                input_ids = batch.to(device)

            outputs = model(input_ids, output_hidden_states=True)
            
            h_L = outputs.hidden_states[layer_idx + 1]
            h_final = outputs.hidden_states[-1]
            
            mid_states.append(h_L.reshape(-1, h_L.shape[-1]).cpu())
            final_states.append(h_final.reshape(-1, h_final.shape[-1]).cpu())
            
    X = torch.cat(mid_states, dim=0).float() # [N, d_in]
    Y = torch.cat(final_states, dim=0).float() # [N, d_out]
    
    print(f"Collected {X.shape[0]} tokens. Solving Least Squares...")
    
    # We want to find W, b such that X @ W.T + b = Y
    # Add bias term to X
    X_bias = torch.cat([X, torch.ones(X.shape[0], 1)], dim=1) # [N, d_in + 1]
    
    # Solve: X_bias @ Theta = Y, where Theta is [d_in+1, d_out]
    # Theta = (X.T X)^-1 X.T Y
    # Using torch.linalg.lstsq
    
    # Depending on size, lstsq might be slow. 
    # N ~ 10 batches * 500 tokens ~ 5000. 
    # dim ~ 1536.
    # X is 5000x1537.
    # This is small enough for direct solve.
    
    solution = torch.linalg.lstsq(X_bias, Y).solution # [d_in+1, d_out]
    
    # Extract W and b
    # Theta = [W.T; b]
    # W.T is [d_in, d_out]
    # b is [1, d_out]
    
    W_T = solution[:-1, :]
    b = solution[-1, :]
    
    weight = W_T.t() # [d_out, d_in]
    bias = b # [d_out]
    
    print("Calibration complete.")
    
    return weight, bias

def early_exit_forward(self, input_ids=None, attention_mask=None, **kwargs):
    """
    Monkey-patchable forward method.
    """
    # Force output_hidden_states
    kwargs['output_hidden_states'] = True
    
    # Call original forward
    outputs = self.original_forward(input_ids, attention_mask=attention_mask, **kwargs)
    
    # Get mid logits
    h_L = outputs.hidden_states[self.ee_head.layer_idx + 1]
    mid_logits = self.ee_head.get_mid_logits(h_L)
    
    # Return CausalLMOutputWithPast with MID LOGITS
    return CausalLMOutputWithPast(
        loss=outputs.loss,
        logits=mid_logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def patch_model_with_early_exit(model, ee_head):
    """
    Patches the model to use early exit logits in forward().
    Stores original forward in model.original_forward.
    """
    if hasattr(model, 'original_forward'):
        print("Model already patched.")
        return
    
    model.ee_head = ee_head
    model.original_forward = model.forward
    # Bind the new method
    model.forward = MethodType(early_exit_forward, model)
    print("Model patched with Early Exit forward.")

def unpatch_model(model):
    if hasattr(model, 'original_forward'):
        model.forward = model.original_forward
        del model.original_forward
        del model.ee_head
        print("Model unpatched.")
