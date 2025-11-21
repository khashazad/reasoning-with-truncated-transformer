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
        
        # Calibration parameters (diagonal affine map)
        self.mu_L = None
        self.sigma_L = None
        self.mu_final = None
        self.sigma_final = None
        
        # Temperature scaling params
        self.temp = 1.0
        self.bias = 0.0
        
        self.is_calibrated = False

    def set_calibration_params(self, mu_L, sigma_L, mu_final, sigma_final):
        self.mu_L = mu_L.to(self.device)
        self.sigma_L = sigma_L.to(self.device)
        self.mu_final = mu_final.to(self.device)
        self.sigma_final = sigma_final.to(self.device)
        self.is_calibrated = True

    def get_mid_logits(self, hidden_states_L):
        """
        Computes logits from mid-layer hidden states using the calibration map and the model's head.
        hidden_states_L: [batch_size, seq_len, hidden_dim]
        """
        if not self.is_calibrated:
            h_mapped = hidden_states_L
        else:
            # Apply diagonal affine map
            h_mapped = self.sigma_final * (hidden_states_L - self.mu_L) / (self.sigma_L + 1e-6) + self.mu_final
            
        # Apply final norm and head
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'norm'):
             norm_layer = self.model.model.norm
        elif hasattr(self.model, 'norm'):
             norm_layer = self.model.norm
        else:
             raise ValueError("Could not find final normalization layer in model")

        h_normed = norm_layer(h_mapped)
        logits = self.model.lm_head(h_normed)
        
        return logits

def calibrate_mid_layer(model, dataloader, layer_idx, device, num_batches=10):
    """
    Computes Mean and Std for the mid-layer and the final layer (pre-norm).
    """
    model.eval()
    mid_states = []
    final_states = []
    
    print(f"Calibrating layer {layer_idx}...")
    
    # Ensure we use the ORIGINAL forward for calibration!
    # If model is already patched, we must access original_forward if possible, 
    # OR assume this is called BEFORE patching.
    
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
            
    mid_states = torch.cat(mid_states, dim=0)
    final_states = torch.cat(final_states, dim=0)
    
    print(f"Collected {mid_states.shape[0]} tokens for calibration.")
    
    mu_L = mid_states.mean(dim=0)
    sigma_L = mid_states.std(dim=0)
    
    mu_final = final_states.mean(dim=0)
    sigma_final = final_states.std(dim=0)
    
    return mu_L, sigma_L, mu_final, sigma_final

def early_exit_forward(self, input_ids=None, attention_mask=None, **kwargs):
    """
    Monkey-patchable forward method.
    'self' will be bound to the model instance.
    """
    # Force output_hidden_states
    kwargs['output_hidden_states'] = True
    
    # Call original forward
    # We assume self.original_forward exists
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
