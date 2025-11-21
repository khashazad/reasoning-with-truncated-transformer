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
        
        # No calibration parameters anymore.
        # We will just pass mid-layer activations directly to the final norm/head.
        
    def set_calibration_params(self, *args):
        # Deprecated/No-op
        pass

    def get_mid_logits(self, hidden_states_L):
        """
        Computes logits from mid-layer hidden states directly using the model's final head.
        hidden_states_L: [batch_size, seq_len, hidden_dim]
        """
        
        # No projection/calibration.
        # Just take h_L and treat it as h_final.
        h_mapped = hidden_states_L
            
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

# Removed calibrate_mid_layer function as it is no longer needed.

def early_exit_forward(self, input_ids=None, attention_mask=None, **kwargs):
    """
    Monkey-patchable forward method.
    """
    # Force output_hidden_states
    kwargs['output_hidden_states'] = True
    
    # Call original forward
    outputs = self.original_forward(input_ids, attention_mask=attention_mask, **kwargs)
    
    # Get mid logits
    # layer_idx + 1 because index 0 is embeddings
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
    if hasattr(model, 'original_forward'):
        print("Model already patched.")
        return
    
    model.ee_head = ee_head
    model.original_forward = model.forward
    model.forward = MethodType(early_exit_forward, model)
    print("Model patched with Early Exit forward (Direct Connection).")

def unpatch_model(model):
    if hasattr(model, 'original_forward'):
        model.forward = model.original_forward
        del model.original_forward
        del model.ee_head
        print("Model unpatched.")
