import torch
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
from power_samp_utils import naive_temp

def compute_full_log_probs(model, input_ids, temp):
    """
    Computes the log probability of the sequence under the full model (power distribution).
    Returns sum of log probs (scalar) and the per-token log probs (list).
    """
    device = input_ids.device
    
    with torch.no_grad():
        # Use original_forward if patched
        if hasattr(model, 'original_forward'):
             outputs = model.original_forward(input_ids)
        else:
             outputs = model(input_ids)
             
        logits = outputs.logits # [batch, seq, vocab]
        
        # Shift logits and labels
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        gathered_log_probs = torch.gather(log_probs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
        
        scaled_log_probs = (1.0 / temp) * gathered_log_probs
        
        log_probs_list = scaled_log_probs[0].tolist()
        
        return sum(log_probs_list), log_probs_list

def mcmc_power_samp_ee(p_sampler, context, temp, mcmc_steps, max_new_tokens, block_num=16, debug=False):
    """
    Early-Exit Power Sampling with Delayed Acceptance.
    p_sampler: AutoregressiveSampler with patched Early Exit model.
    """
    c = len(context)
    print(f'alpha: {1/temp} (EE-PS)')
    gen = []
    if context is not None:
        gen = context.copy()
        
    log_probs_norm_cheap = [] 
    log_probs_unnorm_cheap = [] 
    log_probs_unnorm_full = [] 
    
    total_proposals = 0
    cheap_accepts = 0
    full_accepts = 0
    
    assert max_new_tokens % block_num == 0
    jump_size = int(max_new_tokens // block_num)
    
    # The model inside sampler is patched
    model = p_sampler.model
    
    for _ in tqdm(range(block_num)):
        # 1. Grow the chain (Block generation using CHEAP model via patched forward)
        gen, lp_norm_cheap, lp_unnorm_cheap = naive_temp(p_sampler, gen, temp=temp, seq_len=jump_size+len(gen))
        
        log_probs_norm_cheap.extend(lp_norm_cheap)
        log_probs_unnorm_cheap.extend(lp_unnorm_cheap)
        
        # Compute full log probs for the new block
        input_ids = torch.tensor([gen], dtype=torch.long, device=p_sampler.device)
        _, full_log_probs_all = compute_full_log_probs(model, input_ids, temp)
        
        current_full_log_probs = full_log_probs_all[c-1:]
        new_full_probs = current_full_log_probs[-jump_size:]
        log_probs_unnorm_full.extend(new_full_probs)
        
        assert len(log_probs_unnorm_cheap) == len(log_probs_unnorm_full) == len(gen) - c

        for _ in range(mcmc_steps):
            total_proposals += 1
            t = len(gen)
            idx = random.randint(c, t-1)
            
            # Propose using CHEAP model (patched)
            prop, log_prob_prop_cheap, target_log_prob_prop_cheap = naive_temp(p_sampler, gen[:idx], temp=temp, seq_len=t)
            
            s = len(prop)
            log_prob_cur_cheap = log_probs_norm_cheap[idx-c:]
            target_log_prob_cur_cheap = log_probs_unnorm_cheap[idx-c:]
            
            log_r_cheap = sum(target_log_prob_prop_cheap) + sum(log_prob_cur_cheap) - sum(target_log_prob_cur_cheap) - sum(log_prob_prop_cheap)
            
            u1 = random.random()
            acc_cheap = min(1.0, np.exp(log_r_cheap))
            
            if u1 < acc_cheap:
                cheap_accepts += 1
                
                # Full Correction
                prop_ids = torch.tensor([prop], dtype=torch.long, device=p_sampler.device)
                _, full_log_probs_all_prop = compute_full_log_probs(model, prop_ids, temp)
                target_log_prob_prop_full = full_log_probs_all_prop[idx-1:] 
                
                target_log_prob_cur_full = log_probs_unnorm_full[idx-c:]
                
                log_r_full = sum(target_log_prob_prop_full) + sum(log_prob_cur_cheap) - sum(target_log_prob_cur_full) - sum(log_prob_prop_cheap)
                
                if log_r_cheap > 0:
                    correction_log_prob = log_r_full
                else:
                    correction_log_prob = log_r_full - log_r_cheap
                
                u2 = random.random()
                if u2 < min(1.0, np.exp(correction_log_prob)):
                    full_accepts += 1
                    gen = prop.copy()
                    
                    log_probs_norm_cheap[idx-c:] = log_prob_prop_cheap.copy()
                    log_probs_unnorm_cheap[idx-c:] = target_log_prob_prop_cheap.copy()
                    log_probs_unnorm_full[idx-c:] = target_log_prob_prop_full
                    
                    del prop
            
        if p_sampler.tokenizer.eos_token_id in gen:
             eos_idx = gen.index(p_sampler.tokenizer.eos_token_id)
             gen = gen[:eos_idx + 1]
             new_len = eos_idx + 1 - c
             if new_len < 0: new_len = 0
             
             log_probs_norm_cheap = log_probs_norm_cheap[:new_len]
             log_probs_unnorm_cheap = log_probs_unnorm_cheap[:new_len]
             log_probs_unnorm_full = log_probs_unnorm_full[:new_len]
             
             break

    acceptance_ratio = full_accepts / total_proposals if total_proposals > 0 else 0
    
    if debug:
        print(f"EE Stats: Cheap Accepts: {cheap_accepts}/{total_proposals}, Full Accepts: {full_accepts}/{cheap_accepts if cheap_accepts>0 else 1}")

    return gen, log_probs_norm_cheap, log_probs_unnorm_cheap, acceptance_ratio
