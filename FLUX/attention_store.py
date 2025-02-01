from typing import List
import numpy as np
import torch

class AttentionStore:
    def __init__(self, save_timesteps=None):
        if save_timesteps is None:
            self.save_timesteps = list(range(50))
        else:
            self.save_timesteps = save_timesteps

        self.step_store = {}
        self.step_store_count = {}
        self.curr_iter = 0

    def store_attention(self, attention_probs, step_index: int, place_in_unet: str, batch_size, num_heads):
        text_len = attention_probs.size(1) - 4096

        # Split batch and heads
        attention_probs = attention_probs.view(batch_size, num_heads, *attention_probs.shape[1:])

        # Mean over the heads
        attention_probs = attention_probs.mean(dim=1)

        # Attention: image -> text
        attention_probs_image2text = attention_probs[:, text_len:, :text_len].transpose(1,2)

        if step_index in self.save_timesteps:
            if step_index not in self.step_store:
                self.step_store[step_index] = torch.zeros_like(attention_probs_image2text)
                self.step_store_count[step_index] = 0
            
            self.step_store[step_index] += attention_probs_image2text
            self.step_store_count[step_index] += 1

    def aggregate_attention(self, step_indices = None):
        if step_indices is None:
            step_indices = list(self.step_store.keys())

        attns = []
        for step_index in step_indices:
            attns.append(self.step_store[step_index] / self.step_store_count[step_index])
        
        attns = torch.stack(attns, dim=0).mean(dim=0)
        
        H = W = int(np.sqrt(attns.shape[2]))
        attns = attns.view(attns.shape[0], attns.shape[1], H, W)

        return attns