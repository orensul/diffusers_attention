from typing import List
import numpy as np
import torch

class AttentionStore:
    def __init__(self, save_attn_res=32, save_timesteps=None):
        if save_timesteps is None:
            self.save_timesteps = list(range(50))
        else:
            self.save_timesteps = save_timesteps

        self.save_attn_res = (save_attn_res, save_attn_res)
        self.cross_step_store = {}

    def store_attention(self, attn, step_index: int, place_in_unet: str, attn_heads):
        if (attn.shape[1] == np.prod(self.save_attn_res) and step_index in self.save_timesteps):
            guided_attn = attn[attn.size(0)//2:]
            guided_attn = guided_attn.reshape([guided_attn.shape[0]//attn_heads, attn_heads, *guided_attn.shape[1:]]).mean(dim=1)

            if step_index not in self.cross_step_store:
                self.cross_step_store[step_index] = {}
            
            if place_in_unet not in self.cross_step_store[step_index]:
                self.cross_step_store[step_index][place_in_unet] = {}

            self.cross_step_store[step_index][place_in_unet] = guided_attn

    def aggregate_attention(self, step_indices = None, layers = None):
        if step_indices is None:
            step_indices = list(self.cross_step_store.keys())
        if layers is None:
            layers = list(self.cross_step_store[step_indices[0]].keys())

        attns = []
        for step_index in step_indices:
            for layer in layers:
                attns.append(self.cross_step_store[step_index][layer])
        
        attns = torch.stack(attns, dim=0).mean(dim=0)
        attns = attns.view(attns.shape[0], *self.save_attn_res, attns.shape[2])

        return attns