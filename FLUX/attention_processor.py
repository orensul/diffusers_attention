from diffusers.models.attention_processor import Attention, apply_rope
import torch.nn.functional as F
import torch
from typing import Callable, Optional
import inspect
import warnings
from typing import Any, Dict, Union
from packaging import version
import torch
from typing import Optional

import torch
import torch.nn.functional as F



# def apply_standard_attention(query, key, value, attn):
#     batch_size, attn_heads, _, head_dim = query.shape
#
#     query = query.reshape(batch_size*attn_heads, -1, head_dim)
#     key = key.reshape(batch_size*attn_heads, -1, head_dim)
#     value = value.reshape(batch_size*attn_heads, -1, head_dim)
#
#     attention_probs = attn.get_attention_scores(query, key)
#
#     hidden_states = torch.bmm(attention_probs, value)
#     hidden_states = hidden_states.view(batch_size, attn_heads, -1, head_dim)
#
#     return hidden_states, attention_probs


# def extended_self_attention(query, key, value):
#     batch_size, num_heads, seq_len, d_k = key.shape
#     # Concatenate keys and values across the batch dimension efficiently
#     extended_key = torch.cat([key[i] for i in range(batch_size)], dim=1).unsqueeze(0).expand(batch_size, -1, -1, -1)
#     extended_value = torch.cat([value[i] for i in range(batch_size)], dim=1).unsqueeze(0).expand(batch_size, -1, -1, -1)
#     # Compute extended self-attention
#     hidden_states = F.scaled_dot_product_attention(
#         query,
#         extended_key,
#         extended_value,
#         dropout_p=0.0,
#         is_causal=False
#     )
#     return hidden_states



# def extended_attention_batch_n(query, key, value, curr_step, t_range, text_token_count=77):
#     """
#     Applies extended attention only if the current step is within the specified t_range.
#     Shares only image tokens across the batch.
#
#     Args:
#         query (torch.Tensor): Query tensor of shape [batch, heads, tokens, dim].
#         key (torch.Tensor): Key tensor of shape [batch, heads, tokens, dim].
#         value (torch.Tensor): Value tensor of shape [batch, heads, tokens, dim].
#         curr_step (int): The current step in the diffusion process.
#         t_range (list of tuples): List of (start, end) step ranges where extended attention is applied.
#         text_token_count (int): Number of text tokens to separate from image tokens.
#
#     Returns:
#         torch.Tensor: The hidden states after applying attention.
#     """
#     batch_size, heads, tokens, dim = query.shape
#     apply_extended = any(start <= curr_step <= end for start, end in t_range)
#
#     # First image attends normally
#     hidden_states_list = [F.scaled_dot_product_attention(query[:1], key[:1], value[:1], dropout_p=0.0, is_causal=False)]
#
#     if apply_extended:
#         # Extract image tokens from the first image (excluding text tokens)
#         added_key = key[0, :, text_token_count:]  # Shape: [heads, img_tokens, dim]
#         added_value = value[0, :, text_token_count:]  # Shape: [heads, img_tokens, dim]
#
#         for i in range(1, batch_size):
#             # Extend key and value for each image by adding first image's image tokens
#             extended_key = torch.cat([added_key, key[i]], dim=1).unsqueeze(0)  # Shape: [1, heads, new_tokens, dim]
#             extended_value = torch.cat([added_value, value[i]], dim=1).unsqueeze(0)  # Shape: [1, heads, new_tokens, dim]
#
#             # Compute extended attention for each image in the batch
#             hidden_states_i = F.scaled_dot_product_attention(query[i:i+1], extended_key, extended_value, dropout_p=0.0, is_causal=False)
#             hidden_states_list.append(hidden_states_i)
#     else:
#         # Each image attends normally without sharing tokens
#         for i in range(1, batch_size):
#             hidden_states_i = F.scaled_dot_product_attention(query[i:i+1], key[i:i+1], value[i:i+1], dropout_p=0.0, is_causal=False)
#             hidden_states_list.append(hidden_states_i)
#
#     # Concatenate outputs along the batch dimension
#     hidden_states = torch.cat(hidden_states_list, dim=0)  # Shape: [batch_size, heads, tokens, dim]
#
#     return hidden_states



# def extended_subject_attention(query, key, value, curr_step, t_range, text_token_count=77):
#     batch_size, heads, tokens, dim = query.shape
#     apply_extended = any(start <= curr_step <= end for start, end in t_range)
#
#     # Standard self-attention
#     hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
#
#     if apply_extended:
#         image_key = key[:, :, text_token_count:]  # Shape: [batch, heads, img_tokens, dim]
#         image_value = value[:, :, text_token_count:]  # Shape: [batch, heads, img_tokens, dim]
#
#         # First two images (anchors) attend to themselves only
#         anchor_keys = image_key[:2]
#         anchor_values = image_value[:2]
#
#         # Non-anchor images should attend to themselves + the two anchors
#         non_anchor_keys = image_key[2:]
#         non_anchor_values = image_value[2:]
#
#         # Reshape anchors for broadcasting
#         anchor_keys_expanded = anchor_keys.permute(1, 0, 2, 3).reshape(heads, -1, dim)
#         anchor_values_expanded = anchor_values.permute(1, 0, 2, 3).reshape(heads, -1, dim)
#
#         # Expand for non-anchor images only
#         anchor_keys_expanded = anchor_keys_expanded.unsqueeze(0).expand(batch_size - 2, -1, -1, -1)
#         anchor_values_expanded = anchor_values_expanded.unsqueeze(0).expand(batch_size - 2, -1, -1, -1)
#
#         # Concatenate extended context only for non-anchor images
#         extended_key = torch.cat([key[2:, :, :text_token_count], non_anchor_keys, anchor_keys_expanded], dim=2)
#         extended_value = torch.cat([value[2:, :, :text_token_count], non_anchor_values, anchor_values_expanded], dim=2)
#
#         # Compute extended attention only for non-anchor images
#         hidden_states[2:] = F.scaled_dot_product_attention(query[2:], extended_key, extended_value, dropout_p=0.0, is_causal=False)
#
#     return hidden_states


def anchored_attention_batch(query, key, value, curr_step, t_range, text_token_count=77, dropout_value=0.0):
    """
    Implements anchored subject-driven self-attention where every image attends to:
    - Itself
    - The first two images in the batch (acting as anchors)

    Args:
        query (torch.Tensor): Query tensor of shape [batch, heads, tokens, dim].
        key (torch.Tensor): Key tensor of shape [batch, heads, tokens, dim].
        value (torch.Tensor): Value tensor of shape [batch, heads, tokens, dim].
        curr_step (int): The current step in the diffusion process.
        t_range (list of tuples): List of (start, end) step ranges where extended attention is applied.
        text_token_count (int): Number of text tokens before the image tokens start.

    Returns:
        torch.Tensor: The hidden states after applying self-attention.
    """
    batch_size, heads, tokens, dim = query.shape
    apply_extended = any(start <= curr_step <= end for start, end in t_range)

    # Compute standard self-attention for all images (default behavior)
    hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

    if apply_extended:
        N_ANCHORS = 2
        N_EXTRA = batch_size - N_ANCHORS

        # Extract only the image tokens (excluding text tokens) from all images
        image_key = key[:, :, text_token_count:]  # Shape: [batch, heads, img_tokens, dim]
        image_value = value[:, :, text_token_count:]  # Shape: [batch, heads, img_tokens, dim]

        # Get image tokens from the first two images (anchors)
        anchor_keys = image_key[:N_ANCHORS]  # Shape: [2, heads, img_tokens, dim]
        anchor_values = image_value[:N_ANCHORS]  # Shape: [2, heads, img_tokens, dim]

        # Reshape anchors to match batch size for easy broadcasting
        anchor_keys = anchor_keys.permute(1, 0, 2, 3).reshape(heads, -1, dim).unsqueeze(0)  # Shape: [1, heads, 2 * img_tokens, dim]
        anchor_values = anchor_values.permute(1, 0, 2, 3).reshape(heads, -1, dim).unsqueeze(0)  # Shape: [1, heads, 2 * img_tokens, dim]

        # Concatenate each image's own tokens with the anchor tokens
        extended_key_anchor = torch.cat([key[:N_ANCHORS, :, :text_token_count], anchor_keys.expand(N_ANCHORS, -1, -1, -1)], dim=2)
        extended_value_anchor = torch.cat([value[:N_ANCHORS, :, :text_token_count], anchor_values.expand(N_ANCHORS, -1, -1, -1)], dim=2)
        extended_key_extra = torch.cat([key[N_ANCHORS:], anchor_keys.expand(N_EXTRA, -1, -1, -1)], dim=2)
        extended_value_extra = torch.cat([value[N_ANCHORS:], anchor_values.expand(N_EXTRA, -1, -1, -1)], dim=2)

        # Apply extended self-attention
        hidden_states_anchors = F.scaled_dot_product_attention(query[:N_ANCHORS], extended_key_anchor, extended_value_anchor, dropout_p=dropout_value, is_causal=False)
        hidden_states_extra = F.scaled_dot_product_attention(query[N_ANCHORS:], extended_key_extra, extended_value_extra, dropout_p=dropout_value, is_causal=False)
        hidden_states = torch.cat([hidden_states_anchors, hidden_states_extra], dim=0)

    return hidden_states


# def extended_attention_batch_n(query, key, value, curr_step, t_range, text_token_count=77):
#     """
#     Implements full subject-driven self-attention where image tokens are shared across all images in the batch.
#
#     Args:
#         query (torch.Tensor): Query tensor of shape [batch, heads, tokens, dim].
#         key (torch.Tensor): Key tensor of shape [batch, heads, tokens, dim].
#         value (torch.Tensor): Value tensor of shape [batch, heads, tokens, dim].
#         curr_step (int): The current step in the diffusion process.
#         t_range (list of tuples): List of (start, end) step ranges where extended attention is applied.
#         text_token_count (int): Number of text tokens before the image tokens start.
#
#     Returns:
#         torch.Tensor: The hidden states after applying self-attention.
#     """
#     batch_size, heads, tokens, dim = query.shape
#     apply_extended = any(start <= curr_step <= end for start, end in t_range)
#
#     # Compute standard self-attention for all images
#     hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
#
#     if apply_extended:
#         # Extract only the image tokens (excluding text tokens) from all images
#         image_key = key[:, :, text_token_count:]  # Shape: [batch, heads, img_tokens, dim]
#         image_value = value[:, :, text_token_count:]  # Shape: [batch, heads, img_tokens, dim]
#
#         # **Fix: Reshape and correctly align batch dimensions**
#         global_key = image_key.permute(1, 0, 2, 3).reshape(heads, -1, dim)  # Shape: [heads, batch * img_tokens, dim]
#         global_value = image_value.permute(1, 0, 2, 3).reshape(heads, -1, dim)  # Shape: [heads, batch * img_tokens, dim]
#
#         # Expand to match batch size correctly
#         global_key = global_key.unsqueeze(0).expand(batch_size, -1, -1, -1)  # Shape: [batch, heads, batch * img_tokens, dim]
#         global_value = global_value.unsqueeze(0).expand(batch_size, -1, -1, -1)  # Shape: [batch, heads, batch * img_tokens, dim]
#
#         # Concatenate text tokens back with the global image token pool
#         extended_key = torch.cat([key[:, :, :text_token_count], global_key], dim=2)  # Shape: [batch, heads, text_token_count + batch * img_tokens, dim]
#         extended_value = torch.cat([value[:, :, :text_token_count], global_value], dim=2)  # Same shape
#
#         # Apply extended self-attention
#         hidden_states = F.scaled_dot_product_attention(query, extended_key, extended_value, dropout_p=0.0, is_causal=False)
#
#     return hidden_states


#
# def extended_attention_batch2(query, key, value, curr_step, t_range, text_token_count=77):
#     # Check if we are in the extended attention range
#     apply_extended = any(start <= curr_step <= end for start, end in t_range)
#
#     # First image attends normally. query.shape torch.Size([2, 24, 4173, 128]) key.shape torch.Size([2, 24, 4173, 128]) value.shape torch.Size([2, 24, 4173, 128]) hidden_states_0.shape  torch.Size([1, 24, 4173, 128])
#     hidden_states_0 = F.scaled_dot_product_attention(query[:1], key[:1], value[:1], dropout_p=0.0, is_causal=False)
#
#     if apply_extended:
#         # Select only the image tokens (not text tokens) from the first image. added_key.shape torch.Size([24, 4096, 128]) added_value.shape torch.Size([24, 4096, 128])
#         added_key = key[0, :, text_token_count:]
#         added_value = value[0, :, text_token_count:]
#
#         # Extend key and value for the second image by adding first image's image tokens. extended_key.shape torch.Size([1, 24, 8269, 128]) extended_value.shape torch.Size([1, 24, 8269, 128])
#         extended_key = torch.cat([added_key, key[1]], dim=1).unsqueeze(0)
#         extended_value = torch.cat([added_value, value[1]], dim=1).unsqueeze(0)
#
#         # Compute extended attention for the second image hidden_states_1.shape torch.Size([1, 24, 4173, 128])
#         hidden_states_1 = F.scaled_dot_product_attention(query[1:2], extended_key, extended_value, dropout_p=0.0, is_causal=False)
#     else:
#         # Second image attends normally if not in the t_range
#         hidden_states_1 = F.scaled_dot_product_attention(query[1:2], key[1:2], value[1:2], dropout_p=0.0, is_causal=False)
#
#     # Concatenate outputs hidden_states.shape torch.Size([2, 24, 4173, 128])
#     hidden_states = torch.cat([hidden_states_0, hidden_states_1], dim=0)
#
#     return hidden_states



# class AttentionFluxSingleAttnProcessor2_0:
#     r"""
#     Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
#     """
#     def __init__(self,  layer_name, attention_store):
#         self.layer_name = layer_name
#         self.attention_store = attention_store
#
#     def __call__(
#         self,
#         attn: Attention,
#         hidden_states: torch.Tensor,
#         encoder_hidden_states: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         image_rotary_emb: Optional[torch.Tensor] = None,
#         step_index: Optional[int] = None,
#     ) -> torch.Tensor:
#         input_ndim = hidden_states.ndim
#
#         if input_ndim == 4:
#             batch_size, channel, height, width = hidden_states.shape
#             hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
#
#         batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
#
#         query = attn.to_q(hidden_states)
#         if encoder_hidden_states is None:
#             encoder_hidden_states = hidden_states
#
#         key = attn.to_k(encoder_hidden_states)
#         value = attn.to_v(encoder_hidden_states)
#
#         inner_dim = key.shape[-1]
#         head_dim = inner_dim // attn.heads
#
#         query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
#
#         key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
#         value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
#
#         if attn.norm_q is not None:
#             query = attn.norm_q(query)
#         if attn.norm_k is not None:
#             key = attn.norm_k(key)
#
#         # Apply RoPE if needed
#         if image_rotary_emb is not None:
#             # YiYi to-do: update uising apply_rotary_emb
#             # from ..embeddings import apply_rotary_emb
#             # query = apply_rotary_emb(query, image_rotary_emb)
#             # key = apply_rotary_emb(key, image_rotary_emb)
#             query, key = apply_rope(query, key, image_rotary_emb)
#
#         # the output of sdp = (batch, num_heads, seq_len, head_dim)
#         # TODO: add support for attn.scale when we move to Torch 2.1
#         # hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
#         hidden_states, attention_probs = apply_standard_attention(query, key, value, attn)
#         self.attention_store.store_attention(attention_probs, step_index, self.layer_name, batch_size, attn.heads)
#
#
#         hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
#         hidden_states = hidden_states.to(query.dtype)
#
#         if input_ndim == 4:
#             hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
#
#         return hidden_states
#
# class AttentionFluxAttnProcessor2_0:
#     """Attention processor used typically in processing the SD3-like self-attention projections."""
#
#     def __init__(self,  layer_name, attention_store):
#         self.layer_name = layer_name
#         self.attention_store = attention_store
#
#     def __call__(
#         self,
#         attn: Attention,
#         hidden_states: torch.FloatTensor,
#         encoder_hidden_states: torch.FloatTensor = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         image_rotary_emb: Optional[torch.Tensor] = None,
#         step_index: Optional[int] = None,
#     ) -> torch.FloatTensor:
#         input_ndim = hidden_states.ndim
#         if input_ndim == 4:
#             batch_size, channel, height, width = hidden_states.shape
#             hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
#         context_input_ndim = encoder_hidden_states.ndim
#         if context_input_ndim == 4:
#             batch_size, channel, height, width = encoder_hidden_states.shape
#             encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
#
#         batch_size = encoder_hidden_states.shape[0]
#
#         # `sample` projections.
#         query = attn.to_q(hidden_states)
#         key = attn.to_k(hidden_states)
#         value = attn.to_v(hidden_states)
#
#         inner_dim = key.shape[-1]
#         head_dim = inner_dim // attn.heads
#
#         query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
#         key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
#         value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
#
#         if attn.norm_q is not None:
#             query = attn.norm_q(query)
#         if attn.norm_k is not None:
#             key = attn.norm_k(key)
#
#         # `context` projections.
#         encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
#         encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
#         encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
#
#         encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
#         encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
#         encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
#
#         if attn.norm_added_q is not None:
#             encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
#         if attn.norm_added_k is not None:
#             encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)
#
#         # attention
#         query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
#         key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
#         value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
#
#         if image_rotary_emb is not None:
#             # YiYi to-do: update uising apply_rotary_emb
#             # from ..embeddings import apply_rotary_emb
#             # query = apply_rotary_emb(query, image_rotary_emb)
#             # key = apply_rotary_emb(key, image_rotary_emb)
#             query, key = apply_rope(query, key, image_rotary_emb)
#
#         # hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
#         hidden_states, attention_probs = apply_standard_attention(query, key, value, attn)
#         self.attention_store.store_attention(attention_probs, step_index, self.layer_name, batch_size, attn.heads)
#
#         hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
#         hidden_states = hidden_states.to(query.dtype)
#
#         encoder_hidden_states, hidden_states = (
#             hidden_states[:, : encoder_hidden_states.shape[1]],
#             hidden_states[:, encoder_hidden_states.shape[1] :],
#         )
#
#         # linear proj
#         hidden_states = attn.to_out[0](hidden_states)
#         # dropout
#         hidden_states = attn.to_out[1](hidden_states)
#         encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
#
#         if input_ndim == 4:
#             hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
#         if context_input_ndim == 4:
#             encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
#
#         return hidden_states, encoder_hidden_states


class FluxAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, attention_store, extended_attn_kwargs):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.t_range = extended_attn_kwargs.get('t_range', [])
        self.attention_store = attention_store

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        dropout_value: Optional[float] = 0.0,
    ) -> torch.FloatTensor:
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size = encoder_hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

        # attention
        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            # YiYi to-do: update uising apply_rotary_emb
            # from ..embeddings import apply_rotary_emb
            # query = apply_rotary_emb(query, image_rotary_emb)
            # key = apply_rotary_emb(key, image_rotary_emb)
            query, key = apply_rope(query, key, image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        encoder_hidden_states, hidden_states = (
            hidden_states[:, : encoder_hidden_states.shape[1]],
            hidden_states[:, encoder_hidden_states.shape[1] :],
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states, encoder_hidden_states

class ExtendedFluxAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, attention_store, extended_attn_kwargs):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.attention_store = attention_store
        self.t_range = extended_attn_kwargs.get('t_range', [])

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        dropout_value: Optional[float] = 0.0,
    ) -> torch.FloatTensor:
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size = encoder_hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

        # attention

        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            query, key = apply_rope(query, key, image_rotary_emb)

        # # We make extended key and value by concatenating the original key and value with the query.
        # hidden_states = extended_self_attention(query, key, value)
        curr_step = self.attention_store.curr_iter
        t_range = self.t_range
        text_token_count = query.shape[2] - 4096 # Assuming image of shape 1024x1024
        hidden_states = anchored_attention_batch(query, key, value, curr_step, t_range, text_token_count=text_token_count, dropout_value=dropout_value)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        encoder_hidden_states, hidden_states = (
            hidden_states[:, : encoder_hidden_states.shape[1]],
            hidden_states[:, encoder_hidden_states.shape[1] :],
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        # hidden_states.shape torch.Size([2, 4096, 3072])
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        # encoder_hidden_states.shape torch.Size([2, 77, 3072])

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states, encoder_hidden_states


class FluxSingleAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, attention_store, extended_attn_kwargs):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.t_range = extended_attn_kwargs.get('t_range', [])
        self.attention_store = attention_store


    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        dropout_value: Optional[float] = 0.0,
    ) -> torch.Tensor:
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            # YiYi to-do: update uising apply_rotary_emb
            # from ..embeddings import apply_rotary_emb
            # query = apply_rotary_emb(query, image_rotary_emb)
            # key = apply_rotary_emb(key, image_rotary_emb)
            query, key = apply_rope(query, key, image_rotary_emb)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states

class ExtendedFluxSingleAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, attention_store, extended_attn_kwargs):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.t_range = extended_attn_kwargs.get('t_range', [])
        self.attention_store = attention_store

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        dropout_value: Optional[float] = 0.0,
    ) -> torch.Tensor:
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            # YiYi to-do: update uising apply_rotary_emb
            # from ..embeddings import apply_rotary_emb
            # query = apply_rotary_emb(query, image_rotary_emb)
            # key = apply_rotary_emb(key, image_rotary_emb)
            query, key = apply_rope(query, key, image_rotary_emb)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1

        # # concat keys and values by concat on the second dimension
        # hidden_states = extended_self_attention(query, key, value)
        curr_step = self.attention_store.curr_iter
        t_range = self.t_range
        text_token_count = query.shape[2] - 4096 # Assuming image of shape 1024x1024
        hidden_states = anchored_attention_batch(query, key, value, curr_step, t_range, text_token_count=text_token_count, dropout_value=dropout_value)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states



