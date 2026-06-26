#!/usr/bin/env python

# Copyright 2025 Reflex team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Attention Layer with KV Cache.

This module provides a minimal scaled dot-product attention implementation
with optional KV caching for efficient autoregressive inference.

The KV cache strategy:
- First call with use_cache=True: Initialize cache with K/V (prefix prefill)
- Subsequent calls: Concatenate cached prefix K/V with new suffix K/V

"""

# TODO: use flashinfer kernels

import torch
import torch.nn.functional as F
from torch import nn


class Attention(nn.Module):
    """Scaled dot-product attention with optional KV cache.
    
    Computes: Attention(Q, K, V) = softmax(Q @ K^T / scale) @ V
    
    The KV cache enables efficient inference by storing prefix K/V
    and reusing them across multiple forward passes.
    """

    def __init__(self, scale: float):
        """Initialize attention layer.
        
        Args:
            scale: Scaling factor for attention scores, typically 1/sqrt(head_dim).
        """
        super().__init__()
        self.scale = scale
        
        # KV cache buffers: [B, H, L_prefix, D]
        self.k_cache = None
        self.v_cache = None

    def reset_cache(self):
        """Clear KV cache. Call when starting a new sequence."""
        self.k_cache = None
        self.v_cache = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Compute scaled dot-product attention.
        
        Args:
            q: Query tensor [B, H, L_q, D].
            k: Key tensor [B, H, L_k_new, D].
            v: Value tensor [B, H, L_v_new, D].
            attention_mask: Additive mask [B, 1, L_q, L_k] or [B, H, L_q, L_k].
                           Use 0 for positions to attend, -inf for masked positions.
            use_cache: If True, use KV caching for prefix tokens.
            
        Returns:
            Output tensor [B, H, L_q, D].
        """
        # Handle KV cache
        if use_cache:
            if self.k_cache is None:
                # First call: initialize cache with prefix K/V
                self.k_cache = k.detach()
                self.v_cache = v.detach()
                k_full = k
                v_full = v
            else:
                # Subsequent calls: concatenate cached prefix with new suffix
                # Note: cache is not updated, always stores prefix only
                k_full = torch.cat([self.k_cache, k], dim=2)
                v_full = torch.cat([self.v_cache, v], dim=2)
        else:
            k_full = k
            v_full = v

        # Compute attention scores: [B, H, L_q, L_k]
        attn_scores = torch.matmul(q, k_full.transpose(-2, -1)) * self.scale

        # Apply attention mask (additive)
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                mask = attention_mask
            elif attention_mask.dim() == 3:
                mask = attention_mask[:, None, :, :]
            else:
                raise ValueError(f"Unsupported attention_mask ndim: {attention_mask.ndim}")
            
            # Ensure mask matches attn_scores shape
            # attn_scores: [B, H, L_q, L_k] where L_k = k_full.shape[2]
            # mask: [B, H, L_q_mask, L_k_mask]
            B, H, L_q, L_k = attn_scores.shape
            mask_B, mask_H, mask_L_q, mask_L_k = mask.shape
            
            # If mask's last dimension doesn't match k_full length, pad or slice
            if mask_L_k != L_k:
                if mask_L_k < L_k:
                    # Pad mask to match k_full length
                    pad_size = L_k - mask_L_k
                    mask = F.pad(mask, (0, pad_size), value=float('-inf'))
                else:
                    # Slice mask to match k_full length
                    mask = mask[:, :, :, :L_k]
            
            # Ensure query dimension matches
            if mask_L_q != L_q:
                if mask_L_q < L_q:
                    # Pad query dimension
                    pad_size = L_q - mask_L_q
                    mask = F.pad(mask, (0, 0, 0, pad_size), value=float('-inf'))
                else:
                    # Slice query dimension
                    mask = mask[:, :, :L_q, :]
            
            # If mask is provided with a single head dimension (H=1), expand it to match
            # the number of attention heads for correct boolean indexing and masking.
            if mask.shape[1] == 1 and attn_scores.shape[1] != 1:
                mask = mask.expand(-1, attn_scores.shape[1], -1, -1)
            
            # IMPORTANT: Some masked key positions may have NaN scores (e.g., due to NaNs in padded K/V).
            # Using "+ mask" would keep NaNs (NaN + -inf = NaN) and poison the softmax.
            # Instead, convert the additive mask to a boolean "allow" mask and overwrite all
            # disallowed positions with -inf (hard mask), which is robust to NaNs.
            allow = mask == 0
            attn_scores = torch.where(
                allow,
                attn_scores,
                torch.full_like(attn_scores, float("-inf")),
            )
            # For fully-masked query rows (e.g., padded query tokens), softmax would return NaNs.
            # Keep them numerically safe by allowing a single dummy key (index 0).
            row_all_masked = ~allow.any(dim=-1)  # [B, H, L_q]
            if row_all_masked.any():
                # Boolean indexing returns a [N_rows, L_k] view; set first key score to 0.
                masked_rows = attn_scores[row_all_masked]
                masked_rows[..., 0] = 0.0
                attn_scores[row_all_masked] = masked_rows

        # Softmax and weighted sum
        attn_weights = torch.softmax(attn_scores, dim=-1)
        # Extra safety: replace any remaining NaNs (shouldn't happen after hard masking)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)
        out = torch.matmul(attn_weights, v_full)
        
        return out
