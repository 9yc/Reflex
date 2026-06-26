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
"""Streaming Input Manager for incremental processing.

This module implements the streaming input management for StreamingThinker-style
incremental processing, maintaining prefix history and supporting dynamic growth.
"""

import torch
from torch import Tensor
from typing import Optional


class StreamingInputManager:
    """Manages streaming input history and prefix length.
    
    This class maintains the history of prefix embeddings and tracks the
    current prefix length, supporting incremental processing as described
    in StreamingThinker Algorithm 1.
    """
    
    def __init__(self, max_history_length: Optional[int] = None):
        """Initialize streaming input manager.
        
        Args:
            max_history_length: Maximum number of prefix chunks to keep in history.
                               If None, keeps all history (no limit).
        """
        self.max_history_length = max_history_length
        self.prefix_history: list[Tensor] = []
        self.current_prefix_length = 0
        self.prefix_pad_masks_history: list[Tensor] = []
        self.prefix_att_masks_history: list[Tensor] = []
    
    def reset(self):
        """Reset the streaming state. Call when starting a new sequence."""
        self.prefix_history.clear()
        self.prefix_pad_masks_history.clear()
        self.prefix_att_masks_history.clear()
        self.current_prefix_length = 0
    
    def add_new_prefix(
        self,
        prefix_embs: Tensor,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor,
    ) -> tuple[int, int]:
        """Add new prefix embeddings to history (incremental processing).
        
        This implements the "Parallel Prefill" step from Algorithm 1:
        new input is appended to history without stopping reasoning.
        
        Args:
            prefix_embs: New prefix embeddings [B, L_new, D].
            prefix_pad_masks: Padding masks for new prefix [B, L_new].
            prefix_att_masks: Attention masks for new prefix [B, L_new].
            
        Returns:
            (updated_prefix_length, removed_tokens) where removed_tokens is the number
            of tokens dropped due to max_history_length cap.
        """
        # Append to history
        self.prefix_history.append(prefix_embs.detach())
        self.prefix_pad_masks_history.append(prefix_pad_masks.detach())
        self.prefix_att_masks_history.append(prefix_att_masks.detach())
        
        # Update prefix length
        new_length = prefix_embs.shape[1]
        self.current_prefix_length += new_length
        
        # Maintain history length limit
        removed_tokens = 0
        if self.max_history_length is not None:
            while len(self.prefix_history) > self.max_history_length:
                removed = self.prefix_history.pop(0)
                self.current_prefix_length -= removed.shape[1]
                removed_tokens += removed.shape[1]
                self.prefix_pad_masks_history.pop(0)
                self.prefix_att_masks_history.pop(0)
        
        return self.current_prefix_length, removed_tokens
    
    def get_full_prefix(
        self,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Get concatenated prefix embeddings from history.
        
        Args:
            start_idx: Start index in the full prefix sequence.
            end_idx: End index in the full prefix sequence (exclusive).
                     If None, returns all.
            
        Returns:
            Tuple of (prefix_embs, prefix_pad_masks, prefix_att_masks).
        """
        if len(self.prefix_history) == 0:
            raise ValueError("No prefix history available")
        
        # Concatenate all history
        full_prefix_embs = torch.cat(self.prefix_history, dim=1)
        full_prefix_pad_masks = torch.cat(self.prefix_pad_masks_history, dim=1)
        full_prefix_att_masks = torch.cat(self.prefix_att_masks_history, dim=1)
        
        # Apply slicing if specified
        if end_idx is None:
            end_idx = full_prefix_embs.shape[1]
        
        prefix_embs = full_prefix_embs[:, start_idx:end_idx]
        prefix_pad_masks = full_prefix_pad_masks[:, start_idx:end_idx]
        prefix_att_masks = full_prefix_att_masks[:, start_idx:end_idx]
        
        return prefix_embs, prefix_pad_masks, prefix_att_masks
    
    def get_prefix_length(self) -> int:
        """Get current prefix length."""
        return self.current_prefix_length
    
    def has_history(self) -> bool:
        """Check if there is any prefix history."""
        return len(self.prefix_history) > 0





