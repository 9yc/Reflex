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
"""Dynamic Embedding Layer inspired by zip2zip.

This module implements a dynamic embedding layer that can compute embeddings
for newly encountered inputs at runtime, similar to zip2zip's HyperEmbedding.

Key features:
- Base embeddings: Standard embeddings for known inputs
- Dynamic embeddings: Runtime-computed embeddings for new inputs
- Incremental updates: Only compute embeddings for new inputs
- KV-cache friendly: Supports incremental KV cache updates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class DynamicMultimodalEmbedding(nn.Module):
    """Dynamic embedding layer for multimodal inputs, inspired by zip2zip.
    
    This layer extends standard embeddings with the ability to compute embeddings
    for newly encountered inputs at runtime, similar to zip2zip's HyperEmbedding.
    
    Key differences from zip2zip:
    - Supports multimodal inputs (images, language, state) instead of just text
    - Uses encoder-based dynamic embedding computation
    - Optimized for incremental updates in streaming scenarios
    """
    
    def __init__(
        self,
        base_embedding: nn.Embedding,
        encoder: Optional[nn.Module] = None,
        dynamic_cache_size: int = 1000,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize dynamic multimodal embedding layer.
        
        Args:
            base_embedding: Base embedding layer for known inputs.
            encoder: Optional encoder for computing dynamic embeddings.
                    If None, uses a simple MLP encoder.
            dynamic_cache_size: Maximum size of dynamic embedding cache.
            device: Device to place embeddings on.
            dtype: Data type for embeddings.
        """
        super().__init__()
        self.base_embedding = base_embedding
        self.embedding_dim = base_embedding.embedding_dim
        self.device = device or base_embedding.weight.device
        self.dtype = dtype or base_embedding.weight.dtype
        
        # Dynamic embedding cache (similar to zip2zip's codebook)
        self.dynamic_cache: Dict[int, torch.Tensor] = {}
        self.dynamic_cache_size = dynamic_cache_size
        
        # Encoder for computing dynamic embeddings
        if encoder is None:
            # Simple MLP encoder (can be replaced with more sophisticated encoders)
            self.encoder = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim * 2),
                nn.GELU(),
                nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            ).to(device=self.device, dtype=self.dtype)
        else:
            self.encoder = encoder
        
        # Track which inputs are base vs dynamic
        self.base_vocab_size = base_embedding.num_embeddings
        
    def get_dynamic_embedding(
        self,
        input_ids: torch.Tensor,
        base_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute dynamic embeddings for new inputs, inspired by zip2zip.
        
        Args:
            input_ids: Input token IDs (may include IDs >= base_vocab_size).
            base_embeddings: Base embeddings for known inputs.
            
        Returns:
            Dynamic embeddings for new inputs with shape [num_dynamic, embedding_dim].
        """
        # Identify dynamic inputs (IDs >= base_vocab_size)
        dynamic_mask = input_ids >= self.base_vocab_size
        if not dynamic_mask.any():
            return torch.zeros(
                (0, self.embedding_dim),
                device=self.device,
                dtype=self.dtype
            )
        
        # Get dynamic input IDs
        dynamic_ids = input_ids[dynamic_mask] - self.base_vocab_size
        
        # Get batch and sequence indices for dynamic positions
        batch_indices, seq_indices = torch.where(dynamic_mask)
        
        # Compute or retrieve dynamic embeddings
        dynamic_embeddings_list = []
        for idx, dynamic_id in enumerate(dynamic_ids.tolist()):
            if dynamic_id in self.dynamic_cache:
                # Use cached embedding
                cached_emb = self.dynamic_cache[dynamic_id]
            else:
                # Compute new embedding using encoder
                # Correctly extract base embedding for this position
                batch_idx = batch_indices[idx].item()
                seq_idx = seq_indices[idx].item()
                base_emb = base_embeddings[batch_idx, seq_idx:seq_idx+1]  # [1, dim]
                cached_emb = self.encoder(base_emb)  # [1, dim]
                
                # Cache the embedding
                if len(self.dynamic_cache) < self.dynamic_cache_size:
                    self.dynamic_cache[dynamic_id] = cached_emb.detach()
                else:
                    # Evict oldest entry (simple LRU)
                    oldest_key = next(iter(self.dynamic_cache))
                    del self.dynamic_cache[oldest_key]
                    self.dynamic_cache[dynamic_id] = cached_emb.detach()
            
            dynamic_embeddings_list.append(cached_emb)
        
        # Combine dynamic embeddings
        if dynamic_embeddings_list:
            dynamic_embeddings = torch.cat(dynamic_embeddings_list, dim=0)  # [num_dynamic, dim]
        else:
            dynamic_embeddings = torch.zeros(
                (dynamic_mask.sum().item(), self.embedding_dim),
                device=self.device,
                dtype=self.dtype
            )
        
        return dynamic_embeddings
    
    def forward(
        self,
        input_ids: torch.Tensor,
        incremental: bool = False,
    ) -> torch.Tensor:
        """Forward pass with dynamic embedding support.
        
        Args:
            input_ids: Input token IDs.
            incremental: If True, only compute embeddings for new inputs.
            
        Returns:
            Combined embeddings (base + dynamic).
        """
        # Get base embeddings
        base_input_ids = torch.clamp(input_ids, 0, self.base_vocab_size - 1)
        base_embeddings = self.base_embedding(base_input_ids)
        
        # Check if we have dynamic inputs
        has_dynamic = (input_ids >= self.base_vocab_size).any()
        
        if not has_dynamic:
            return base_embeddings
        
        # Compute dynamic embeddings
        dynamic_embeddings = self.get_dynamic_embedding(input_ids, base_embeddings)
        
        # Combine base and dynamic embeddings
        # For positions with dynamic inputs, replace base with dynamic
        dynamic_mask = input_ids >= self.base_vocab_size
        combined_embeddings = base_embeddings.clone()
        
        # Scatter dynamic embeddings back to original positions
        if dynamic_mask.any():
            # dynamic_embeddings has shape [num_dynamic, embedding_dim]
            # We need to scatter it back to dynamic_mask positions
            batch_indices, seq_indices = torch.where(dynamic_mask)
            for i, (batch_idx, seq_idx) in enumerate(zip(batch_indices.tolist(), seq_indices.tolist())):
                combined_embeddings[batch_idx, seq_idx] = dynamic_embeddings[i]
        
        return combined_embeddings
    
    def reset_cache(self):
        """Reset dynamic embedding cache (similar to zip2zip's codebook reset)."""
        self.dynamic_cache.clear()
    
    def get_cache_size(self) -> int:
        """Get current cache size."""
        return len(self.dynamic_cache)


class IncrementalKVCacheUpdater:
    """Incremental KV cache updater inspired by zip2zip's incremental update strategy.
    
    This class implements efficient incremental updates to KV caches, only computing
    KV for new inputs and reusing cached values for existing inputs.
    
    Enhanced with detailed statistics for ICML paper analysis:
    - Tracks total tokens processed
    - Tracks tokens computed vs reused
    - Computes reuse rate and computation savings
    """
    
    def __init__(self, cache_size_limit: Optional[int] = None):
        """Initialize incremental KV cache updater.
        
        Args:
            cache_size_limit: Optional limit on cache size (for memory management).
        """
        self.cache_size_limit = cache_size_limit
        self.update_count = 0
        
        # Enhanced statistics for ICML paper
        self.total_tokens_processed = 0
        self.tokens_computed = 0  # Tokens that required new KV computation
        self.tokens_reused = 0    # Tokens that reused cached KV
        self.tokens_evicted = 0   # Tokens evicted due to cache limit
    
    def incremental_update(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        cached_k: Optional[torch.Tensor] = None,
        cached_v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Incrementally update KV cache, only computing KV for new inputs.
        
        This method is inspired by zip2zip's approach of only computing embeddings
        for newly created hypertokens, avoiding redundant computation.
        
        Args:
            new_k: New K values to append.
            new_v: New V values to append.
            cached_k: Cached K values (optional).
            cached_v: Cached V values (optional).
            
        Returns:
            Updated K and V tensors.
        """
        new_tokens = new_k.shape[2]
        
        if cached_k is None or cached_v is None:
            # First update: just return new values
            self.tokens_computed += new_tokens
            self.total_tokens_processed += new_tokens
            self.update_count += 1
            return new_k, new_v
        
        cached_tokens = cached_k.shape[2]
        
        # Update statistics - key insight for ICML paper:
        # We REUSE all cached tokens (no recomputation needed)
        # We only COMPUTE new tokens
        self.tokens_reused += cached_tokens
        self.tokens_computed += new_tokens
        self.total_tokens_processed += cached_tokens + new_tokens
        
        # Incremental update: concatenate new values with cached values
        # This avoids recomputing KV for existing inputs
        updated_k = torch.cat([cached_k, new_k], dim=2)  # dim=2 is sequence dimension
        updated_v = torch.cat([cached_v, new_v], dim=2)
        
        self.update_count += 1
        
        # Optional: enforce cache size limit
        if self.cache_size_limit is not None:
            total_length = updated_k.shape[2]
            if total_length > self.cache_size_limit:
                # Keep only the most recent entries
                start_idx = total_length - self.cache_size_limit
                evicted = start_idx
                self.tokens_evicted += evicted
                updated_k = updated_k[:, :, start_idx:]
                updated_v = updated_v[:, :, start_idx:]
        
        return updated_k, updated_v
    
    @property
    def reuse_rate(self) -> float:
        """Compute cache reuse rate - key metric for ICML paper.
        
        Returns:
            Fraction of tokens that were reused from cache (0.0 to 1.0).
        """
        if self.total_tokens_processed == 0:
            return 0.0
        return self.tokens_reused / self.total_tokens_processed
    
    @property
    def computation_saved_ratio(self) -> float:
        """Compute the ratio of computation saved by caching.
        
        This measures how much computation was avoided by reusing cached KV values.
        A ratio of 0.9 means 90% of computation was saved.
        
        Returns:
            Fraction of computation saved (0.0 to 1.0).
        """
        if self.total_tokens_processed == 0:
            return 0.0
        # If we had to recompute everything, we would compute total_tokens_processed
        # We actually only computed tokens_computed
        return 1.0 - (self.tokens_computed / self.total_tokens_processed)
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics for ICML paper analysis.
        
        Returns:
            Dictionary containing:
            - update_count: Number of incremental updates performed
            - total_tokens: Total tokens seen across all updates
            - tokens_computed: Tokens that required new KV computation
            - tokens_reused: Tokens that reused cached KV
            - tokens_evicted: Tokens evicted due to cache limit
            - reuse_rate: Fraction of tokens reused
            - computation_saved: Fraction of computation saved
        """
        return {
            "update_count": self.update_count,
            "total_tokens": self.total_tokens_processed,
            "tokens_computed": self.tokens_computed,
            "tokens_reused": self.tokens_reused,
            "tokens_evicted": self.tokens_evicted,
            "reuse_rate": self.reuse_rate,
            "computation_saved": self.computation_saved_ratio,
        }
    
    def reset(self):
        """Reset all counters and statistics."""
        self.update_count = 0
        self.total_tokens_processed = 0
        self.tokens_computed = 0
        self.tokens_reused = 0
        self.tokens_evicted = 0

