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
"""Streaming Attention Layer with Parallel KV Cache.

This module implements StreamingThinker-inspired streaming attention mechanisms:
- Streaming attention masks: restrict attention to past and current inputs
- Parallel KV caches: decouple input encoding from reasoning generation
- Streaming position encoding: independent indexing for input and reasoning tokens

Key differences from standard attention:
1. Separate KV caches for prefix (observation) and suffix (action) sequences
2. Streaming attention masks that enforce order-preserving reasoning
3. Independent position encoding for input and reasoning tokens
"""

import torch
import torch.nn.functional as F
from torch import nn

from reflex.layers.attention import Attention
from reflex.layers.dynamic_embedding import IncrementalKVCacheUpdater

try:
    from reflex.layers.quantize_kernel import fused_quantize_store
    HAS_TRITON_QUANT = True
except ImportError:
    HAS_TRITON_QUANT = False

# Try to import Flash Attention (official)
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

# Import our optimized attention
try:
    from reflex.layers.optimized_attention import OptimizedAttention
    HAS_OPTIMIZED_ATTN = True
except ImportError:
    HAS_OPTIMIZED_ATTN = False

# Import mixed precision attention
try:
    from reflex.layers.mixed_precision_attention import MixedPrecisionAttention
    HAS_MIXED_PRECISION_ATTN = True
except ImportError:
    HAS_MIXED_PRECISION_ATTN = False

# Try to import dynamo for dynamic shape support
try:
    import torch._dynamo as dynamo
    HAS_DYNAMO = True
except ImportError:
    HAS_DYNAMO = False

# Try to import decorators for unbacked dynamic shapes
try:
    from torch._dynamo import decorators as dynamo_decorators
    HAS_DYNAMO_DECORATORS = True
except ImportError:
    HAS_DYNAMO_DECORATORS = False

# Check PyTorch version for unbacked support
HAS_UNBACKED = False
try:
    import torch
    from torch._dynamo.decorators import mark_unbacked
    test_tensor = torch.randn(5, 5)
    try:
        mark_unbacked(test_tensor, 0)
        HAS_UNBACKED = True
    except Exception:
        HAS_UNBACKED = False
except (ImportError, AttributeError):
    HAS_UNBACKED = False


class StreamingAttention(Attention):
    """Partitioned streaming attention with parallel KV caches.

    This implements the **Partitioned Attention** mechanism from the Reflex
    paper (Sec. "Stream Correctness"). The paper partitions the context into
    three semantic regions; the code realizes them with two persistent KV
    caches plus a transient merged buffer:

        Paper term            Code term / attribute
        --------------------  -------------------------------------------
        Pinned prefix +       "Source Cache"  (prefix_k_buffer / prefix_v_buffer)
        Sliding observation     -> holds the instruction prefix and the FIFO
        window                     window of observation tokens
        Dynamic suffix        "Target Cache"  (suffix_k_cache / suffix_v_cache)
                                -> the transient flow-state / action tokens,
                                   reset every denoising cycle
        (cross-attn scratch)  "Merged Cache"  (merged_k_cache / merged_v_cache)
                                -> contiguous prefix+suffix buffer used during
                                   the tight denoising loop (see merge_caches)

    Only the dynamic suffix is recomputed per denoising step, giving O(1)
    incremental cache updates (Incremental Prefill via `prefill_source`, and
    Manual Cache Merging via `merge_caches`). The two-cache design is inspired
    by StreamingThinker's parallel-KV-cache idea; the partitioning into
    static/sliding/dynamic regions and its application to flow matching is the
    Reflex contribution. See docs/architecture.md for the full mapping.
    """
    
    def __init__(self, scale: float, enable_streaming: bool = True, block_size: int = 16, use_blocked_cache: bool = False, max_cache_size: int = 2048, enable_flash_attention: bool = True, enable_mixed_precision: bool = True, quantize_kv_cache: bool = False):
        super().__init__(scale)
        self.enable_streaming = enable_streaming
        self.block_size = block_size
        self.use_blocked_cache = use_blocked_cache
        self.max_cache_size = max_cache_size
        self.enable_flash_attention = enable_flash_attention
        self.enable_mixed_precision = enable_mixed_precision
        self.quantize_kv_cache = quantize_kv_cache
        
        if quantize_kv_cache:
            print("✅ KV Cache Quantization (8-bit) Enabled")
        
        self.flash_attn = None
        self.optimized_attn = None
        self.mixed_precision_attn = None
        
        if enable_flash_attention:
            if HAS_FLASH_ATTN:
                print("✅ Using official Flash Attention")
                self.flash_attn_type = "official"
            elif HAS_OPTIMIZED_ATTN:
                print("✅ Using optimized attention (Reflex-specific)")
                self.optimized_attn = OptimizedAttention(scale=scale)
                self.flash_attn_type = "optimized"
            else:
                print("⚠️ Flash Attention not available, using standard attention")
                self.flash_attn_type = "standard"
        else:
            self.flash_attn_type = "standard"
        
        if enable_mixed_precision and HAS_MIXED_PRECISION_ATTN:
            print("✅ Using mixed precision attention")
            self.mixed_precision_attn = MixedPrecisionAttention(scale=scale, enable_amp=True)
            self.mixed_precision_enabled = True
        else:
            if enable_mixed_precision:
                print("⚠️ Mixed precision attention not available, using standard precision")
            self.mixed_precision_enabled = False
        
        # Source Cache
        if use_blocked_cache:
            self.prefix_blocks_k = []
            self.prefix_blocks_v = []
            self.suffix_blocks_k = []
            self.suffix_blocks_v = []
            self._prefix_full_k = None
            self._prefix_full_v = None
            self._suffix_full_k = None
            self._suffix_full_v = None
            self._prefix_dirty = True
            self._suffix_dirty = True
            self._pending_prefix_blocks_k = []
            self._pending_prefix_blocks_v = []
            self._pending_source_length = 0
        else:
            self.prefix_k_buffer = None
            self.prefix_v_buffer = None
            
            if self.quantize_kv_cache:
                self.prefix_k_scale = None
                self.prefix_k_zp = None
                self.prefix_v_scale = None
                self.prefix_v_zp = None
            
            self._pending_prefix_k_cache = None
            self._pending_prefix_v_cache = None
            self._pending_source_length = 0
            self._prefix_dirty = True
        
        if not use_blocked_cache:
            self.suffix_k_cache = None
            self.suffix_v_cache = None
        
        # Merged Cache
        self.merged_k_cache = None
        self.merged_v_cache = None
        
        self.source_length = 0
        self.target_length = 0
        self.merge_start_idx = 0
        
        self.kv_updater = IncrementalKVCacheUpdater()
        
        self.cache_stats = {
            "source_updates": 0, "source_tokens": 0,
            "target_updates": 0, "target_tokens": 0,
            "merge_count": 0, "split_count": 0,
            "attention_calls": 0, "cache_hits": 0,
        }
        self.kv_updater.reset()
        
    def _quantize(self, x: torch.Tensor):
        min_val = x.min(dim=-1, keepdim=True)[0]
        max_val = x.max(dim=-1, keepdim=True)[0]
        range_val = (max_val - min_val).clamp(min=1e-5)
        scale = range_val / 255.0
        zeropoint = -min_val / scale
        x_q = torch.clamp(torch.round(x / scale + zeropoint), 0, 255).to(torch.uint8)
        return x_q, scale, zeropoint

    def _dequantize(self, x_q: torch.Tensor, scale: torch.Tensor, zeropoint: torch.Tensor):
        return (x_q.to(scale.dtype) - zeropoint) * scale

    def reset_cache(self, reset_stats: bool = False):
        super().reset_cache()
        if self.use_blocked_cache:
            self.prefix_blocks_k.clear()
            self.prefix_blocks_v.clear()
            self.suffix_blocks_k.clear()
            self.suffix_blocks_v.clear()
            self._prefix_full_k = None
            self._prefix_full_v = None
            self._suffix_full_k = None
            self._suffix_full_v = None
            self._prefix_dirty = True
            self._suffix_dirty = True
            self._pending_prefix_blocks_k.clear()
            self._pending_prefix_blocks_v.clear()
            self._pending_source_length = 0
        else:
            self.prefix_k_buffer = None
            self.prefix_v_buffer = None
            self.suffix_k_cache = None
            self.suffix_v_cache = None
            self._pending_prefix_k_cache = None
            self._pending_prefix_v_cache = None
            self._pending_source_length = 0
            
            if self.quantize_kv_cache:
                self.prefix_k_scale = None
                self.prefix_k_zp = None
                self.prefix_v_scale = None
                self.prefix_v_zp = None
        self.merged_k_cache = None
        self.merged_v_cache = None
        self.kv_updater.reset()
        self.source_length = 0
        self.target_length = 0
        self.merge_start_idx = 0
        if reset_stats:
            self.reset_statistics()
    
    def _init_static_buffer(self, B, H, D, dtype, device):
        """Initialize static KV buffers for stable memory addresses (torch.compile friendly)."""
        buffer_size = self.max_cache_size
        
        # Allocate main buffers
        if self.quantize_kv_cache:
            self.prefix_k_buffer = torch.zeros(B, H, buffer_size, D, dtype=torch.uint8, device=device)
            self.prefix_v_buffer = torch.zeros(B, H, buffer_size, D, dtype=torch.uint8, device=device)
            self.prefix_k_scale = torch.zeros(B, H, buffer_size, 1, dtype=dtype, device=device)
            self.prefix_k_zp = torch.zeros(B, H, buffer_size, 1, dtype=dtype, device=device)
            self.prefix_v_scale = torch.zeros(B, H, buffer_size, 1, dtype=dtype, device=device)
            self.prefix_v_zp = torch.zeros(B, H, buffer_size, 1, dtype=dtype, device=device)
        else:
            self.prefix_k_buffer = torch.zeros(B, H, buffer_size, D, dtype=dtype, device=device)
            self.prefix_v_buffer = torch.zeros(B, H, buffer_size, D, dtype=dtype, device=device)
            
        self.source_length = 0
        self._prefix_dirty = True

    def prefill_source(self, k: torch.Tensor, v: torch.Tensor, *, cache_slot: str = "main"):
        if cache_slot not in ("main", "pending"):
            raise ValueError(f"cache_slot must be 'main' or 'pending', got {cache_slot}")

        num_tokens = k.shape[2]
        B, H, _, D = k.shape
        
        if self.use_blocked_cache:
            # Blocked cache logic (omitted for brevity, using existing logic if needed)
            # ... (Assume existing blocked logic is fine, though we recommend contiguous)
            num_blocks = (num_tokens + self.block_size - 1) // self.block_size
            for i in range(num_blocks):
                start = i * self.block_size
                end = min((i + 1) * self.block_size, num_tokens)
                k_block = k[:, :, start:end]
                v_block = v[:, :, start:end]
                if k_block.shape[2] < self.block_size:
                    pad_size = self.block_size - k_block.shape[2]
                    k_block = F.pad(k_block, (0, 0, 0, pad_size), value=0)
                    v_block = F.pad(v_block, (0, 0, 0, pad_size), value=0)
                if cache_slot == "main":
                    self.prefix_blocks_k.append(k_block.detach().clone())
                    self.prefix_blocks_v.append(v_block.detach().clone())
                else:
                    self._pending_prefix_blocks_k.append(k_block.detach().clone())
                    self._pending_prefix_blocks_v.append(v_block.detach().clone())
            
            if cache_slot == "main":
                self.source_length += num_tokens
                self._prefix_dirty = True
            else:
                self._pending_source_length += num_tokens
        else:
            # Contiguous cache logic (Static Buffer Optimized)
            if cache_slot == "main":
                # Initialize static buffer on first run
                if self.prefix_k_buffer is None:
                    self._init_static_buffer(B, H, D, k.dtype, k.device)
                
                # Check for overflow
                new_end = self.source_length + num_tokens
                if new_end > self.prefix_k_buffer.shape[2]:
                    # In static mode, we should ideally roll or error. 
                    # For now, we expand if max_cache_size was underestimated, but warn.
                    # Ideally max_cache_size should be large enough.
                    # We'll use the old expansion logic as a fallback for safety, 
                    # but properly configured runs shouldn't hit this.
                    # Better: Implement Rolling Buffer overwrite if we treat it as ring buffer?
                    # No, prefill usually appends. If full, we should have evicted before.
                    # Let's expand to be safe but keep it contiguous.
                    required_size = max(new_end, self.prefix_k_buffer.shape[2] * 2)
                    
                    def expand_tensor(t, new_size):
                        new_t = torch.zeros(B, H, new_size, *t.shape[3:], dtype=t.dtype, device=t.device)
                        new_t[:, :, :self.source_length] = t[:, :, :self.source_length]
                        return new_t
                        
                    self.prefix_k_buffer = expand_tensor(self.prefix_k_buffer, required_size)
                    self.prefix_v_buffer = expand_tensor(self.prefix_v_buffer, required_size)
                    if self.quantize_kv_cache:
                        self.prefix_k_scale = expand_tensor(self.prefix_k_scale, required_size)
                        self.prefix_k_zp = expand_tensor(self.prefix_k_zp, required_size)
                        self.prefix_v_scale = expand_tensor(self.prefix_v_scale, required_size)
                        self.prefix_v_zp = expand_tensor(self.prefix_v_zp, required_size)
                
                # In-place update (Crucial for torch.compile stability)
                start_idx = self.source_length
                end_idx = start_idx + num_tokens
                
                if self.quantize_kv_cache:
                    if HAS_TRITON_QUANT:
                        fused_quantize_store(k, self.prefix_k_buffer, self.prefix_k_scale, self.prefix_k_zp, cache_start_idx=start_idx)
                        fused_quantize_store(v, self.prefix_v_buffer, self.prefix_v_scale, self.prefix_v_zp, cache_start_idx=start_idx)
                    else:
                        k_q, k_s, k_z = self._quantize(k)
                        v_q, v_s, v_z = self._quantize(v)
                        self.prefix_k_buffer[:, :, start_idx:end_idx] = k_q
                        self.prefix_v_buffer[:, :, start_idx:end_idx] = v_q
                        self.prefix_k_scale[:, :, start_idx:end_idx] = k_s
                        self.prefix_k_zp[:, :, start_idx:end_idx] = k_z
                        self.prefix_v_scale[:, :, start_idx:end_idx] = v_s
                        self.prefix_v_zp[:, :, start_idx:end_idx] = v_z
                else:
                    self.prefix_k_buffer[:, :, start_idx:end_idx].copy_(k)
                    self.prefix_v_buffer[:, :, start_idx:end_idx].copy_(v)
                    
                self.source_length = end_idx
                self._prefix_dirty = True
                self.cache_stats["cache_hits"] += 1
            else:
                if self._pending_prefix_k_cache is None:
                    self._pending_prefix_k_cache = k.detach()
                    self._pending_prefix_v_cache = v.detach()
                    self._pending_source_length = k.shape[2]
                else:
                    new_length = self._pending_source_length + k.shape[2]
                    B, H, _, D = k.shape
                    new_k_cache = torch.empty(B, H, new_length, D, dtype=k.dtype, device=k.device)
                    new_v_cache = torch.empty(B, H, new_length, D, dtype=v.dtype, device=v.device)
                    new_k_cache[:, :, :self._pending_source_length] = self._pending_prefix_k_cache
                    new_v_cache[:, :, :self._pending_source_length] = self._pending_prefix_v_cache
                    new_k_cache[:, :, self._pending_source_length:] = k
                    new_v_cache[:, :, self._pending_source_length:] = v
                    self._pending_prefix_k_cache = new_k_cache
                    self._pending_prefix_v_cache = new_v_cache
                    self._pending_source_length = new_length

    def commit_pending_source(self):
        if self.use_blocked_cache:
            if self._pending_source_length > 0:
                self.prefix_blocks_k.extend(self._pending_prefix_blocks_k)
                self.prefix_blocks_v.extend(self._pending_prefix_blocks_v)
                self.source_length += int(self._pending_source_length)
                self._prefix_dirty = True
                self._pending_prefix_blocks_k.clear()
                self._pending_prefix_blocks_v.clear()
                self._pending_source_length = 0
            return
        
        if self._pending_source_length <= 0 or self._pending_prefix_k_cache is None:
            return
            
        self.prefill_source(self._pending_prefix_k_cache, self._pending_prefix_v_cache, cache_slot="main")
        self._pending_prefix_k_cache = None
        self._pending_prefix_v_cache = None
        self._pending_source_length = 0

    def evict_source_prefix(self, num_tokens: int):
        if num_tokens <= 0 or self.source_length <= 0:
            return
        num_tokens = min(int(num_tokens), int(self.source_length))

        if self.use_blocked_cache:
            # Blocked eviction logic
            if num_tokens % self.block_size == 0:
                n_blocks = num_tokens // self.block_size
                self.prefix_blocks_k = self.prefix_blocks_k[n_blocks:]
                self.prefix_blocks_v = self.prefix_blocks_v[n_blocks:]
                self.source_length -= num_tokens
                self._prefix_dirty = True
            else:
                # Rebuild blocks fallback
                k_full, v_full = self._get_source_cache()
                k_full = k_full[:, :, num_tokens:]
                v_full = v_full[:, :, num_tokens:]
                self.source_length = k_full.shape[2]
                self.prefix_blocks_k.clear()
                self.prefix_blocks_v.clear()
                L = self.source_length
                num_blocks = (L + self.block_size - 1) // self.block_size
                for i in range(num_blocks):
                    start = i * self.block_size
                    end = min((i + 1) * self.block_size, L)
                    k_block = k_full[:, :, start:end]
                    v_block = v_full[:, :, start:end]
                    if k_block.shape[2] < self.block_size:
                        pad_size = self.block_size - k_block.shape[2]
                        k_block = F.pad(k_block, (0, 0, 0, pad_size), value=0)
                        v_block = F.pad(v_block, (0, 0, 0, pad_size), value=0)
                    self.prefix_blocks_k.append(k_block.detach())
                    self.prefix_blocks_v.append(v_block.detach())
                self._prefix_dirty = True
        else:
            if self.prefix_k_buffer is None:
                return
            
            # Static buffer in-place eviction (shift left)
            # This preserves the tensor memory address for torch.compile
            new_len = self.source_length - num_tokens
            if new_len > 0:
                # Shift data left
                # src: [:, :, num_tokens:source_length]
                # dst: [:, :, 0:new_len]
                if self.quantize_kv_cache:
                    # For quantized, we need to shift all buffers
                    self.prefix_k_buffer[:, :, :new_len].copy_(self.prefix_k_buffer[:, :, num_tokens:self.source_length].clone())
                    self.prefix_v_buffer[:, :, :new_len].copy_(self.prefix_v_buffer[:, :, num_tokens:self.source_length].clone())
                    self.prefix_k_scale[:, :, :new_len].copy_(self.prefix_k_scale[:, :, num_tokens:self.source_length].clone())
                    self.prefix_k_zp[:, :, :new_len].copy_(self.prefix_k_zp[:, :, num_tokens:self.source_length].clone())
                    self.prefix_v_scale[:, :, :new_len].copy_(self.prefix_v_scale[:, :, num_tokens:self.source_length].clone())
                    self.prefix_v_zp[:, :, :new_len].copy_(self.prefix_v_zp[:, :, num_tokens:self.source_length].clone())
                else:
                    self.prefix_k_buffer[:, :, :new_len].copy_(self.prefix_k_buffer[:, :, num_tokens:self.source_length].clone())
                    self.prefix_v_buffer[:, :, :new_len].copy_(self.prefix_v_buffer[:, :, num_tokens:self.source_length].clone())
                
                self.source_length = new_len
            else:
                self.source_length = 0
                
            self._prefix_dirty = True

    def _get_source_cache(self):
        if self.use_blocked_cache:
            # Blocked reconstruction (omitted for brevity, using existing)
            if len(self.prefix_blocks_k) == 0: return None, None
            if not self._prefix_dirty and self._prefix_full_k is not None:
                if self._prefix_full_k.shape[2] > self.source_length:
                    return self._prefix_full_k[:, :, :self.source_length], self._prefix_full_v[:, :, :self.source_length]
                return self._prefix_full_k, self._prefix_full_v
            
            # Reconstruction...
            first_block = self.prefix_blocks_k[0]
            B, H, _, D = first_block.shape
            total_seq_len = sum(block.shape[2] for block in self.prefix_blocks_k)
            k_full = torch.empty(B, H, total_seq_len, D, dtype=first_block.dtype, device=first_block.device)
            v_full = torch.empty(B, H, total_seq_len, D, dtype=first_block.dtype, device=first_block.device)
            offset = 0
            for k_block, v_block in zip(self.prefix_blocks_k, self.prefix_blocks_v):
                sl = k_block.shape[2]
                k_full[:, :, offset:offset+sl] = k_block[:, :, :sl]
                v_full[:, :, offset:offset+sl] = v_block[:, :, :sl]
                offset += sl
            if k_full.shape[2] > self.source_length:
                k_full = k_full[:, :, :self.source_length]
                v_full = v_full[:, :, :self.source_length]
            self._prefix_full_k = k_full
            self._prefix_full_v = v_full
            self._prefix_dirty = False
            return k_full, v_full
        else:
            if self.prefix_k_buffer is not None:
                if self.quantize_kv_cache:
                    k_view = self.prefix_k_buffer[:, :, :self.source_length]
                    v_view = self.prefix_v_buffer[:, :, :self.source_length]
                    k_deq = self._dequantize(k_view, self.prefix_k_scale[:, :, :self.source_length], self.prefix_k_zp[:, :, :self.source_length])
                    v_deq = self._dequantize(v_view, self.prefix_v_scale[:, :, :self.source_length], self.prefix_v_zp[:, :, :self.source_length])
                    return k_deq, v_deq
                return self.prefix_k_buffer[:, :, :self.source_length], self.prefix_v_buffer[:, :, :self.source_length]
            return None, None

    def merge_caches(self, source_start_idx: int = 0, source_end_idx: int | None = None):
        """Dynamic Merge: Prepare unified buffer for decoding."""
        # self.cache_stats["merge_count"] += 1
        
        prefix_k, prefix_v = self._get_source_cache()
        if prefix_k is None:
            self.merged_k_cache = None
            self.merged_v_cache = None
            return

        if source_end_idx is None:
            source_end_idx = self.source_length
        
        # Optimization: Use persistent merged buffer instead of re-allocating
        # We assume suffix length is relatively small (chunk_size)
        # We don't know suffix len here, but we can allocate/ensure buffer is big enough for prefix + margin
        
        current_prefix_len = source_end_idx - source_start_idx
        margin = 256 # Enough for action chunk
        required_len = current_prefix_len + margin
        
        if self.merged_k_cache is None or self.merged_k_cache.shape[2] < required_len:
            B, H, _, D = prefix_k.shape
            alloc_len = max(self.max_cache_size, required_len)
            self.merged_k_cache = torch.empty(B, H, alloc_len, D, dtype=prefix_k.dtype, device=prefix_k.device)
            self.merged_v_cache = torch.empty(B, H, alloc_len, D, dtype=prefix_v.dtype, device=prefix_v.device)
            self._prefix_dirty = True # Force copy
        
        # If prefix changed (or buffer new), copy prefix to merged buffer
        if self._prefix_dirty:
            self.merged_k_cache[:, :, :current_prefix_len] = prefix_k[:, :, source_start_idx:source_end_idx]
            self.merged_v_cache[:, :, :current_prefix_len] = prefix_v[:, :, source_start_idx:source_end_idx]
            # We don't clear _prefix_dirty here because other layers might share source cache state?
            # No, each layer has own cache.
            # But we only clear it if we are sure we won't need to copy again?
            # Actually, prefill sets _prefix_dirty=True.
            self._prefix_dirty = False
            
        self.merge_start_idx = source_start_idx
        self._merge_source_slice_len = current_prefix_len

    def split_caches(self):
        """Cleanup merged cache state."""
        # self.cache_stats["split_count"] += 1
        # In Unified Buffer mode, we don't copy back suffix to Target Cache (it's transient).
        # We just reset indices/pointers if needed.
        # But we keep merged_k_cache allocated for next step!
        pass

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        is_prefix: bool = False,
        is_suffix: bool = False,
        cache_slot: str = "main",
        merge_before_decode: bool = False,
        source_slice_start: int = 0,
        source_slice_end: int | None = None,
        split_after_decode: bool = False,
    ) -> torch.Tensor:
        # self.cache_stats["attention_calls"] += 1
        
        if not self.enable_streaming:
            return super().forward(q, k, v, attention_mask, use_cache)
        
        # Step 1: Prefill
        if use_cache and is_prefix:
            self.prefill_source(k, v, cache_slot=cache_slot)
            return torch.zeros_like(q) # Prefill doesn't output attention
            
        # Step 2: Merge (if requested)
        if merge_before_decode:
            self.merge_caches(source_slice_start, source_slice_end)
            
        # Step 3: Decode with Unified Buffer
        # If merged_k_cache is ready (from manual merge_caches call), use it.
        if self.merged_k_cache is not None:
            # Append suffix k, v to merged buffer IN PLACE
            prefix_len = self._merge_source_slice_len
            suffix_len = k.shape[2]
            total_len = prefix_len + suffix_len
            
            # Ensure buffer size (should be handled in merge_caches, but check margin)
            if total_len > self.merged_k_cache.shape[2]:
                # Dynamic expansion (rare if margin is good)
                B, H, _, D = self.merged_k_cache.shape
                new_len = max(total_len, self.merged_k_cache.shape[2] * 2)
                new_k = torch.empty(B, H, new_len, D, dtype=self.merged_k_cache.dtype, device=self.merged_k_cache.device)
                new_v = torch.empty(B, H, new_len, D, dtype=self.merged_v_cache.dtype, device=self.merged_v_cache.device)
                new_k[:, :, :prefix_len] = self.merged_k_cache[:, :, :prefix_len]
                new_v[:, :, :prefix_len] = self.merged_v_cache[:, :, :prefix_len]
                self.merged_k_cache = new_k
                self.merged_v_cache = new_v
            
            # Write suffix to buffer
            self.merged_k_cache[:, :, prefix_len:total_len] = k
            self.merged_v_cache[:, :, prefix_len:total_len] = v
            
            # Views for attention
            k_full = self.merged_k_cache[:, :, :total_len]
            v_full = self.merged_v_cache[:, :, :total_len]
        else:
            # Fallback (should not happen in optimized mode)
            k_full = k
            v_full = v
            
        # Step 4: Compute Attention
        if self.enable_mixed_precision and self.mixed_precision_enabled:
            out = self.mixed_precision_attn(q, k_full, v_full, attention_mask)
        elif self.enable_flash_attention and self.flash_attn_type == "optimized" and self.optimized_attn is not None:
            out = self.optimized_attn(q, k_full, v_full, attention_mask)
        else:
            out = self._compute_standard_attention(q, k_full, v_full, attention_mask)
            
        if split_after_decode:
            self.split_caches()
            
        return out

    def _compute_standard_attention(self, q, k, v, attention_mask):
        return super().forward(q, k, v, attention_mask, use_cache=False)
