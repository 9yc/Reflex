#!/usr/bin/env python
"""
Mixed Precision Attention Implementation

This module provides mixed precision optimizations for Reflex streaming attention.
It uses FP16 for non-critical computations while maintaining FP32 for numerical
stability in critical operations like softmax.

Expected improvement: 2-5% for Reflex streaming workloads
"""

import torch
import torch.nn.functional as F
from typing import Optional
import math

class MixedPrecisionAttention(torch.nn.Module):
    """
    Mixed precision attention module for Reflex streaming inference.
    
    This implementation uses:
    - FP16 for attention score computation (memory bandwidth reduction)
    - FP32 for softmax computation (numerical stability)
    - FP16 for output computation (performance)
    - Automatic fallback to FP32 for edge cases
    """
    
    def __init__(self, scale: Optional[float] = None, enable_amp: bool = True):
        super().__init__()
        self.scale = scale
        self.enable_amp = enable_amp
        
        # Performance tracking
        self.stats = {
            'forward_calls': 0,
            'fp16_calls': 0,
            'fp32_fallback_calls': 0,
            'total_time': 0.0,
            'amp_time': 0.0,
            'fallback_time': 0.0
        }
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with mixed precision optimization."""
        import time
        start_time = time.time()
        
        self.stats['forward_calls'] += 1
        
        if self.enable_amp and torch.cuda.is_available():
            try:
                result = self._mixed_precision_attention(q, k, v, attention_mask)
                self.stats['fp16_calls'] += 1
                self.stats['amp_time'] += (time.time() - start_time)
                return result
            except Exception as e:
                # Fallback to FP32 if mixed precision fails
                result = self._fp32_attention(q, k, v, attention_mask)
                self.stats['fp32_fallback_calls'] += 1
                self.stats['fallback_time'] += (time.time() - start_time)
                return result
        else:
            result = self._fp32_attention(q, k, v, attention_mask)
            self.stats['fp32_fallback_calls'] += 1
            self.stats['fallback_time'] += (time.time() - start_time)
            return result
    
    def _mixed_precision_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Mixed precision attention computation."""
        if self.scale is None:
            scale = 1.0 / math.sqrt(q.shape[-1])
        else:
            scale = self.scale
        
        # Store original dtypes
        orig_q_dtype = q.dtype
        orig_k_dtype = k.dtype
        orig_v_dtype = v.dtype
        
        # Use autocast for automatic mixed precision
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            # Convert to FP16 for attention score computation
            q_fp16 = q.to(torch.float16)
            k_fp16 = k.to(torch.float16)
            v_fp16 = v.to(torch.float16)
            
            # Compute attention scores in FP16 (memory bandwidth optimization)
            attn_scores = torch.matmul(q_fp16, k_fp16.transpose(-2, -1)) * scale
            
            # Apply attention mask in FP16
            if attention_mask is not None:
                mask = attention_mask
                if mask.dtype != torch.float16:
                    mask = mask.to(torch.float16)
                
                # Handle mask dimensions
                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)
                
                # Ensure mask matches attention scores shape
                B, H, L_q, L_k = attn_scores.shape
                mask_B, mask_H, mask_L_q, mask_L_k = mask.shape
                
                # Adjust mask dimensions if needed
                if mask_L_k != L_k:
                    if mask_L_k < L_k:
                        pad_size = L_k - mask_L_k
                        mask = F.pad(mask, (0, pad_size), value=torch.finfo(torch.float16).min)
                    else:
                        mask = mask[:, :, :, :L_k]
                
                if mask_L_q != L_q:
                    if mask_L_q < L_q:
                        pad_size = L_q - mask_L_q
                        mask = F.pad(mask, (0, 0, 0, pad_size), value=torch.finfo(torch.float16).min)
                    else:
                        mask = mask[:, :, :L_q, :]
                
                # Expand head dimension if needed
                if mask.shape[1] == 1 and H != 1:
                    mask = mask.expand(-1, H, -1, -1)
                
                # Apply mask with numerical stability
                mask_value = torch.finfo(torch.float16).min
                attn_scores = torch.where(
                    mask == 0,
                    attn_scores,
                    torch.full_like(attn_scores, mask_value)
                )
                
                # Handle fully masked rows
                row_all_masked = (mask != 0).all(dim=-1)  # [B, H, L_q]
                if row_all_masked.any():
                    # Set first position to 0 for fully masked rows
                    first_pos_mask = torch.zeros_like(mask, dtype=torch.bool)
                    first_pos_mask[:, :, :, 0] = True
                    attn_scores = torch.where(
                        row_all_masked.unsqueeze(-1) & first_pos_mask,
                        torch.zeros_like(attn_scores),
                        attn_scores
                    )
        
        # CRITICAL: Softmax computation in FP32 for numerical stability
        with torch.cuda.amp.autocast(enabled=False):
            attn_scores_fp32 = attn_scores.to(torch.float32)
            attn_weights = F.softmax(attn_scores_fp32, dim=-1, dtype=torch.float32)
            
            # Handle any remaining NaNs
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Output computation in FP16 for performance
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            attn_weights_fp16 = attn_weights.to(torch.float16)
            output = torch.matmul(attn_weights_fp16, v_fp16)
        
        # Convert back to original dtype
        output = output.to(orig_q_dtype)
        
        return output
    
    def _fp32_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Standard FP32 attention computation (fallback)."""
        if self.scale is None:
            scale = 1.0 / math.sqrt(q.shape[-1])
        else:
            scale = self.scale
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Apply attention mask
        if attention_mask is not None:
            mask = attention_mask
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            
            # Ensure mask matches attention scores shape
            B, H, L_q, L_k = attn_scores.shape
            mask_B, mask_H, mask_L_q, mask_L_k = mask.shape
            
            # Adjust mask dimensions if needed
            if mask_L_k != L_k:
                if mask_L_k < L_k:
                    pad_size = L_k - mask_L_k
                    mask = F.pad(mask, (0, pad_size), value=float('-inf'))
                else:
                    mask = mask[:, :, :, :L_k]
            
            if mask_L_q != L_q:
                if mask_L_q < L_q:
                    pad_size = L_q - mask_L_q
                    mask = F.pad(mask, (0, 0, 0, pad_size), value=float('-inf'))
                else:
                    mask = mask[:, :, :L_q, :]
            
            # Expand head dimension if needed
            if mask.shape[1] == 1 and H != 1:
                mask = mask.expand(-1, H, -1, -1)
            
            # Apply mask with numerical stability
            mask_value = torch.finfo(attn_scores.dtype).min
            attn_scores = torch.where(
                mask == 0,
                attn_scores,
                torch.full_like(attn_scores, mask_value)
            )
            
            # Handle fully masked rows
            row_all_masked = (mask != 0).all(dim=-1)  # [B, H, L_q]
            if row_all_masked.any():
                # Set first position to 0 for fully masked rows
                first_pos_mask = torch.zeros_like(mask, dtype=torch.bool)
                first_pos_mask[:, :, :, 0] = True
                attn_scores = torch.where(
                    row_all_masked.unsqueeze(-1) & first_pos_mask,
                    torch.zeros_like(attn_scores),
                    attn_scores
                )
        
        # Softmax and output computation
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)
        output = torch.matmul(attn_weights, v)
        
        return output
    
    def get_stats(self) -> dict:
        """Get performance statistics."""
        stats = self.stats.copy()
        if stats['forward_calls'] > 0:
            stats['avg_time_ms'] = (stats['total_time'] / stats['forward_calls']) * 1000
            stats['fp16_ratio'] = stats['fp16_calls'] / stats['forward_calls']
            stats['fallback_ratio'] = stats['fp32_fallback_calls'] / stats['forward_calls']
            
            if stats['fp16_calls'] > 0:
                stats['avg_amp_time_ms'] = (stats['amp_time'] / stats['fp16_calls']) * 1000
            if stats['fp32_fallback_calls'] > 0:
                stats['avg_fallback_time_ms'] = (stats['fallback_time'] / stats['fp32_fallback_calls']) * 1000
        
        return stats
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.stats = {
            'forward_calls': 0,
            'fp16_calls': 0,
            'fp32_fallback_calls': 0,
            'total_time': 0.0,
            'amp_time': 0.0,
            'fallback_time': 0.0
        }

def benchmark_mixed_precision_attention():
    """Benchmark the mixed precision attention implementation."""
    print("🔬 Mixed Precision Attention Benchmark")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cpu":
        print("⚠️ CUDA not available, mixed precision benefits limited on CPU")
        return
    
    # Test configurations typical for Reflex
    configs = [
        {"seq_len_q": 32, "seq_len_k": 256, "heads": 32, "head_dim": 128, "batch": 1},
        {"seq_len_q": 64, "seq_len_k": 512, "heads": 32, "head_dim": 128, "batch": 1},
        {"seq_len_q": 128, "seq_len_k": 1024, "heads": 32, "head_dim": 128, "batch": 1},
    ]
    
    for config in configs:
        print(f"\nTesting: q_len={config['seq_len_q']}, k_len={config['seq_len_k']}, heads={config['heads']}")
        
        # Create test tensors
        q = torch.randn(config['batch'], config['heads'], config['seq_len_q'], config['head_dim'], device=device)
        k = torch.randn(config['batch'], config['heads'], config['seq_len_k'], config['head_dim'], device=device)
        v = torch.randn(config['batch'], config['heads'], config['seq_len_k'], config['head_dim'], device=device)
        
        scale = 1.0 / math.sqrt(config['head_dim'])
        
        # Standard FP32 attention
        mixed_attn = MixedPrecisionAttention(scale=scale, enable_amp=False)
        start_time = time.time()
        fp32_out = mixed_attn(q, k, v)
        torch.cuda.synchronize()
        fp32_time = time.time() - start_time
        
        # Mixed precision attention
        mixed_attn = MixedPrecisionAttention(scale=scale, enable_amp=True)
        start_time = time.time()
        mp_out = mixed_attn(q, k, v)
        torch.cuda.synchronize()
        mp_time = time.time() - start_time
        
        # Check correctness
        max_diff = torch.max(torch.abs(fp32_out - mp_out)).item()
        
        # Results
        speedup = fp32_time / mp_time if mp_time > 0 else 1.0
        print(f"  FP32: {fp32_time*1000:.2f}ms")
        print(f"  Mixed Precision: {mp_time*1000:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Max diff: {max_diff:.2e}")
        print(f"  Correct: {'✅' if max_diff < 1e-3 else '❌'}")
        
        # Get statistics
        stats = mixed_attn.get_stats()
        print(f"  FP16 ratio: {stats.get('fp16_ratio', 0):.1%}")
        print(f"  Fallback ratio: {stats.get('fallback_ratio', 0):.1%}")

if __name__ == "__main__":
    import time
    benchmark_mixed_precision_attention()