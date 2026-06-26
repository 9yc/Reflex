import torch
import triton
import triton.language as tl

@triton.jit
def _quantize_store_kernel(
    X_ptr,          # Pointer to input tensor (BF16/FP16)
    Cache_ptr,      # Pointer to cache tensor (INT8)
    Scale_ptr,      # Pointer to scale tensor
    ZP_ptr,         # Pointer to zeropoint tensor
    stride_xn, stride_xh, stride_xl, stride_xd, # Strides for X
    stride_cn, stride_ch, stride_cl, stride_cd, # Strides for Cache
    stride_sn, stride_sh, stride_sl, stride_sd, # Strides for Scale
    stride_zn, stride_zh, stride_zl, stride_zd, # Strides for ZP
    l_offset,       # Offset in L dimension for cache writing
    BLOCK_D: tl.constexpr
):
    pid_l = tl.program_id(0) # 0..L_new
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    # Input X: Read from 0..L_new
    x_offset = pid_b * stride_xn + pid_h * stride_xh + pid_l * stride_xl
    offs_d = tl.arange(0, BLOCK_D)
    x_ptrs = X_ptr + x_offset + offs_d * stride_xd
    x = tl.load(x_ptrs)
    
    # Compute quantization stats
    min_val = tl.min(x, axis=0)
    max_val = tl.max(x, axis=0)
    
    range_val = max_val - min_val
    # Prevent div by zero if range is 0 (constant input)
    # If range is 0, scale can be anything (e.g. 1.0), zp = -min
    scale = tl.where(range_val < 1e-6, 1.0, range_val / 255.0)
    zeropoint = -min_val / scale
    
    # Quantize: x_q = clamp(round(x/scale + zp), 0, 255)
    x_q_f = x / scale + zeropoint + 0.5
    x_q_i = tl.cast(x_q_f, tl.int32)
    x_q = tl.clamp(x_q_i, 0, 255)
    x_q_u8 = tl.cast(x_q, tl.uint8)
    
    # Output Cache: Write to (l_offset + pid_l)
    target_l = l_offset + pid_l
    
    cache_offset = pid_b * stride_cn + pid_h * stride_ch + target_l * stride_cl
    cache_ptrs = Cache_ptr + cache_offset + offs_d * stride_cd
    tl.store(cache_ptrs, x_q_u8)
    
    # Output Scale/ZP
    scale_offset = pid_b * stride_sn + pid_h * stride_sh + target_l * stride_sl
    zp_offset = pid_b * stride_zn + pid_h * stride_zh + target_l * stride_zl
    
    tl.store(Scale_ptr + scale_offset, scale)
    tl.store(ZP_ptr + zp_offset, zeropoint)

def fused_quantize_store(x, cache, scale, zp, cache_start_idx=0):
    """
    Quantizes x (B,H,L_new,D) and stores into cache (B,H,L_total,D) at offset.
    """
    B, H, L_new, D = x.shape
    
    # Triton block size constraint
    # We assume D fits in block or pad? 
    # For PI0.5, head_dim is usually 64 or 128.
    # Let's find next power of 2
    BLOCK_D = triton.next_power_of_2(D)
    
    grid = (L_new, H, B)
    
    _quantize_store_kernel[grid](
        x, cache, scale, zp,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        cache.stride(0), cache.stride(1), cache.stride(2), cache.stride(3),
        scale.stride(0), scale.stride(1), scale.stride(2), scale.stride(3),
        zp.stride(0), zp.stride(1), zp.stride(2), zp.stride(3),
        cache_start_idx,
        BLOCK_D=BLOCK_D
    )