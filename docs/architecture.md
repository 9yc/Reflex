# Reflex Architecture: Paper → Code Map

This document maps the five technical components described in the Reflex
paper to their implementation in the source tree. File references use
`path:line` and point at the class/function that implements each idea.

Reflex enables **real-time streaming inference** for flow-matching
Vision-Language-Action (VLA) policies by exploiting the
*Timestep-Invariance Property*: the perception encoder is functionally
independent of the flow-matching denoising step. The implementation here
targets the PI0.5 architecture (`reflex/policies/pi05/`); PI0 is also
included (`reflex/policies/pi0/`).

---

## 1. Partitioned Attention (Stream Correctness)

> Context is split into a pinned instruction prefix, a sliding observation
> window (FIFO), and a dynamic flow-generation suffix. Only the dynamic
> suffix is recomputed per denoising step, giving `O(1)` cache updates with
> full-batch-equivalent outputs.

| Concept | Location |
| --- | --- |
| Streaming attention layer | `reflex/layers/streaming_attention.py:91` — `class StreamingAttention(Attention)` |
| Base KV-cache attention | `reflex/layers/attention.py:38` — `class Attention` |
| Incremental prefill (encode only newest frame) | `reflex/layers/streaming_attention.py:267` — `prefill_source(...)` |
| Manual cache merging (pre-merge prefix + history into contiguous buffer) | `reflex/layers/streaming_attention.py:497` — `merge_caches(...)` |
| Incremental KV-cache updater | `reflex/layers/dynamic_embedding.py` — `IncrementalKVCacheUpdater` |
| Streaming input manager (FIFO sliding window) | `reflex/policies/pi05/streaming_manager.py` — `StreamingInputManager` |

The `StreamingAttention` layer is selected inside the policy at
`reflex/policies/pi05/modeling_pi05.py:392`.

---

## 2. AdaRMSNorm (Stream Stability)

> A precision-aware adaptive normalization that gates on flow phase to
> prevent BFloat16 numerical collapse at 50 Hz operation. Variance is
> computed in FP32; the gating MLP runs in the backbone's compute dtype,
> detected via *Robust Dtype Inference* (probing `o_proj`).

| Concept | Location |
| --- | --- |
| Adaptive RMSNorm operator | `reflex/policies/pi05/modeling_pi05.py:50` — `class AdaRMSNorm` |
| FP32 variance / gated modulation | `reflex/policies/pi05/modeling_pi05.py:63` — `AdaRMSNorm.forward` |
| Injection into VLM + action-expert norms | `reflex/policies/pi05/modeling_pi05.py:748` |
| Robust dtype inference (probe `o_proj.weight.dtype`) | `reflex/policies/pi05/modeling_pi05.py:1002`, `:1200`, `:1270` |

---

## 3. Asynchronous Pipeline (Stream Throughput)

> Vision encoding (producer) and action generation (consumer) run on
> separate threads. *Future-Conditional State Prediction* compensates for
> the async delay by conditioning on the last commanded action instead of
> the stale sensor reading. *Adaptive Overlap Scheduling* tunes the
> lookahead `K`.

| Concept | Location |
| --- | --- |
| Concurrent producer/consumer executor | `reflex/policies/pi05/concurrent_executor.py:26` — `class ConcurrentStreamingExecutor` |
| Thread-safe streaming manager | `reflex/policies/pi05/concurrent_executor.py:192` — `class ThreadSafeStreamingManager` |
| Streaming control loop | `reflex/run_streaming.py:178` — `run_loop_streaming(...)` |
| Future-state substitution (`ŝ_{t+Δ} ≈ a_t^cmd`) | `reflex/run_streaming.py:120`, `:133` |
| Adaptive overlap parameter | `inference_overlap_steps` — `reflex/run_streaming.py:187`; validated in `reflex/configs/run_config.py:130` |

---

## 4. Operator Fusion (System Optimization)

> Fuse Q/K/V projections into a single packed kernel and the Gate/Up
> projections in SwiGLU blocks, halving kernel launches per layer.

| Concept | Location |
| --- | --- |
| Fused QKV projection | `reflex/layers/linear.py:29` — `class QKVLinear` |
| Fused gate/up (SwiGLU) projection | `reflex/layers/linear.py:103` — `class MergedColumnLinear` |
| Fused quantize/store Triton kernel | `reflex/layers/quantize_kernel.py` — `fused_quantize_store` |
| Config toggles | `reflex/policies/pi05/configuration_pi05.py:158` — `fuse_qkv`, `fuse_gate_up` |

---

## 5. Ring Buffer & Static Memory (System Optimization)

> Pre-allocate a monolithic KV tensor and index it with circular pointer
> arithmetic to eliminate dynamic allocation and GC latency spikes during
> the 24/7 control loop.

| Concept | Location |
| --- | --- |
| Rolling/ring-buffer cache handling | `reflex/layers/streaming_attention.py:314` (ring-buffer overwrite path in `prefill_source`) |
| Pre-allocated prefix KV cache | `reflex/optimizations/single_frame_optimizer.py:75` — `class PrefixKVCache` |
| CUDA graph capture (static addresses) | `reflex/optimizations/single_frame_optimizer.py:157` — `class CUDAGraphWrapper` |

---

## Supporting: Similarity-Based Embedding Cache

Not a core paper claim but referenced as a throughput optimization: skips
re-encoding visually static frames.

| Concept | Location |
| --- | --- |
| Single-frame optimizer | `reflex/optimizations/single_frame_optimizer.py:299` — `class SingleFrameOptimizer` |
| Dynamic embedding cache | `reflex/optimizations/dynamic_embedding_cache.py` — `DynamicEmbeddingCache`, `MultiCameraEmbeddingCache` |
| Adaptive denoising | `reflex/optimizations/single_frame_optimizer.py:238` — `class AdaptiveDenoising` |

---

## Entry Points

| Command | Function |
| --- | --- |
| `reflex run <cfg>` | `reflex/run.py:471` — `run(cfg)` (synchronous control loop) |
| `reflex run-streaming <cfg>` | `reflex/run_streaming.py:294` — `run_streaming(cfg)` (async streaming loop) |
| Policy factory | `reflex/policies/factory.py:47` — `get_policy_class("pi05" | "pi0")` |

---

## Naming map & implementation notes

The code predates the paper's final terminology and reuses some names from
prior work. The mechanisms are the same; only the vocabulary differs. Nothing
below is a semantic deviation from the paper.

### Partitioned attention: code names ↔ paper regions

`StreamingAttention` (`reflex/layers/streaming_attention.py`) realizes the
paper's three context regions with two persistent caches plus a transient
merged buffer. The "Source/Target Cache" naming is inherited from the
StreamingThinker parallel-KV-cache idea:

| Paper region | Code term / attribute |
| --- | --- |
| Pinned instruction prefix + Sliding observation window | **Source Cache** — `prefix_k_buffer` / `prefix_v_buffer`, filled by `prefill_source(...)`; FIFO eviction of oldest observations happens upstream in `reflex/run_streaming.py:127` |
| Dynamic flow/action suffix (reset each denoising cycle) | **Target Cache** — `suffix_k_cache` / `suffix_v_cache` |
| Contiguous prefix+suffix scratch for the denoising loop | **Merged Cache** — `merged_k_cache` / `merged_v_cache`, built once by `merge_caches(...)` (Manual Cache Merging, avoids per-step `torch.cat`) |

### AdaRMSNorm: `gamma(c) = 1 + MLP(c)`

The paper writes the adaptive scale as `gamma(c) = 1 + MLP(c)`. In code
(`reflex/policies/pi05/modeling_pi05.py:50`), `MLP` is the affine layer
`self.dense` (a `nn.Linear` *with bias*), and the gate is `dense(c) = W·c + b`.
Since `1 + (W·c + b') = W·c + (1 + b')`, the paper's constant `1` is absorbed
into `dense.bias`; the two forms are mathematically equivalent. The
implementation is kept in this form to stay aligned with the pretrained PI0.5
`dense.weight` / `dense.bias` checkpoint keys — adding an explicit `+1` would
double-count the constant and corrupt loaded weights.

Additionally, the returned `gate` is reused for the layer's **gated residual**
connection (`_gated_residual = residual + gate * sublayer`, used in
`PI05ModelLayer.forward` at `:588`/`:609`), i.e. an adaLN-Zero–style
conditioning that is slightly more expressive than the bare normalization
formula in the paper.

### Ring buffer / static memory

Statically pre-allocated KV buffers (`torch.zeros(B, H, buffer_size, D)`) are
used on the quantized / blocked-cache path
(`reflex/layers/streaming_attention.py:254`). On the default path, the
"zero dynamic allocation" property comes from `merge_caches` reusing a
persistent `merged_*_cache` buffer (allocated once with a 256-token margin)
and writing the suffix in place during the denoising loop.
