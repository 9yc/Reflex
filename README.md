# Reflex

**Reflex** is a streaming inference framework that enables **real-time
control** for flow-matching Vision-Language-Action (VLA) policies.

Flow-matching VLA models promise precise continuous control, but their
iterative denoising loop is fundamentally at odds with real-time robotics:
global timestep injection invalidates KV-caching, forcing a choice between
slow `O(N²)` recomputation and mathematically incorrect cache reuse. Reflex
resolves this by exploiting the **Timestep-Invariance Property** — that
perception encoders are independent of the denoising step — and partitioning
the attention context into static, sliding, and dynamic regions for `O(1)`
incremental cache updates.

This repository contains the **inference / deployment** implementation
targeting the PI0.5 (and PI0) architectures.

## Key ideas

- **Partitioned Attention** — context split into a pinned instruction
  prefix, a sliding observation window, and a dynamic flow suffix; only the
  suffix is recomputed per denoising step, giving `O(1)` cache updates with
  full-batch-equivalent attention outputs.
- **AdaRMSNorm** — precision-aware adaptive normalization that prevents
  BFloat16 collapse under continuous high-frequency (50 Hz) inference.
- **Asynchronous Pipeline** — decouples vision encoding from action
  generation across threads, with future-conditional state prediction and
  adaptive overlap scheduling to hide latency.
- **Operator Fusion** — fused QKV and gate/up projections (custom Triton
  kernels) to cut kernel-launch overhead for single-stream inference.
- **Static Memory** — pre-allocated ring-buffer KV cache and CUDA-graph
  capture for deterministic latency.

See **[docs/architecture.md](docs/architecture.md)** for a precise mapping
from each component to its source file.

## Installation

```bash
git clone https://github.com/9yc/Reflex.git
cd Reflex
pip install -e .
```

Reflex builds on [LeRobot](https://github.com/huggingface/lerobot) and a
pinned `transformers` commit (both declared in `pyproject.toml`). A CUDA GPU
and [Triton](https://github.com/openai/triton) are required for the fused
kernels and `torch.compile`-based async path. For the Reachy 2 robot
backend, install the optional extra: `pip install -e ".[reachy]"`.

## Quick start

Reflex runs a trained policy on a connected robot. Edit
`examples/inference/*.yaml` to point at your robot, cameras, and checkpoint.

### Synchronous control

```bash
reflex run examples/inference/sync.yaml \
    --policy.path=/path/to/pretrained_model \
    --single_task="pick up the red block"
```

### Asynchronous streaming control

Decouples vision and policy onto separate threads and starts the next
action chunk before the current one finishes. `inference_overlap_steps`
controls the lookahead; `compile_model` must be enabled.

```bash
reflex run-streaming examples/inference/async.yaml \
    --policy.path=/path/to/pretrained_model \
    --single_task="pick up the red block" \
    --policy.compile_model=true \
    --inference_overlap_steps=4
```

Relevant policy flags (see `examples/inference/async.yaml`):

| Flag | Meaning |
| --- | --- |
| `policy.enable_streaming` | Enable partitioned/streaming attention |
| `policy.compile_model` | `torch.compile` the policy (required for async) |
| `policy.fuse_qkv` | Fuse Q/K/V projections |
| `policy.fuse_gate_up` | Fuse gate/up projections in SwiGLU MLP |
| `inference_overlap_steps` | Steps before chunk end to launch next inference |

## Project structure

```
reflex/
├── reflex/
│   ├── cli.py                  # `reflex run` / `reflex run-streaming`
│   ├── run.py, run_streaming.py
│   ├── configs/                # RunConfig + policy config registration
│   ├── policies/
│   │   ├── pi05/               # PI0.5 model, AdaRMSNorm, async executor
│   │   └── pi0/                # PI0 model
│   ├── layers/                 # Streaming attention, fused linear, kernels
│   └── optimizations/          # Prefix cache, CUDA graphs, embedding cache
├── examples/inference/         # sync.yaml, async.yaml templates
└── docs/architecture.md        # paper → code map
```

## License

[Apache 2.0](LICENSE)

## Citation

The accompanying paper is included in this repository: [paper.pdf](paper.pdf).

If you find Reflex useful in your research, please cite:

```bibtex
@inproceedings{guo2026reflex,
  title     = {Reflex: Real-Time Vision-Language-Action Control through Streaming Inference},
  author    = {Guo, Yuanchun and Liu, Bingyan},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  year      = {2026}
}
```

