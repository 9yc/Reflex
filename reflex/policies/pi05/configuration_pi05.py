#!/usr/bin/env python

# Copyright 2025 Reflex team. All rights reserved.
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
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
"""PI0.5 Policy Configuration.

This module defines the configuration for the PI0.5 (π0.5) Vision-Language-Action
model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

from transformers.models.gemma.configuration_gemma import GemmaConfig
from transformers.models.paligemma.configuration_paligemma import PaliGemmaConfig


@dataclass
class PI05VLMConfig(PaliGemmaConfig):
    """Configuration for the PaliGemma vision-language backbone.
    
    This configures the multimodal encoder that processes images and
    text prompts to produce embeddings for action generation.
    """
    
    def __init__(self):
        super().__init__()
        # Vocabulary configuration
        self._vocab_size = 257152
        self.image_token_index = 257152
        
        # Text encoder (Gemma) configuration
        self.text_config.hidden_size = 2048
        self.text_config.intermediate_size = 16_384
        self.text_config.num_attention_heads = 8
        self.text_config.head_dim = 256
        self.text_config.num_hidden_layers = 18
        self.text_config.num_key_value_heads = 1
        self.text_config.hidden_activation = "gelu_pytorch_tanh"
        self.text_config.torch_dtype = "float32"
        self.text_config.vocab_size = 257152
        self.text_config.use_adarms = False
        self.text_config.adarms_cond_dim = None
        
        # Vision encoder (SigLIP) configuration
        self.vision_config.intermediate_size = 4304
        self.vision_config.projection_dim = 2048
        self.vision_config.projector_hidden_act = "gelu_fast"
        self.vision_config.torch_dtype = "float32"


@dataclass
class PI05ActionExpertConfig(GemmaConfig):
    """Configuration for the action expert network.
    
    The action expert is a smaller Gemma model that takes VLM embeddings
    and generates action predictions via flow matching.
    """
    
    def __init__(self):
        super().__init__(
            head_dim=256,
            hidden_size=1024,
            intermediate_size=4096,
            num_attention_heads=8,
            num_hidden_layers=18,
            num_key_value_heads=1,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=True,  # Adaptive RMS normalization for conditioning
            adarms_cond_dim=None,
        )


@dataclass
class PI05Config(PreTrainedConfig):
    """Main configuration for PI0.5 policy.
    

    """

    # === Model Architecture ===
    paligemma_variant: str = "gemma_2b"  # VLM backbone: "gemma_300m" or "gemma_2b"
    action_expert_variant: str = "gemma_300m"  # Action expert: "gemma_300m" or "gemma_2b"
    dtype: str = "bfloat16"  # Compute dtype: "bfloat16" or "float32"

    # === Action Prediction ===
    n_obs_steps: int = 1  # Number of observation frames to use
    chunk_size: int = 50  # Number of actions to predict per inference
    n_action_steps: int = 50  # Number of actions to execute before re-inference

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # State conditioning: use robot state in adarmsnorm
    # This significantly improves stability and performance for PI0.5
    state_cond: bool = False

    # === Flow Matching Parameters ===
    num_inference_steps: int = 10  # Denoising steps during inference
    time_sampling_beta_alpha: float = 1.5  # Beta distribution alpha for time sampling
    time_sampling_beta_beta: float = 1.0  # Beta distribution beta for time sampling
    time_sampling_scale: float = 0.999  # Time scaling factor
    time_sampling_offset: float = 0.001  # Minimum time value
    min_period: float = 4e-3  # Minimum sinusoidal embedding period
    max_period: float = 4.0  # Maximum sinusoidal embedding period

    # === Image Processing ===
    image_resolution: tuple[int, int] = (224, 224)  # Input image size
    empty_cameras: int = 0  # Number of placeholder cameras to add

    # === Tokenization ===
    tokenizer_max_length: int = 200  # Maximum prompt token length

    # Tokenizer location override (useful for offline deployment).
    # - If None: defaults to upstream "google/paligemma-3b-pt-224"
    # - If set to a local directory: will load tokenizer files from disk.
    tokenizer_name_or_path: str | None = None
    tokenizer_path: str | None = None

    # === Normalization ===
    # Training with MEAN_STD will be a lot smoother and more stable than QUANTILES
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,  # Images: no normalization
            "STATE": NormalizationMode.MEAN_STD,   # States: z-score normalization
            "ACTION": NormalizationMode.MEAN_STD,  # Actions: z-score normalization
        }
    )

    # === Training Settings ===
    gradient_checkpointing: bool = False  # Trade compute for memory
    compile_model: bool = False  # Use torch.compile optimization
    compile_mode: str = "max-autotune"  # Torch compile mode
    device: str | None = None  # Target device (None = auto-detect)

    # Attention/MLP fusion (for inference optimization)
    fuse_qkv: bool = False  # Fuse Q/K/V projections
    fuse_gate_up: bool = False  # Fuse gate/up projections in MLP

    # === Streaming VLA Optimization ===
    # Single switch to enable all streaming optimizations
    # This is the main user-facing parameter - everything else is internal
    enable_streaming: bool = False  # Enable streaming VLA optimizations

    # Enable concurrent prefill/decode execution (threads/CUDA streams).
    # Default off to avoid OOM in constrained GPUs; enable explicitly for performance experiments.
    enable_concurrent_execution: bool = False

    # Enable block-based KV cache (PagedAttention-style) in StreamingAttention.
    # This is a no-training optimization that can reduce cache growth overhead and
    # improve compilation stability by using fixed-size blocks.
    enable_blocked_kv_cache: bool = False

    # Fixed block size for blocked KV cache.
    kv_block_size: int = 16

    # Optional streaming attention window (tokens). None means "no window limit".
    # Note: this is used by streaming attention mask builder.
    streaming_window: int | None = None
    
    # === Phase 4C Advanced Optimizations ===
    # Mixed precision optimization (Phase 4C.1)
    enable_mixed_precision: bool = True  # Use FP16/FP32 mixed precision for attention
    
    # Attention mask optimization (Phase 4C.2) 
    enable_mask_caching: bool = True  # Cache attention masks across steps
    
    # Compilation optimization (Phase 4C.3)
    enable_advanced_compilation: bool = False  # Advanced torch.compile settings
    
    # Memory layout optimization (Phase 4C.4)
    enable_memory_optimization: bool = True  # Optimize tensor memory layouts
    
    # Internal parameters (not exposed to users, use validated defaults)
    # These are implementation details documented in the paper's ablation study
    _streaming_config: dict = field(default_factory=lambda: {
        # Parallel KV Cache: separate source/target caches for concurrent execution
        "parallel_kv_cache": True,
        # Batched camera processing: process multiple cameras in one forward pass
        "batched_cameras": True,
        # Embedding cache: reuse embeddings for similar consecutive frames
        "embedding_cache": True,
        "embedding_cache_threshold": 0.98,  # Validated: 71% hit rate, 1.9% error
        # Concurrent execution settings
        "use_cuda_streams": True,
        # Cache settings
        "max_history_length": None,  # Unlimited by default
        "kv_block_size": 16,
        "use_blocked_cache": False,
    })

    # === Optimizer Settings ===
    optimizer_lr: float = 2.5e-5  # Peak learning rate
    optimizer_betas: tuple[float, float] = (0.9, 0.95)  # Adam betas
    optimizer_eps: float = 1e-8  # Adam epsilon
    optimizer_weight_decay: float = 0.01  # Weight decay
    optimizer_grad_clip_norm: float = 1.0  # Gradient clipping threshold

    # === Scheduler Settings ===
    scheduler_warmup_steps: int = 1_000  # Linear warmup steps
    scheduler_decay_steps: int = 30_000  # Cosine decay steps
    scheduler_decay_lr: float = 2.5e-6  # Final learning rate

    tokenizer_max_length: int = 200

    # === Sub-model Configurations ===
    vlm_config: PI05VLMConfig = field(default_factory=PI05VLMConfig)
    action_expert_config: PI05ActionExpertConfig = field(default_factory=PI05ActionExpertConfig)

    def __post_init__(self):
        """Validate configuration after initialization."""
        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

    def validate_features(self) -> None:
        """Validate and set up input/output features.
        
        Adds placeholder cameras and ensures state/action features are defined.
        """
        # Add empty camera placeholders if configured
        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),
            )
            self.input_features[key] = empty_camera

        # Ensure state feature exists
        if "observation.state" not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )
            self.input_features["observation.state"] = state_feature

        # Ensure action feature exists
        if "action" not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
            )
            self.output_features["action"] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        """Get optimizer configuration from policy settings."""
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        """Get learning rate scheduler configuration from policy settings."""
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        """Observation frame indices relative to current timestep."""
        return None

    @property
    def action_delta_indices(self) -> list:
        """Action frame indices to predict (0 to chunk_size-1)."""
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        """Reward indices (not used in PI0.5)."""
        return None

    # === Streaming Config Accessors (internal use) ===
    
    @property
    def use_parallel_kv_cache(self) -> bool:
        """Whether to use parallel KV caches (source/target separation)."""
        return self.enable_streaming and self._streaming_config.get("parallel_kv_cache", True)
    
    @property
    def use_batched_cameras(self) -> bool:
        """Whether to batch multiple camera images for processing."""
        return self.enable_streaming and self._streaming_config.get("batched_cameras", True)
    
    @property
    def use_embedding_cache(self) -> bool:
        """Whether to cache image embeddings for similar frames."""
        return self.enable_streaming and self._streaming_config.get("embedding_cache", True)
    
    @property
    def embedding_cache_threshold(self) -> float:
        """Similarity threshold for embedding cache hits."""
        return self._streaming_config.get("embedding_cache_threshold", 0.98)
    
    @property
    def use_cuda_streams(self) -> bool:
        """Whether to use CUDA streams for concurrent execution."""
        return self.enable_streaming and self._streaming_config.get("use_cuda_streams", True)
    
    @property
    def streaming_max_history_length(self) -> int | None:
        """Maximum history length for streaming cache."""
        return self._streaming_config.get("max_history_length", None)
    
    @property
    def kv_cache_block_size(self) -> int:
        """Block size for KV cache."""
        # Prefer explicit top-level knob (more user-friendly than editing _streaming_config dict).
        # Fall back to internal validated defaults.
        return int(getattr(self, "kv_block_size", self._streaming_config.get("kv_block_size", 16)))
    
    @property
    def use_blocked_cache(self) -> bool:
        """Whether to use block-based KV cache (PagedAttention style)."""
        return bool(
            self.enable_streaming
            and (
                getattr(self, "enable_blocked_kv_cache", False)
                or self._streaming_config.get("use_blocked_cache", False)
            )
        )


__all__ = ["PI05Config"]
