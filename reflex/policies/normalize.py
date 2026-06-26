#!/usr/bin/env python

# Copyright 2025 Reflex team. All rights reserved.
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
"""Normalization Modules for Policy Training.

This module provides nn.Module-based normalization and unnormalization,
ensuring statistics are saved as part of policy checkpoints.

Supported normalization modes:
- MEAN_STD: (x - mean) / std
- MIN_MAX: (x - min) / (max - min) * 2 - 1  -> [-1, 1]
- QUANTILES: (x - q01) / (q99 - q01) * 2 - 1  -> [-1, 1]
- QUANTILE10: (x - q10) / (q90 - q10) * 2 - 1  -> [-1, 1]
- IDENTITY: no normalization

Usage:
    normalize = Normalize(features, norm_map, stats)
    normalized_batch = normalize(batch)
    
    unnormalize = Unnormalize(features, norm_map, stats)
    original_batch = unnormalize(normalized_batch)
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch import Tensor, nn

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.utils.constants import ACTION


def no_stats_error_str(name: str) -> str:
    """Generate error message for missing statistics."""
    return (
        f"`{name}` is infinity. You should either initialize with `stats` as an argument, "
        "or use a pretrained model that already contains normalization statistics."
    )


def create_stats_buffers(
    features: Dict[str, PolicyFeature],
    norm_map: Dict[FeatureType, NormalizationMode],
    stats: Dict[str, Dict[str, Tensor]] | None = None,
) -> Dict[str, nn.ParameterDict]:
    """Create statistics buffers for each feature that needs normalization.
    
    Statistics are stored as nn.Parameters (with requires_grad=False) so they
    are automatically saved/loaded with model checkpoints.
    
    Args:
        features: Dictionary of feature definitions (name -> PolicyFeature).
        norm_map: Mapping from feature type to normalization mode.
        stats: Optional pre-computed statistics from dataset.
        
    Returns:
        Dictionary mapping feature names to ParameterDicts containing
        the relevant statistics (mean/std, min/max, or quantiles).
    """
    stats = stats or {}
    stats_buffers: Dict[str, nn.ParameterDict] = {}

    def to_tensor(data) -> Tensor:
        """Convert various data types to float32 tensor."""
        if isinstance(data, torch.Tensor):
            return data.clone().to(dtype=torch.float32)
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(dtype=torch.float32)
        return torch.as_tensor(data, dtype=torch.float32)

    for key, ft in features.items():
        norm_mode = norm_map.get(ft.type, NormalizationMode.IDENTITY)
        if norm_mode is NormalizationMode.IDENTITY:
            continue  # No buffers needed for identity normalization

        # Determine buffer shape from feature shape
        shape = tuple(ft.shape)
        if ft.type is FeatureType.VISUAL:
            # For images: reduce to per-channel statistics (C, 1, 1)
            assert len(shape) == 3, f"number of dimensions of {key} != 3 ({shape=})"
            c, h, w = shape
            assert c < h and c < w, f"{key} is not channel first ({shape=})"
            shape = (c, 1, 1)

        buffer: dict[str, nn.Parameter] = {}

        # Create buffers based on normalization mode
        # Initialize with inf to detect uninitialized stats
        if norm_mode is NormalizationMode.MEAN_STD:
            mean = torch.full(shape, torch.inf, dtype=torch.float32)
            std = torch.full(shape, torch.inf, dtype=torch.float32)

            if key in stats and "mean" in stats[key] and "std" in stats[key]:
                mean = to_tensor(stats[key]["mean"])
                std = to_tensor(stats[key]["std"])

            buffer["mean"] = nn.Parameter(mean, requires_grad=False)
            buffer["std"] = nn.Parameter(std, requires_grad=False)

        elif norm_mode is NormalizationMode.MIN_MAX:
            min_val = torch.full(shape, torch.inf, dtype=torch.float32)
            max_val = torch.full(shape, torch.inf, dtype=torch.float32)

            if key in stats and "min" in stats[key] and "max" in stats[key]:
                min_val = to_tensor(stats[key]["min"])
                max_val = to_tensor(stats[key]["max"])

            buffer["min"] = nn.Parameter(min_val, requires_grad=False)
            buffer["max"] = nn.Parameter(max_val, requires_grad=False)

        elif norm_mode is NormalizationMode.QUANTILES:
            # 1st and 99th percentiles for robust normalization
            q01 = torch.full(shape, torch.inf, dtype=torch.float32)
            q99 = torch.full(shape, torch.inf, dtype=torch.float32)

            if key in stats and "q01" in stats[key] and "q99" in stats[key]:
                q01 = to_tensor(stats[key]["q01"])
                q99 = to_tensor(stats[key]["q99"])

            buffer["q01"] = nn.Parameter(q01, requires_grad=False)
            buffer["q99"] = nn.Parameter(q99, requires_grad=False)

        elif norm_mode is NormalizationMode.QUANTILE10:
            # 10th and 90th percentiles for more aggressive outlier handling
            q10 = torch.full(shape, torch.inf, dtype=torch.float32)
            q90 = torch.full(shape, torch.inf, dtype=torch.float32)

            if key in stats and "q10" in stats[key] and "q90" in stats[key]:
                q10 = to_tensor(stats[key]["q10"])
                q90 = to_tensor(stats[key]["q90"])

            buffer["q10"] = nn.Parameter(q10, requires_grad=False)
            buffer["q90"] = nn.Parameter(q90, requires_grad=False)

        else:
            raise ValueError(f"Unsupported normalization mode: {norm_mode}")

        stats_buffers[key] = nn.ParameterDict(buffer)

    return stats_buffers


class Normalize(nn.Module):
    """Normalize batch tensors using per-feature statistics.
    
    Each feature is normalized according to its type's normalization mode.
    All normalization maps values to approximately [-1, 1] range.
    
    Attributes:
        features: Feature definitions.
        norm_map: Feature type to normalization mode mapping.
        stats: Original statistics dictionary.
    """

    def __init__(
        self,
        features: Dict[str, PolicyFeature],
        norm_map: Dict[FeatureType, NormalizationMode],
        stats: Dict[str, Dict[str, Tensor]] | None = None,
    ):
        super().__init__()
        self.features = features
        self.norm_map = norm_map
        self.stats = stats

        # Create and register statistics buffers
        stats_buffers = create_stats_buffers(features, norm_map, stats)
        for key, buffer in stats_buffers.items():
            # Replace dots with underscores for valid attribute names
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

    @torch.no_grad()
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Normalize a batch of tensors.
        
        Args:
            batch: Dictionary of tensors to normalize.
            
        Returns:
            New dictionary with normalized tensors.
        """
        batch = dict(batch)  # Shallow copy to avoid mutating input
        eps = 1e-8  # Prevent division by zero

        for key, ft in self.features.items():
            if key not in batch:
                continue

            norm_mode = self.norm_map.get(ft.type, NormalizationMode.IDENTITY)
            if norm_mode is NormalizationMode.IDENTITY:
                continue

            buffer: nn.ParameterDict = getattr(self, "buffer_" + key.replace(".", "_"))

            if norm_mode is NormalizationMode.MEAN_STD:
                mean = buffer["mean"]
                std = buffer["std"]
                batch[key] = (batch[key] - mean) / (std + eps)

            elif norm_mode is NormalizationMode.MIN_MAX:
                min_val = buffer["min"]
                max_val = buffer["max"]
                # Map to [0, 1] then to [-1, 1]
                batch[key] = (batch[key] - min_val) / (max_val - min_val + eps)
                batch[key] = batch[key] * 2 - 1

            elif norm_mode is NormalizationMode.QUANTILES:
                q01 = buffer["q01"]
                q99 = buffer["q99"]
                denom = q99 - q01
                denom = denom.masked_fill(denom == 0, eps)
                # Map [q01, q99] -> [-1, 1]
                batch[key] = 2.0 * (batch[key] - q01) / denom - 1.0

            elif norm_mode is NormalizationMode.QUANTILE10:
                q10 = buffer["q10"]
                q90 = buffer["q90"]
                denom = q90 - q10
                denom = denom.masked_fill(denom == 0, eps)
                # Map [q10, q90] -> [-1, 1]
                batch[key] = 2.0 * (batch[key] - q10) / denom - 1.0

            else:
                raise ValueError(f"Unsupported normalization mode: {norm_mode}")

        return batch


class Unnormalize(nn.Module):
    """Inverse of Normalize - converts normalized values back to original scale.
    
    Attributes:
        features: Feature definitions.
        norm_map: Feature type to normalization mode mapping.
        stats: Original statistics dictionary.
    """

    def __init__(
        self,
        features: Dict[str, PolicyFeature],
        norm_map: Dict[FeatureType, NormalizationMode],
        stats: Dict[str, Dict[str, Tensor]] | None = None,
    ):
        super().__init__()
        self.features = features
        self.norm_map = norm_map
        self.stats = stats

        # Create and register statistics buffers (same as Normalize)
        stats_buffers = create_stats_buffers(features, norm_map, stats)
        for key, buffer in stats_buffers.items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

    @torch.no_grad()
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Unnormalize a batch of tensors.
        
        Args:
            batch: Dictionary of normalized tensors.
            
        Returns:
            Dictionary with tensors in original scale.
        """
        eps = 1e-8

        for key, ft in self.features.items():
            if key not in batch:
                continue

            norm_mode = self.norm_map.get(ft.type, NormalizationMode.IDENTITY)
            if norm_mode is NormalizationMode.IDENTITY:
                continue

            buffer: nn.ParameterDict = getattr(self, "buffer_" + key.replace(".", "_"))

            if norm_mode is NormalizationMode.MEAN_STD:
                mean = buffer["mean"]
                std = buffer["std"]
                batch[key] = batch[key] * std + mean

            elif norm_mode is NormalizationMode.MIN_MAX:
                min_val = buffer["min"]
                max_val = buffer["max"]
                # Map [-1, 1] -> [0, 1] -> [min, max]
                batch[key] = (batch[key] + 1.0) / 2.0
                batch[key] = batch[key] * (max_val - min_val) + min_val

            elif norm_mode is NormalizationMode.QUANTILES:
                q01 = buffer["q01"]
                q99 = buffer["q99"]
                denom = q99 - q01
                denom = denom.masked_fill(denom == 0, eps)
                # Map [-1, 1] -> [q01, q99]
                batch[key] = (batch[key] + 1.0) * denom / 2.0 + q01

            elif norm_mode is NormalizationMode.QUANTILE10:
                q10 = buffer["q10"]
                q90 = buffer["q90"]
                denom = q90 - q10
                denom = denom.masked_fill(denom == 0, eps)
                # Map [-1, 1] -> [q10, q90]
                batch[key] = (batch[key] + 1.0) * denom / 2.0 + q10

            else:
                raise ValueError(f"Unsupported normalization mode: {norm_mode}")

        return batch


__all__ = ["Normalize", "Unnormalize"]
