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
"""Reflex Configuration Module.

This module provides configuration classes for Reflex inference:
- RunConfig: Inference configuration for real robot deployment
"""

from reflex.configs.run_config import RunConfig
from reflex.policies.pi05 import PI05Config
from reflex.policies.pi0 import PI0Config

# Register Reflex policy configs with LeRobot's config registry.
# This ensures `type: pi05` and `type: pi0` in YAML configs resolve
# to Reflex variants that include vlm_config/action_expert_config.
from lerobot.configs.policies import PreTrainedConfig as _LRPreTrainedConfig

_LRPreTrainedConfig._choice_registry["pi05"] = PI05Config
_LRPreTrainedConfig._choice_registry["pi0"] = PI0Config

__all__ = ["RunConfig", "PI05Config", "PI0Config"]
