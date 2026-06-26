#!/usr/bin/env python

# Copyright 2025 Reflex team. All rights reserved.
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Reflex Runtime/Inference Configuration.
"""

from dataclasses import dataclass
from typing import Any, Union

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.robots import RobotConfig

# Import robot configs to register them with draccus parser
# This enables YAML configs like `robot: {type: so100_follower, ...}`
from lerobot.robots.bi_so100_follower import BiSO100FollowerConfig  # noqa: F401
from lerobot.robots.hope_jr import HopeJrArmConfig, HopeJrHandConfig  # noqa: F401
from lerobot.robots.koch_follower import KochFollowerConfig  # noqa: F401
from lerobot.robots.lekiwi import LeKiwiClientConfig, LeKiwiConfig  # noqa: F401
from lerobot.robots.reachy2 import Reachy2RobotConfig  # noqa: F401
from lerobot.robots.so100_follower import SO100FollowerConfig  # noqa: F401
from lerobot.robots.so101_follower import SO101FollowerConfig  # noqa: F401


@dataclass
class RunConfig:
    """Configuration for running a policy on a real robot.
    
    This is an inference-only config (no dataset recording). It supports
    Reflex's asynchronous inference with overlapping action chunks.
    
    Async inference timeline (n_action_steps=16, inference_overlap_steps=6):
        Chunk 1: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
                                      ↑ Start inference for Chunk 2
        Chunk 2:                      [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    """

    # Robot configuration (required)
    robot: RobotConfig
    
    # Policy configuration
    # Via CLI: --policy.path=xxx --policy.device=cuda
    # Or YAML: policy: {path: xxx, device: cuda}
    policy: Union[PreTrainedConfig, dict[str, Any], None] = None

    # Control parameters
    fps: int = 30  # Control frequency (Hz)
    control_time_s: Union[int, float] = 600  # Total runtime (seconds)

    # Visualization and feedback
    display_data: bool = False  # Show camera feeds via rerun
    play_sounds: bool = True  # Audio feedback for events

    # Action quantization
    action_quant_ratio: int = 1

    # Async inference overlap: start next chunk inference N steps before current ends
    # Set to 0 for synchronous inference (no overlap)
    inference_overlap_steps: int = 0

    # Task description passed to policy
    single_task: Union[str, None] = None

    def __post_init__(self):
        """Parse policy config and validate settings."""
        # Handle policy configuration with CLI override support
        if isinstance(self.policy, PreTrainedConfig):
            pass  # Already parsed
        else:
            policy_path = None
            cli_overrides = []
            
            # Process YAML config (base configuration)
            if isinstance(self.policy, dict):
                if "path" not in self.policy:
                    raise ValueError("When specifying policy as a dict in YAML, 'path' key is required")
                
                policy_path = self.policy.pop("path")
                # Convert YAML dict to CLI overrides
                for k, v in self.policy.items():
                    if isinstance(v, bool):
                        cli_overrides.append(f"--{k}={str(v).lower()}")
                    else:
                        cli_overrides.append(f"--{k}={v}")
            
            # Process CLI arguments (override YAML)
            cli_policy_path = parser.get_path_arg("policy")
            if cli_policy_path:
                policy_path = cli_policy_path
            
            cli_policy_overrides = parser.get_cli_overrides("policy")
            if cli_policy_overrides:
                cli_overrides.extend(cli_policy_overrides)
        
            # Load policy from pretrained
            if policy_path:
                self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
                self.policy.pretrained_path = policy_path

        # Validate policy is set
        if self.policy is None:
            raise ValueError(
                "You must provide a policy to control the robot.\n"
                "Use CLI: --policy.path=path/to/model --policy.device=cuda\n"
                "Or YAML: policy: {path: path/to/model}"
            )

        # Validate action quantization
        if not isinstance(self.action_quant_ratio, int) or self.action_quant_ratio < 1:
            raise ValueError("action_quant_ratio must be a positive integer (>= 1)")

        # Validate inference overlap
        if not isinstance(self.inference_overlap_steps, int) or self.inference_overlap_steps < 0:
            raise ValueError("inference_overlap_steps must be a non-negative integer (>= 0)")

        # Async inference requires compiled model for CPU overlap
        if self.inference_overlap_steps > 0 and not self.policy.compile_model:
            raise ValueError(
                "When inference_overlap_steps > 0, policy.compile_model must be True. "
                "Async inference requires compiled model for CPU overlaping."
            )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """Enable draccus parser to load policy from path."""
        return ["policy"]
