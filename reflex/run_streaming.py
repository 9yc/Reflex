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
"""Reflex Robot Inference Module with StreamingThinker Enhancements.

This module extends the standard Reflex inference with StreamingThinker-inspired
features:
- Streaming attention masks for order-preserving reasoning
- Parallel KV caches for decoupled input encoding and action generation
- Streaming position encoding for independent indexing

Key improvements:
1. Reduced latency through parallel processing
2. Better attention alignment with observation sequence
3. Improved stability through streaming constraints
"""

import logging
import time
from dataclasses import asdict
from pprint import pformat
from copy import copy
import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.robots import Robot, make_robot_from_config
from lerobot.utils.constants import OBS_IMAGES
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    prepare_observation_for_inference,
)
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import get_safe_torch_device, init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

from reflex.configs import RunConfig
from reflex.policies.factory import get_policy_class
from reflex.run import (
    ReflexAsyncManager,
    validate_robot_cameras,
    load_and_compile_policy,
    build_dataset_features,
)


class StreamingReflexAsyncManager(ReflexAsyncManager):
    """Extended async manager with StreamingThinker enhancements.
    
    This class extends ReflexAsyncManager with:
    - Streaming attention support
    - Parallel KV cache management
    - Streaming position encoding
    
    The key difference is that it enables the model to process observations
    incrementally while generating actions, reducing latency and improving
    alignment with the observation sequence.
    """
    
    def __init__(
        self,
        policy: PreTrainedPolicy,
        robot: Robot,
        single_task: str | None,
        overlap_steps: int,
        enable_streaming: bool = True,
        streaming_window: int | None = None,
    ):
        """Initialize streaming async manager.
        
        Args:
            policy: Trained policy for action prediction.
            robot: Robot instance to control.
            single_task: Task description string.
            overlap_steps: Number of steps before chunk end to start next inference.
            enable_streaming: If True, enable streaming attention mechanisms.
            streaming_window: Optional window size for streaming attention.
        """
        super().__init__(policy, robot, single_task, overlap_steps)
        self.enable_streaming = enable_streaming
        self.streaming_window = streaming_window
        
        # Track observation sequence for streaming attention
        self.observation_history = []
        self.max_history_length = 10  # Keep last N observations for streaming
        # Track whether we've already fed the first chunk (needed for incremental_input)
        self._streaming_incremental_input = False
        
    def launch_next_inference(self, observation: dict[str, np.ndarray]) -> torch.Tensor:
        """Compute next action chunk with streaming support.
        
        If streaming is enabled, this method:
        1. Maintains observation history for streaming attention
        2. Uses parallel KV caches for efficient processing
        3. Applies streaming attention masks for order-preserving reasoning
        
        Args:
            observation: Current observation dictionary.
            
        Returns:
            Predicted action chunk as a torch tensor [n_action_steps, action_dim].
        """
        observation = copy(observation)
        
        # Future state awareness: use future state instead of current state
        last_action = self.current_chunk[self.n_action_steps - 1] if self.current_chunk is not None else None
        if last_action is not None:
            observation["observation.state"] = last_action
        
        # Maintain observation history for streaming attention
        if self.enable_streaming:
            self.observation_history.append(observation)
            if len(self.observation_history) > self.max_history_length:
                self.observation_history.pop(0)
        
        observation = copy(observation)
        
        # Future state awareness: use future state instead of current state
        last_action = self.current_chunk[self.n_action_steps - 1] if self.current_chunk is not None else None
        if last_action is not None:
            # Handle potential dimension mismatch (e.g. action=7, state=8)
            state_dim = observation["observation.state"].shape[-1]
            if last_action.shape[-1] < state_dim:
                padded_state = np.zeros_like(observation["observation.state"])
                padded_state[:last_action.shape[-1]] = last_action
                observation["observation.state"] = padded_state
            else:
                observation["observation.state"] = last_action[:state_dim]

        with torch.inference_mode():
            # Prepare observation: convert images to CHW format, normalize, add batch dim
            observation = prepare_observation_for_inference(
                observation,
                self.device,
                self.single_task,
                self.robot.robot_type,
            )
            
            # Enable streaming in model if supported
            if self.enable_streaming and hasattr(self.policy.model, 'enable_streaming'):
                self.policy.model.enable_streaming = True
                if self.streaming_window is not None:
                    self.policy.model.streaming_window = self.streaming_window
            
            # Run policy inference to get action chunk
            incremental_input = self.enable_streaming and self._streaming_incremental_input
            action_chunk = self.policy.predict_action_chunk(
                observation,
                incremental_input=incremental_input,
            )
            if self.enable_streaming:
                # After the first inference we always switch to incremental mode
                self._streaming_incremental_input = True
            
            # Disable streaming after inference
            if self.enable_streaming and hasattr(self.policy.model, 'enable_streaming'):
                self.policy.model.enable_streaming = False
        
        # Remove batch dimension
        return action_chunk.squeeze(0)


@torch.inference_mode()
def run_loop_streaming(
    robot: Robot,
    events: dict,
    fps: int,
    dataset_features: dict[str, dict],
    policy: PreTrainedPolicy,
    single_task: str | None,
    action_quant_ratio: int = 1,
    inference_overlap_steps: int = 0,
    display_data: bool = False,
    control_time_s: int | float = 60,
    enable_streaming: bool = True,
    streaming_window: int | None = None,
):
    """Core control loop with StreamingThinker enhancements.
    
    This is similar to the standard run_loop but uses StreamingReflexAsyncManager
    for improved latency and attention alignment.
    
    Args:
        robot: Connected robot instance.
        events: Event dictionary for keyboard control (exit_early flag).
        fps: Target control frequency in Hz.
        dataset_features: Feature definitions for observation/action conversion.
        policy: Loaded policy for action prediction.
        single_task: Task description for policies.
        action_quant_ratio: Action quantization ratio.
        inference_overlap_steps: Steps of overlap between chunks.
        display_data: Whether to log data to Rerun for visualization.
        control_time_s: Total runtime in seconds.
        enable_streaming: If True, enable streaming attention mechanisms.
        streaming_window: Optional window size for streaming attention.
    """
    # Reset policy state (clears any cached observations)
    if policy is not None:
        policy.reset()
    
    # Initialize streaming async manager for Reflex inference
    effective_overlap_steps = inference_overlap_steps * action_quant_ratio
    logging.info(
        f"Streaming Reflex: effective_overlap_steps={effective_overlap_steps} "
        f"(inference_overlap_steps={inference_overlap_steps} * action_quant_ratio={action_quant_ratio})"
    )
    if enable_streaming:
        logging.info(f"Streaming attention enabled (window={streaming_window})")
    
    async_manager = StreamingReflexAsyncManager(
        policy=policy,
        robot=robot,
        single_task=single_task,
        overlap_steps=effective_overlap_steps,
        enable_streaming=enable_streaming,
        streaming_window=streaming_window,
    )
    
    step_count = 0
    observation_frame = None
    start_time = time.perf_counter()
    
    # Track latency metrics
    inference_times = []
    action_times = []
    
    # Main control loop
    while time.perf_counter() - start_time < control_time_s:
        loop_start = time.perf_counter()
        
        # Check for keyboard interrupt (Escape key)
        if events["exit_early"]:
            events["exit_early"] = False
            break
        
        # Fetch observation only when needed (reduces camera latency)
        if async_manager.should_fetch_observation():
            obs_start = time.perf_counter()
            observation = robot.get_observation()
            observation_frame = build_dataset_frame(dataset_features, observation, prefix="observation")
            obs_time = time.perf_counter() - obs_start
            action_times.append(obs_time)
        else:
            observation = None
        
        # Get action from async manager (handles chunk management internally)
        inf_start = time.perf_counter()
        action = async_manager.get_action(observation_frame)
        inf_time = time.perf_counter() - inf_start
        inference_times.append(inf_time)
        
        # Send action based on quantization ratio
        if (step_count + 1) % action_quant_ratio == 0:
            robot.send_action(action)
            
            # Optional: log to Rerun for debugging/visualization
            if display_data and observation is not None:
                log_rerun_data(observation, action)
            
            # Maintain target frequency
            elapsed = time.perf_counter() - loop_start
            busy_wait(1 / fps - elapsed)
        
        step_count += 1
    
    # Log latency statistics
    if inference_times:
        avg_inf_time = np.mean(inference_times) * 1000  # Convert to ms
        max_inf_time = np.max(inference_times) * 1000
        logging.info(f"Streaming Reflex latency stats:")
        logging.info(f"  Average inference time: {avg_inf_time:.2f} ms")
        logging.info(f"  Max inference time: {max_inf_time:.2f} ms")
        if action_times:
            avg_obs_time = np.mean(action_times) * 1000
            logging.info(f"  Average observation time: {avg_obs_time:.2f} ms")


@parser.wrap()
def run_streaming(cfg: RunConfig):
    """Main entry point for Streaming Reflex robot inference.
    
    This extends the standard Reflex run with StreamingThinker enhancements.
    
    Args:
        cfg: Run configuration parsed from YAML and CLI arguments.
    """
    init_logging()
    logging.info("=" * 60)
    logging.info("Streaming Reflex with StreamingThinker Enhancements")
    logging.info("=" * 60)
    logging.info(pformat(asdict(cfg)))
    
    # Validate task description is provided (not placeholder)
    if cfg.single_task is None or cfg.single_task == "<task description>":
        raise ValueError(
            "Please provide a language prompt (task description) in the config file.\n"
            "The 'single_task' field cannot be empty or use the placeholder '<task description>'.\n"
            "Example: single_task: 'pick up the cube and place it in the box'"
        )
    
    # Initialize Rerun visualization if requested
    if cfg.display_data:
        init_rerun(session_name="reflex_streaming_run")
    
    # Setup robot and validate camera configuration
    robot = make_robot_from_config(cfg.robot)
    original_policy_config = PreTrainedConfig.from_pretrained(cfg.policy.pretrained_path)
    validate_robot_cameras(robot, original_policy_config)
    
    # Load policy and prepare feature definitions
    policy = load_and_compile_policy(cfg)
    dataset_features = build_dataset_features(robot)
    
    # Connect to robot and setup keyboard listener for manual control
    robot.connect()
    listener, events = init_keyboard_listener()
    
    log_say("Starting Streaming Reflex run", cfg.play_sounds, blocking=True)
    
    # Get streaming configuration from cfg or use defaults
    enable_streaming = getattr(cfg, 'enable_streaming', True)
    streaming_window = getattr(cfg, 'streaming_window', None)
    
    try:
        # Run the main control loop with streaming enhancements
        run_loop_streaming(
            robot=robot,
            events=events,
            fps=cfg.fps,
            dataset_features=dataset_features,
            policy=policy,
            single_task=cfg.single_task,
            action_quant_ratio=cfg.action_quant_ratio,
            inference_overlap_steps=cfg.inference_overlap_steps,
            display_data=cfg.display_data,
            control_time_s=cfg.control_time_s,
            enable_streaming=enable_streaming,
            streaming_window=streaming_window,
        )
    finally:
        # Cleanup: disconnect robot and stop keyboard listener
        log_say("Stopping Streaming Reflex run", cfg.play_sounds, blocking=True)
        robot.disconnect()
        if listener is not None:
            listener.stop()


def main():
    """CLI entry point."""
    run_streaming()


if __name__ == "__main__":
    main()











