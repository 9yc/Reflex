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
"""Reflex Command Line Interface.

This module provides the main entry point for the Reflex CLI. The
open-source release focuses on streaming inference / robot deployment:

    reflex run <config.yaml> [options]            # synchronous inference
    reflex run-streaming <config.yaml> [options]  # async streaming inference
"""

import sys
from pathlib import Path


def main():
    """Main entry point for the Reflex CLI.

    Parses the first argument as the command and dispatches to the
    appropriate handler function.
    """
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1]

    if command == "run":
        run_command(streaming=False)
    elif command in ["run-streaming", "run_streaming"]:
        run_command(streaming=True)
    elif command in ["--help", "-h", "help"]:
        print_usage()
    else:
        print(f"Unknown command: {command}")
        print_usage()
        sys.exit(1)


def run_command(streaming: bool = False):
    """Handle 'reflex run' / 'reflex run-streaming' for robot inference.

    Loads a trained policy and runs inference on a connected robot. The
    config file specifies robot type, policy path, and runtime settings.

    Args:
        streaming: If True, use the asynchronous streaming control loop
            (vision/policy decoupled, future-state prediction). Otherwise
            use the synchronous control loop.
    """
    entry = "run-streaming" if streaming else "run"
    if len(sys.argv) < 3:
        print(f"Usage: reflex {entry} <config.yaml> [options]")
        print("\nExamples:")
        print(f"  reflex {entry} examples/inference/async.yaml")
        print(f"  reflex {entry} examples/inference/async.yaml \\")
        print("      --policy.path=/path/to/pretrained_model")
        print(f"  reflex {entry} examples/inference/async.yaml --control_time_s=120")
        sys.exit(1)

    config_path = sys.argv[2]

    if not Path(config_path).exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    print(f"Running {'streaming ' if streaming else ''}inference with config: {config_path}")

    # Build arguments for the run module
    run_args = [f"--config_path={config_path}"]
    run_args.extend(sys.argv[3:])

    # Reconstruct sys.argv for the draccus config parser
    sys.argv = [sys.argv[0]] + run_args

    if streaming:
        from reflex.run_streaming import run_streaming
        run_streaming()
    else:
        from reflex.run import run
        run()


def print_usage():
    """Print CLI usage information and examples."""
    print("""
Reflex - Real-Time Streaming Inference for Flow-Matching VLA Policies

Usage:
  reflex <command> [arguments]

Commands:
  run <config.yaml> [options]
      Run synchronous inference with a trained policy on a robot

  run-streaming <config.yaml> [options]
      Run asynchronous streaming inference (decoupled vision/policy
      threads with future-state prediction and adaptive overlap)

  help, --help, -h
      Show this help message

Inference Examples:
  # Synchronous control loop
  reflex run examples/inference/sync.yaml \\
      --policy.path=/path/to/pretrained_model

  # Asynchronous streaming control loop
  reflex run-streaming examples/inference/async.yaml \\
      --policy.path=/path/to/pretrained_model \\
      --control_time_s=120 --inference_overlap_steps=4
    """)


if __name__ == "__main__":
    main()
