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
"""Concurrent Executor for Streaming Parallel Inference.

This module implements concurrent execution of Source-side prefill and
Target-side decoding, as described in StreamingThinker Algorithm 1.
"""

import threading
from typing import Optional, Callable, Any
import torch


class ConcurrentStreamingExecutor:
    """Executor for concurrent Source prefill and Target decode.
    
    This class enables true concurrent execution:
    - Source-side prefill: runs in background thread
    - Target-side decode: runs in main thread
    - Thread-safe cache management
    """
    
    def __init__(self, use_cuda_streams: bool = False):
        """Initialize concurrent executor.
        
        Args:
            use_cuda_streams: If True, use CUDA Streams for GPU-level concurrency.
                            If False, use Python threads for CPU-level concurrency.
        """
        self.use_cuda_streams = use_cuda_streams and torch.cuda.is_available()
        
        # Thread safety
        self.prefill_lock = threading.Lock()
        self.decode_lock = threading.Lock()
        
        # CUDA Streams (if enabled)
        if self.use_cuda_streams:
            self.prefill_stream = torch.cuda.Stream()
            self.decode_stream = torch.cuda.Stream()
            self.default_stream = torch.cuda.current_stream()
            
            # Use Events for synchronization instead of global sync
            # This allows better concurrency and reduces overhead
            self.prefill_event = torch.cuda.Event(enable_timing=False)
            self.decode_event = torch.cuda.Event(enable_timing=False)
        else:
            self.prefill_stream = None
            self.decode_stream = None
            self.default_stream = None
            self.prefill_event = None
            self.decode_event = None
    
    def concurrent_prefill(
        self,
        prefill_func: Callable[[], Any],
        wait: bool = False,
    ) -> Optional[threading.Thread]:
        """Execute prefill concurrently (doesn't block decode).
        
        Args:
            prefill_func: Function to execute for prefill.
            wait: If True, wait for prefill to complete before returning.
                 If False, return immediately and prefill runs in background.
        
        Returns:
            Thread object if not waiting, None if waiting.
        """
        # Capture caller stream (typically default stream) so prefill stream can safely
        # consume tensors produced earlier in the caller stream.
        caller_stream = torch.cuda.current_stream() if self.use_cuda_streams else None

        def prefill_worker():
            """Background worker for prefill."""
            with self.prefill_lock:
                if self.use_cuda_streams:
                    # Execute in prefill stream (non-blocking)
                    with torch.cuda.stream(self.prefill_stream):
                        # Ensure prefill stream sees inputs that were produced on the caller stream.
                        # Without this, prefill may read partially-written tensors (race) and can
                        # corrupt caches / produce NaNs.
                        if caller_stream is not None:
                            self.prefill_stream.wait_stream(caller_stream)
                        result = prefill_func()
                        # Record event instead of immediate sync
                        # This allows decode to run concurrently
                        self.prefill_event.record(self.prefill_stream)
                else:
                    result = prefill_func()
            return result
        
        if wait:
            # Execute synchronously
            prefill_worker()
            # IMPORTANT: If using CUDA streams, the prefill work is enqueued on a non-default
            # stream. Callers that pass wait=True expect prefix KV to be *ready* when we return,
            # so we must insert the proper stream dependency here.
            if self.use_cuda_streams and self.prefill_event is not None:
                torch.cuda.current_stream().wait_event(self.prefill_event)
            return None
        else:
            # Execute in background thread
            thread = threading.Thread(target=prefill_worker, daemon=True)
            thread.start()
            return thread
    
    def concurrent_decode(
        self,
        decode_func: Callable[[], Any],
        wait_for_prefill: bool = True,
    ) -> Any:
        """Execute decode (can run concurrently with prefill).
        
        Args:
            decode_func: Function to execute for decode.
        
        Returns:
            Decode result.
        """
        with self.decode_lock:
            if self.use_cuda_streams:
                # Capture caller stream (typically default stream) so we can establish
                # a correct dependency once decode is enqueued/recorded.
                caller_stream = torch.cuda.current_stream()
                # Execute in decode stream
                with torch.cuda.stream(self.decode_stream):
                    # Ensure decode stream sees inputs (e.g., x_t) produced/updated on caller stream.
                    # This is essential for the diffusion loop where x_t is updated on the default
                    # stream between decode steps.
                    self.decode_stream.wait_stream(caller_stream)
                    # Optionally wait for prefill to complete (if it's running).
                    #
                    # For "think while reading" streaming, decode may intentionally run while
                    # prefill is appending NEW prefix KV, as long as decode only attends to the
                    # already-visible prefix slice (which is immutable). In that case, callers
                    # should pass wait_for_prefill=False.
                    if wait_for_prefill and self.prefill_event is not None:
                        self.prefill_event.wait(stream=self.decode_stream)
                    result = decode_func()
                    # Record decode completion
                    self.decode_event.record(self.decode_stream)
                # Ensure any subsequent ops on the caller stream that consume `result` (or tensors
                # it depends on) will wait for decode stream completion.
                #
                # Without this, the caller may enqueue computations (e.g., `x_t = x_t + dt * v_t`)
                # on the default stream before `v_t` is produced on `decode_stream`, which can lead
                # to race conditions and NaNs.
                if self.decode_event is not None:
                    caller_stream.wait_event(self.decode_event)
            else:
                result = decode_func()
        return result
    
    def wait_for_prefill(self, thread: Optional[threading.Thread]):
        """Wait for prefill thread to complete.
        
        Args:
            thread: Thread object returned from concurrent_prefill.
        """
        if thread is not None:
            thread.join()
    
    def synchronize_streams(self):
        """Synchronize CUDA streams (if using CUDA Streams).
        
        This should be called after all concurrent operations are complete.
        Uses Events for efficient synchronization instead of global sync.
        """
        if self.use_cuda_streams:
            # Wait for both events to complete
            # This is more efficient than torch.cuda.synchronize()
            # as it only waits for the specific operations
            if self.prefill_event is not None:
                self.prefill_event.wait()
            if self.decode_event is not None:
                self.decode_event.wait()
            # Alternatively, can use torch.cuda.synchronize() for simplicity
            # torch.cuda.synchronize()


class ThreadSafeStreamingManager:
    """Thread-safe wrapper for StreamingInputManager."""
    
    def __init__(self, max_history_length: Optional[int] = None):
        """Initialize thread-safe streaming manager.
        
        Args:
            max_history_length: Maximum number of prefix chunks to keep.
        """
        from reflex.policies.pi05.streaming_manager import StreamingInputManager
        
        self.manager = StreamingInputManager(max_history_length)
        self.lock = threading.Lock()
    
    def add_new_prefix(self, prefix_embs, prefix_pad_masks, prefix_att_masks):
        """Thread-safe add new prefix."""
        with self.lock:
            return self.manager.add_new_prefix(prefix_embs, prefix_pad_masks, prefix_att_masks)
    
    def get_full_prefix(self, start_idx=0, end_idx=None):
        """Thread-safe get full prefix."""
        with self.lock:
            return self.manager.get_full_prefix(start_idx, end_idx)
    
    def has_history(self):
        """Thread-safe check history."""
        with self.lock:
            return self.manager.has_history()
    
    def get_prefix_length(self):
        """Thread-safe get prefix length."""
        with self.lock:
            return self.manager.get_prefix_length()
    
    def reset(self):
        """Thread-safe reset."""
        with self.lock:
            self.manager.reset()
    
    def has_history(self):
        """Thread-safe check history."""
        with self.lock:
            return self.manager.has_history()

