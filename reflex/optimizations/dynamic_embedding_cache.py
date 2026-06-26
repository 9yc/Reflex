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
"""Dynamic Embedding Cache for VLA Streaming.

This module implements dynamic embedding caching inspired by zip2zip:
- Cache embeddings for similar inputs to avoid recomputation
- Particularly useful for static scenes or slow-moving cameras
- Provides speedup with configurable accuracy trade-off

Key insight from zip2zip:
- zip2zip dynamically computes embeddings for new hypertokens at runtime
- We adapt this to cache image embeddings when inputs are similar
- This reduces redundant computation in VLA scenarios where consecutive
  frames often have high similarity

Usage:
    cache = DynamicEmbeddingCache(similarity_threshold=0.99)
    
    for frame in frames:
        embedding = cache.get_or_compute(
            image=frame,
            encoder_fn=image_encoder,
        )
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Entry in the embedding cache."""
    
    # Input signature for quick comparison
    input_hash: str
    
    # Cached embedding
    embedding: torch.Tensor
    
    # Original input features for similarity comparison
    input_features: torch.Tensor
    
    # Statistics
    hit_count: int = 0
    compute_time_ms: float = 0.0


class DynamicEmbeddingCache:
    """Dynamic embedding cache inspired by zip2zip.
    
    Key features:
    1. Hash-based lookup for O(1) cache access
    2. Feature-based similarity check for accurate matching
    3. LRU eviction policy for bounded memory usage
    4. Statistics tracking for optimization analysis
    
    This is particularly effective when:
    - Consecutive frames have high visual similarity
    - Some cameras have static or slow-changing views
    - The same task is executed repeatedly
    """
    
    def __init__(
        self,
        cache_size: int = 100,
        similarity_threshold: float = 0.99,
        feature_downsample_size: int = 16,
        enable_stats: bool = True,
    ):
        """Initialize the dynamic embedding cache.
        
        Args:
            cache_size: Maximum number of cached embeddings.
            similarity_threshold: Cosine similarity threshold for cache hit.
                Higher = more conservative (fewer false hits).
                Recommended: 0.99 for safety, 0.95 for more aggressive caching.
            feature_downsample_size: Size for downsampled feature comparison.
            enable_stats: If True, track detailed statistics.
        """
        self.cache_size = cache_size
        self.similarity_threshold = similarity_threshold
        self.feature_downsample_size = feature_downsample_size
        self.enable_stats = enable_stats
        
        # LRU cache using OrderedDict
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "similarity_checks": 0,
            "false_hash_matches": 0,  # Hash matched but similarity too low
            "total_compute_time_ms": 0.0,
            "total_saved_time_ms": 0.0,
        }
    
    def _compute_hash(self, image: torch.Tensor) -> str:
        """Compute a coarse hash for cache lookup.
        
        Uses heavily quantized downsampled image as hash key.
        This allows small variations (noise) to still hit the same bucket.
        """
        # Downsample to very small size for robust hashing
        small = F.interpolate(
            image, 
            size=(8, 8),  # Very small for robust hashing
            mode='bilinear',
            align_corners=False,
        )
        # Heavily quantize to make hash robust to small noise
        # Quantize to ~4 levels per channel (very coarse)
        quantized = (small * 4).round().to(torch.int8)
        # Convert to bytes and hash
        data = quantized.cpu().numpy().tobytes()
        return hashlib.md5(data).hexdigest()
    
    def _extract_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract compact features for similarity comparison.
        
        Uses average pooling for a compact representation.
        """
        # Downsample and flatten
        small = F.interpolate(
            image,
            size=(self.feature_downsample_size, self.feature_downsample_size),
            mode='bilinear',
            align_corners=False,
        )
        return small.flatten(1)  # [B, C*H*W]
    
    def _compute_similarity(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
    ) -> float:
        """Compute cosine similarity between feature vectors."""
        # Normalize and compute cosine similarity
        f1_norm = F.normalize(features1, dim=-1)
        f2_norm = F.normalize(features2, dim=-1)
        similarity = (f1_norm * f2_norm).sum(dim=-1).mean().item()
        return similarity
    
    def get_or_compute(
        self,
        image: torch.Tensor,
        encoder_fn: Callable[[torch.Tensor], torch.Tensor],
        force_compute: bool = False,
    ) -> Tuple[torch.Tensor, bool]:
        """Get cached embedding or compute new one.
        
        Args:
            image: Input image tensor [B, C, H, W].
            encoder_fn: Function to compute embedding if cache miss.
            force_compute: If True, bypass cache and always compute.
            
        Returns:
            Tuple of (embedding, is_cache_hit).
        """
        if self.enable_stats:
            self.stats["total_requests"] += 1
        
        if force_compute:
            embedding = encoder_fn(image)
            return embedding, False
        
        # Compute hash for lookup
        input_hash = self._compute_hash(image)
        
        # Check cache
        if input_hash in self.cache:
            entry = self.cache[input_hash]
            
            # Verify similarity (guard against hash collision)
            input_features = self._extract_features(image)
            similarity = self._compute_similarity(input_features, entry.input_features)
            
            if self.enable_stats:
                self.stats["similarity_checks"] += 1
            
            if similarity >= self.similarity_threshold:
                # Cache hit!
                if self.enable_stats:
                    self.stats["cache_hits"] += 1
                    self.stats["total_saved_time_ms"] += entry.compute_time_ms
                
                entry.hit_count += 1
                
                # Move to end (LRU)
                self.cache.move_to_end(input_hash)
                
                return entry.embedding, True
            else:
                # Hash collision or image changed
                if self.enable_stats:
                    self.stats["false_hash_matches"] += 1
        
        # Cache miss - compute embedding
        if self.enable_stats:
            self.stats["cache_misses"] += 1
        
        import time
        start = time.perf_counter()
        embedding = encoder_fn(image)
        compute_time = (time.perf_counter() - start) * 1000
        
        if self.enable_stats:
            self.stats["total_compute_time_ms"] += compute_time
        
        # Cache the result
        input_features = self._extract_features(image)
        entry = CacheEntry(
            input_hash=input_hash,
            embedding=embedding.detach(),
            input_features=input_features.detach(),
            compute_time_ms=compute_time,
        )
        
        # Add to cache (with LRU eviction)
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            self.cache.popitem(last=False)
        
        self.cache[input_hash] = entry
        
        return embedding, False
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total = self.stats["total_requests"]
        if total == 0:
            hit_rate = 0.0
        else:
            hit_rate = self.stats["cache_hits"] / total
        
        return {
            **self.stats,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "max_cache_size": self.cache_size,
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "similarity_checks": 0,
            "false_hash_matches": 0,
            "total_compute_time_ms": 0.0,
            "total_saved_time_ms": 0.0,
        }
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.reset_stats()


class LastFrameCache:
    """Simple last-frame cache for VLA streaming.
    
    This cache stores only the most recent frame's embedding per camera.
    It uses direct similarity comparison (no hashing) which is more accurate
    for real robot scenarios where frames change gradually.
    
    Key insight: In robot control, consecutive frames are usually very similar.
    We can reuse the previous embedding if similarity is above threshold.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.99,
        feature_downsample_size: int = 16,
    ):
        """Initialize last-frame cache.
        
        Args:
            similarity_threshold: Cosine similarity threshold for cache hit.
            feature_downsample_size: Size for feature extraction.
        """
        self.similarity_threshold = similarity_threshold
        self.feature_downsample_size = feature_downsample_size
        
        # Last frame state
        self.last_features: Optional[torch.Tensor] = None
        self.last_embedding: Optional[torch.Tensor] = None
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
    
    def _extract_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract compact features for similarity comparison."""
        small = F.interpolate(
            image,
            size=(self.feature_downsample_size, self.feature_downsample_size),
            mode='bilinear',
            align_corners=False,
        )
        return small.flatten(1)
    
    def _compute_similarity(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
    ) -> float:
        """Compute cosine similarity between feature vectors."""
        f1_norm = F.normalize(features1, dim=-1)
        f2_norm = F.normalize(features2, dim=-1)
        similarity = (f1_norm * f2_norm).sum(dim=-1).mean().item()
        return similarity
    
    def get_or_compute(
        self,
        image: torch.Tensor,
        encoder_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> Tuple[torch.Tensor, bool]:
        """Get cached embedding or compute new one.
        
        Args:
            image: Input image tensor [B, C, H, W].
            encoder_fn: Function to compute embedding if cache miss.
            
        Returns:
            Tuple of (embedding, is_cache_hit).
        """
        self.stats["total_requests"] += 1
        
        # Extract features for current frame
        current_features = self._extract_features(image)
        
        # Check if we have a cached frame
        if self.last_features is not None and self.last_embedding is not None:
            similarity = self._compute_similarity(current_features, self.last_features)
            
            if similarity >= self.similarity_threshold:
                # Cache hit - reuse last embedding
                self.stats["cache_hits"] += 1
                return self.last_embedding, True
        
        # Cache miss - compute new embedding
        self.stats["cache_misses"] += 1
        embedding = encoder_fn(image)
        
        # Update cache
        self.last_features = current_features.detach()
        self.last_embedding = embedding.detach()
        
        return embedding, False
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total = self.stats["total_requests"]
        hit_rate = self.stats["cache_hits"] / total if total > 0 else 0.0
        return {
            **self.stats,
            "hit_rate": hit_rate,
        }
    
    def reset(self):
        """Reset cache state."""
        self.last_features = None
        self.last_embedding = None
        self.stats = {"total_requests": 0, "cache_hits": 0, "cache_misses": 0}


class MultiCameraEmbeddingCache:
    """Embedding cache for multiple cameras.
    
    Each camera has its own cache since different cameras may have
    different change rates (e.g., overhead camera more static than wrist camera).
    """
    
    def __init__(
        self,
        num_cameras: int,
        cache_size_per_camera: int = 50,
        similarity_threshold: float = 0.99,
        use_last_frame_cache: bool = True,  # Use simpler, more accurate cache
    ):
        """Initialize multi-camera cache.
        
        Args:
            num_cameras: Number of cameras.
            cache_size_per_camera: Cache size for each camera.
            similarity_threshold: Similarity threshold for cache hit.
            use_last_frame_cache: If True, use LastFrameCache (recommended for VLA).
        """
        self.num_cameras = num_cameras
        self.use_last_frame_cache = use_last_frame_cache
        
        if use_last_frame_cache:
            self.caches = [
                LastFrameCache(similarity_threshold=similarity_threshold)
                for _ in range(num_cameras)
            ]
        else:
            self.caches = [
                DynamicEmbeddingCache(
                    cache_size=cache_size_per_camera,
                    similarity_threshold=similarity_threshold,
                )
                for _ in range(num_cameras)
            ]
    
    def get_or_compute_all(
        self,
        images: list[torch.Tensor],
        encoder_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> Tuple[list[torch.Tensor], list[bool]]:
        """Get or compute embeddings for all cameras.
        
        Args:
            images: List of image tensors, one per camera.
            encoder_fn: Encoder function.
            
        Returns:
            Tuple of (embeddings_list, cache_hits_list).
        """
        if self.use_last_frame_cache:
            return self._get_or_compute_last_frame(images, encoder_fn)
        else:
            return self._get_or_compute_hash_based(images, encoder_fn)
    
    def _get_or_compute_last_frame(
        self,
        images: list[torch.Tensor],
        encoder_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> Tuple[list[torch.Tensor], list[bool]]:
        """Use last-frame similarity-based cache."""
        embeddings = []
        cache_hits = []
        need_compute = []
        cached_results = {}
        
        # Check each camera against its last frame
        for cam_idx, image in enumerate(images):
            cache = self.caches[cam_idx]
            
            # Extract features for similarity check
            current_features = cache._extract_features(image)
            
            if cache.last_features is not None and cache.last_embedding is not None:
                similarity = cache._compute_similarity(current_features, cache.last_features)
                
                if similarity >= cache.similarity_threshold:
                    # Cache hit
                    cached_results[cam_idx] = cache.last_embedding
                    cache.stats["cache_hits"] += 1
                else:
                    # Similar but changed - need to recompute
                    need_compute.append((cam_idx, image, current_features))
                    cache.stats["cache_misses"] += 1
            else:
                # First frame for this camera
                need_compute.append((cam_idx, image, current_features))
                cache.stats["cache_misses"] += 1
            
            cache.stats["total_requests"] += 1
        
        # Batch compute for cache misses
        if need_compute:
            compute_images = [img for _, img, _ in need_compute]
            batched = torch.stack(compute_images, dim=1)
            batched = batched.reshape(-1, *batched.shape[2:])
            
            batched_emb = encoder_fn(batched)
            
            # Split results
            batch_size = images[0].shape[0]
            num_computed = len(need_compute)
            emb_per_image = batched_emb.shape[1]
            emb_dim = batched_emb.shape[2]
            computed_embs = batched_emb.reshape(batch_size, num_computed, emb_per_image, emb_dim)
            
            # Update caches
            for i, (cam_idx, image, current_features) in enumerate(need_compute):
                emb = computed_embs[:, i]
                cache = self.caches[cam_idx]
                
                # Update last frame cache
                cache.last_features = current_features.detach()
                cache.last_embedding = emb.detach()
                
                cached_results[cam_idx] = emb
        
        # Build results in order
        for cam_idx in range(len(images)):
            embeddings.append(cached_results[cam_idx])
            cache_hits.append(cam_idx not in [c[0] for c in need_compute])
        
        return embeddings, cache_hits
    
    def _get_or_compute_hash_based(
        self,
        images: list[torch.Tensor],
        encoder_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> Tuple[list[torch.Tensor], list[bool]]:
        """Use hash-based cache (legacy method)."""
        embeddings = []
        cache_hits = []
        
        # Check which cameras need computation
        need_compute = []
        cached_results = {}
        
        for cam_idx, image in enumerate(images):
            cache = self.caches[cam_idx]
            input_hash = cache._compute_hash(image)
            
            if input_hash in cache.cache:
                entry = cache.cache[input_hash]
                input_features = cache._extract_features(image)
                similarity = cache._compute_similarity(input_features, entry.input_features)
                
                if similarity >= cache.similarity_threshold:
                    cached_results[cam_idx] = entry.embedding
                    entry.hit_count += 1
                    cache.cache.move_to_end(input_hash)
                    cache.stats["cache_hits"] += 1
                else:
                    need_compute.append((cam_idx, image, input_hash))
                    cache.stats["false_hash_matches"] += 1
            else:
                need_compute.append((cam_idx, image, input_hash))
                cache.stats["cache_misses"] += 1
        
        # Batch compute for cache misses
        if need_compute:
            compute_images = [img for _, img, _ in need_compute]
            batched = torch.stack(compute_images, dim=1)
            batched = batched.reshape(-1, *batched.shape[2:])
            
            import time
            start = time.perf_counter()
            batched_emb = encoder_fn(batched)
            compute_time = (time.perf_counter() - start) * 1000
            
            # Split results
            batch_size = images[0].shape[0]
            num_computed = len(need_compute)
            emb_per_image = batched_emb.shape[1]
            emb_dim = batched_emb.shape[2]
            computed_embs = batched_emb.reshape(batch_size, num_computed, emb_per_image, emb_dim)
            
            # Cache results
            for i, (cam_idx, image, input_hash) in enumerate(need_compute):
                emb = computed_embs[:, i]
                cache = self.caches[cam_idx]
                
                input_features = cache._extract_features(image)
                entry = CacheEntry(
                    input_hash=input_hash,
                    embedding=emb.detach(),
                    input_features=input_features.detach(),
                    compute_time_ms=compute_time / num_computed,
                )
                
                if len(cache.cache) >= cache.cache_size:
                    cache.cache.popitem(last=False)
                cache.cache[input_hash] = entry
                
                cached_results[cam_idx] = emb
        
        # Build results in order
        for cam_idx in range(len(images)):
            embeddings.append(cached_results[cam_idx])
            cache_hits.append(cam_idx not in [c[0] for c in need_compute])
        
        return embeddings, cache_hits
    
    def get_stats(self) -> Dict:
        """Get aggregated statistics."""
        total_hits = sum(c.stats["cache_hits"] for c in self.caches)
        total_requests = sum(c.stats["total_requests"] for c in self.caches)
        
        return {
            "total_hits": total_hits,
            "total_requests": total_requests,
            "hit_rate": total_hits / total_requests if total_requests > 0 else 0,
            "per_camera_stats": [c.get_stats() for c in self.caches],
        }
    
    def clear(self):
        """Clear all caches."""
        for cache in self.caches:
            cache.clear()


def benchmark_dynamic_embedding_cache():
    """Benchmark the dynamic embedding cache."""
    import time
    import numpy as np
    
    print("="*70)
    print("DYNAMIC EMBEDDING CACHE BENCHMARK")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 1
    num_cameras = 3
    img_size = 224
    
    # Create mock encoder
    class MockEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 256, 16, stride=16)
            self.pool = torch.nn.AdaptiveAvgPool2d((7, 7))
            
        def forward(self, x):
            x = F.gelu(self.conv(x))
            x = self.pool(x)
            return x.flatten(2).transpose(1, 2)
    
    encoder = MockEncoder().to(device)
    cache = DynamicEmbeddingCache(similarity_threshold=0.99)
    
    # Test 1: Cache miss (first frame)
    print("\n1. First frame (cache miss):")
    image = torch.randn(batch_size, 3, img_size, img_size, device=device)
    
    start = time.perf_counter()
    emb1, hit1 = cache.get_or_compute(image, encoder)
    time1 = (time.perf_counter() - start) * 1000
    print(f"   Cache hit: {hit1}")
    print(f"   Time: {time1:.3f}ms")
    
    # Test 2: Same image (cache hit)
    print("\n2. Same image (cache hit):")
    start = time.perf_counter()
    emb2, hit2 = cache.get_or_compute(image, encoder)
    time2 = (time.perf_counter() - start) * 1000
    print(f"   Cache hit: {hit2}")
    print(f"   Time: {time2:.3f}ms")
    print(f"   Speedup: {time1/time2:.2f}x")
    
    # Test 3: Similar image (should hit with high threshold)
    print("\n3. Similar image (small noise):")
    similar_image = image + torch.randn_like(image) * 0.001  # Very small noise
    start = time.perf_counter()
    emb3, hit3 = cache.get_or_compute(similar_image, encoder)
    time3 = (time.perf_counter() - start) * 1000
    print(f"   Cache hit: {hit3}")
    print(f"   Time: {time3:.3f}ms")
    
    # Test 4: Different image (cache miss)
    print("\n4. Different image (cache miss):")
    different_image = torch.randn(batch_size, 3, img_size, img_size, device=device)
    start = time.perf_counter()
    emb4, hit4 = cache.get_or_compute(different_image, encoder)
    time4 = (time.perf_counter() - start) * 1000
    print(f"   Cache hit: {hit4}")
    print(f"   Time: {time4:.3f}ms")
    
    # Statistics
    print("\n5. Cache Statistics:")
    stats = cache.get_stats()
    print(f"   Total requests: {stats['total_requests']}")
    print(f"   Cache hits: {stats['cache_hits']}")
    print(f"   Hit rate: {stats['hit_rate']*100:.1f}%")
    
    # Simulate static scene
    print("\n6. Static Scene Simulation (100 frames):")
    cache.clear()
    base_image = torch.randn(batch_size, 3, img_size, img_size, device=device)
    
    hits = 0
    total_time = 0
    for i in range(100):
        # Add very small noise (simulating sensor noise in static scene)
        frame = base_image + torch.randn_like(base_image) * 0.0001
        
        start = time.perf_counter()
        _, hit = cache.get_or_compute(frame, encoder)
        total_time += (time.perf_counter() - start) * 1000
        
        if hit:
            hits += 1
    
    print(f"   Frames: 100")
    print(f"   Cache hits: {hits}")
    print(f"   Hit rate: {hits}%")
    print(f"   Total time: {total_time:.2f}ms")
    print(f"   Time per frame: {total_time/100:.3f}ms")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    benchmark_dynamic_embedding_cache()

