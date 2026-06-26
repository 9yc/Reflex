#!/usr/bin/env python
"""Demo: How to enable block-based KV cache in PI0.5."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from reflex.policies.pi05.configuration_pi05 import PI05Config


def demo_default_config():
    """示例 1: 默认配置（block cache 关闭）"""
    print("=" * 60)
    print("示例 1: 默认配置")
    print("=" * 60)
    
    config = PI05Config()
    config.enable_streaming = True
    
    print(f"enable_streaming: {config.enable_streaming}")
    print(f"use_blocked_cache: {config.use_blocked_cache}")
    print(f"kv_cache_block_size: {config.kv_cache_block_size}")
    print()
    print("默认使用 continuous cache（标准模式）")
    print()


def demo_enable_block_cache():
    """示例 2: 启用 block cache"""
    print("=" * 60)
    print("示例 2: 启用 block cache")
    print("=" * 60)
    
    config = PI05Config()
    config.enable_streaming = True
    
    # 启用 block cache
    config._streaming_config["use_blocked_cache"] = True
    
    print(f"enable_streaming: {config.enable_streaming}")
    print(f"use_blocked_cache: {config.use_blocked_cache}")
    print(f"kv_cache_block_size: {config.kv_cache_block_size}")
    print()
    print("使用 block-based cache（PagedAttention 风格）")
    print()


def demo_custom_block_size():
    """示例 3: 自定义 block size"""
    print("=" * 60)
    print("示例 3: 自定义 block size")
    print("=" * 60)
    
    config = PI05Config()
    config.enable_streaming = True
    config._streaming_config["use_blocked_cache"] = True
    
    # 测试不同的 block sizes
    for block_size in [8, 16, 32, 64]:
        config._streaming_config["kv_block_size"] = block_size
        print(f"block_size={block_size}: use_blocked_cache={config.use_blocked_cache}")
    
    print()
    print("可以根据序列长度和硬件选择合适的 block size")
    print()


def demo_usage_in_code():
    """示例 4: 在代码中使用"""
    print("=" * 60)
    print("示例 4: 在代码中使用")
    print("=" * 60)
    
    print("""
from reflex.policies.pi05.modeling_pi05 import PI05Policy
from reflex.policies.pi05.configuration_pi05 import PI05Config

# 创建配置
config = PI05Config()
config.enable_streaming = True

# 启用 block cache（可选）
config._streaming_config["use_blocked_cache"] = True
config._streaming_config["kv_block_size"] = 16

# 创建 policy
policy = PI05Policy(config)

# 使用 policy
# policy.select_action(observation)
    """)
    print()


def demo_when_to_use():
    """示例 5: 何时使用 block cache"""
    print("=" * 60)
    print("示例 5: 何时使用 block cache")
    print("=" * 60)
    
    print("""
推荐使用 block cache 的场景：

✅ 长序列（>200 tokens）
   - 显存占用更少
   - 内存碎片更少

✅ 长时间运行（100+ steps）
   - 缓存增长更可控
   - 稳定性更好

✅ 多相机场景（3-4 个相机）
   - 显存优势更明显

⚠️ 可能不适用的场景：

❌ 短序列（<100 tokens）
   - Block overhead 可能无优势

❌ 对延迟极敏感的场景
   - 可能有轻微延迟增加（通常 <10%）

推荐：
- 先使用默认的 continuous cache
- 如果遇到显存或稳定性问题，尝试 block cache
- 通过 benchmark 评估是否有收益
    """)
    print()


if __name__ == "__main__":
    demo_default_config()
    demo_enable_block_cache()
    demo_custom_block_size()
    demo_usage_in_code()
    demo_when_to_use()
    
    print("=" * 60)
    print("更多信息")
    print("=" * 60)
    print()
    print("1. 配置文档: reflex/policies/pi05/configuration_pi05.py")
    print("2. 实现文档: docs/implementation/BLOCK_CACHE_IMPLEMENTATION_PLAN.md")
    print("3. 测试脚本: tests/test_block_cache_*.py")
    print()




























