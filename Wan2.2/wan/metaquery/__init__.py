"""
MetaQuery + Wan2.2 集成模块
将基于 Qwen3-VL 的 MetaQuery 视觉语义特征注入 Wan2.2 视频生成模型。

支持:
  - MetaQueryWanBridge:          T2V (文本到视频) 增强
  - MetaQueryWanI2VBridge:       I2V (图生视频) 增强 — 首帧条件 + MetaQuery 语义条件
  - MetaQueryWanAnimateBridge:   Animate (人物动画) 增强 — 无骨架 + 面部 + MetaQuery 语义
"""

from .encoder import MetaQueryEncoder
from .bridge import MetaQueryWanBridge

# I2V / Animate bridge 的依赖可能不存在 (例如 animate_utils 模块)
# 使用延迟导入避免在只需 T2V 时因缺少依赖而崩溃
try:
    from .bridge_i2v import MetaQueryWanI2VBridge
except ImportError:
    MetaQueryWanI2VBridge = None

try:
    from .bridge_animate import MetaQueryWanAnimateBridge
except ImportError:
    MetaQueryWanAnimateBridge = None

try:
    from .bridge_animate_multigpu import MultiGPUMetaQueryAnimateBridge
except ImportError:
    MultiGPUMetaQueryAnimateBridge = None

__all__ = [
    "MetaQueryEncoder",
    "MetaQueryWanBridge",
    "MetaQueryWanI2VBridge",
    "MetaQueryWanAnimateBridge",
    "MultiGPUMetaQueryAnimateBridge",
]
