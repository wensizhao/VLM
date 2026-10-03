# 这个文件是 eval 包的统一导出入口。
"""Evaluation: metrics and video-level inference with explanations.

- ``metrics.py``   —— frame / clip / video 级 AUC & AP
- ``inference.py`` —— 视频级滑窗推理 + template-based 可解释性
"""

from .metrics import (
    clip_level_metrics,
    frame_level_metrics,
    video_level_metrics,
)
from .inference import evaluate_videos

__all__ = [
    # metrics
    "frame_level_metrics",
    "clip_level_metrics",
    "video_level_metrics",
    # inference
    "evaluate_videos",
]
