# 这个文件是 data 包的向后兼容导出入口——实际实现已迁移至 datasets/。
"""Dataset and dataloader modules — re-exports from ``datasets/`` for backward compat."""

from datasets.video_dataset import VideoDataset
from datasets.dataloader import (
    build_video_dataset,
    build_video_dataloader,
    build_feature_dataset,
    build_feature_dataloader,
    seed_worker,
)

# 保持旧命名的兼容别名
AvenueDataset = VideoDataset
build_avenue_dataset = build_video_dataset
build_avenue_dataloader = build_video_dataloader

__all__ = [
    "VideoDataset",
    "AvenueDataset",
    "build_video_dataset",
    "build_video_dataloader",
    "build_feature_dataset",
    "build_feature_dataloader",
    "build_avenue_dataset",
    "build_avenue_dataloader",
    "seed_worker",
]
