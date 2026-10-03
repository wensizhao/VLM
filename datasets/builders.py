# 这个文件为指定 split 构建 DataLoader——自动在 VideoDataset 与 FeatureDataset 之间切换。
"""Dataloader builders — auto-select VideoDataset (pixels) or FeatureDataset (.pt).

判断逻辑（按优先级）：
    1. ``data_cfg.use_feature = true`` → 强制 FeatureDataset
    2. ``{feature_dir}/{split}`` 目录存在 → 自动 FeatureDataset（推荐：快 ~10x）
    3. 否则 → VideoDataset（每次实时解码视频帧）

两种 Dataset 返回的 batch 字典差异：
    VideoDataset   → {"video": (B,T,C,H,W), ...}
    FeatureDataset → {"vis_feat": (B,T,D), ...}
Trainer 和 eval 会根据 key 自动选 forward 路径（见 train/trainer.py / eval/inference.py）。
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig
from torch.utils.data import DataLoader

from datasets.dataloader import (
    build_feature_dataloader,
    build_video_dataloader,
)


def _feature_dir_exists(data_cfg: DictConfig, split: str) -> bool:
    """判断指定 split 的预提取特征目录是否存在。"""
    fd = data_cfg.get("feature_dir", None)
    if not fd:
        return False
    return (Path(fd) / split).is_dir()


def build_split_dataloader(
    data_cfg: DictConfig,
    split: str,
    shuffle: bool,
    seed: int | None = None,
) -> tuple[DataLoader, str]:
    """为某个 split 构建 DataLoader。

    Args:
        data_cfg: ``cfg.data`` 配置段。
        split: "training" 或 "testing"。
        shuffle: 是否打乱（训练 True，评估 False）。
        seed: 传给 dataloader 固定 shuffle 顺序（可复现）。

    Returns:
        (dataloader, source_type)，source_type ∈ {"feature", "video"}，
        供上层决定走 forward_from_visual 还是 forward。
    """
    root = str(data_cfg.root)
    common = dict(
        clip_length=int(data_cfg.get("clip_length", 16)),
        clip_stride=int(data_cfg.get("clip_stride", 1)),
        clip_step=int(data_cfg.get("clip_step", 16)),
        include_last_clip=bool(data_cfg.get("include_last_clip", True)),
    )

    use_feature = bool(data_cfg.get("use_feature", False)) or _feature_dir_exists(
        data_cfg, split
    )

    if use_feature:
        feature_dir = Path(data_cfg.feature_dir) / split
        loader = build_feature_dataloader(
            feature_dir=feature_dir,
            root=root,
            split=split,
            batch_size=int(data_cfg.get("batch_size", 2)),
            shuffle=shuffle,
            num_workers=int(data_cfg.get("num_workers", 0)),
            seed=seed,
            allow_missing=bool(data_cfg.get("allow_missing", False)),
            label_dir=(
                Path(data_cfg.label_dir) / split
                if data_cfg.get("label_dir", None) is not None else None
            ),
            **common,
        )
        return loader, "feature"

    loader = build_video_dataloader(
        root=root,
        split=split,
        batch_size=int(data_cfg.get("batch_size", 2)),
        shuffle=shuffle,
        num_workers=int(data_cfg.get("num_workers", 0)),
        seed=seed,
        image_size=tuple(data_cfg.get("image_size", (224, 224))),
        video_backend=str(data_cfg.get("video_backend", "auto")),
        frame_label_mode=str(data_cfg.get("frame_label_mode", "pixel")),
        motion_threshold=float(data_cfg.get("motion_threshold", 3.0)),
        **common,
    )
    return loader, "video"
