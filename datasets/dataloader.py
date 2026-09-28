# 这个文件负责为 VideoDataset 和 FeatureDataset 构建 DataLoader。
"""Dataloader builders for video and feature datasets."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .video_dataset import VideoBackend, VideoDataset
from .feature_dataset import FeatureDataset


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python RNGs from PyTorch worker state."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ═════════════════════════════════════════════════════════════════
# VideoDataset builders
# ═════════════════════════════════════════════════════════════════

def build_video_dataset(
    video_dir: str | Path,
    mask_dir: str | Path | None = None,
    split: str = "testing",
    clip_length: int = 16,
    clip_stride: int = 1,
    clip_step: int | None = None,
    image_size: tuple[int, int] | None = (224, 224),
    video_backend: VideoBackend = "auto",
    resize_mask_to_video: bool = False,
    include_last_clip: bool = True,
    pad_short_clips: bool = False,
) -> VideoDataset:
    """Construct a :class:`VideoDataset`."""
    return VideoDataset(
        video_dir=video_dir, mask_dir=mask_dir, split=split,
        clip_length=clip_length, clip_stride=clip_stride, clip_step=clip_step,
        image_size=image_size, video_backend=video_backend,
        resize_mask_to_video=resize_mask_to_video,
        include_last_clip=include_last_clip, pad_short_clips=pad_short_clips,
    )


def build_video_dataloader(
    video_dir: str | Path,
    mask_dir: str | Path | None = None,
    split: str = "testing",
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    seed: int | None = None,
    **dataset_kwargs: Any,
) -> DataLoader:
    """Build a DataLoader for :class:`VideoDataset`."""
    dataset = build_video_dataset(
        video_dir=video_dir, mask_dir=mask_dir, split=split,
        **dataset_kwargs,
    )

    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        worker_init_fn = seed_worker

    return DataLoader(
        dataset,
        batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn, generator=generator,
    )


# ═════════════════════════════════════════════════════════════════
# FeatureDataset builders
# ═════════════════════════════════════════════════════════════════

def build_feature_dataset(
    feature_dir: str | Path,
    video_dir: str | Path,
    mask_dir: str | Path | None = None,
    split: str = "testing",
    clip_length: int = 16,
    clip_stride: int = 1,
    clip_step: int | None = None,
    include_last_clip: bool = True,
    pad_short_clips: bool = False,
    preload: bool = True,
    allow_missing: bool = False,
) -> FeatureDataset:
    """Construct a :class:`FeatureDataset`."""
    return FeatureDataset(
        feature_dir=feature_dir, video_dir=video_dir, mask_dir=mask_dir,
        split=split,
        clip_length=clip_length, clip_stride=clip_stride, clip_step=clip_step,
        include_last_clip=include_last_clip, pad_short_clips=pad_short_clips,
        preload=preload, allow_missing=allow_missing,
    )


def build_feature_dataloader(
    feature_dir: str | Path,
    video_dir: str | Path,
    mask_dir: str | Path | None = None,
    split: str = "testing",
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    seed: int | None = None,
    allow_missing: bool = False,
    **dataset_kwargs: Any,
) -> DataLoader:
    """Build a DataLoader for :class:`FeatureDataset`."""
    dataset = build_feature_dataset(
        feature_dir=feature_dir, video_dir=video_dir, mask_dir=mask_dir,
        split=split,
        allow_missing=allow_missing, **dataset_kwargs,
    )

    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        worker_init_fn = seed_worker

    return DataLoader(
        dataset,
        batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn, generator=generator,
    )


# ═════════════════════════════════════════════════════════════════
# Backward-compat aliases
# ═════════════════════════════════════════════════════════════════

# 保持旧 data/ 模块的接口兼容
build_avenue_dataset = build_video_dataset
build_avenue_dataloader = build_video_dataloader
