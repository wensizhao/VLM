# 这个文件负责构建数据集对象和 DataLoader 对象。
"""Dataloader builders for clip-based video anomaly datasets."""

# 让类型注解延迟解析，减少前向引用问题。
from __future__ import annotations

# random 是 Python 标准库中的随机数模块。
import random
# Path 用于处理文件路径。
from pathlib import Path
# Any 表示任意类型，这里主要给 **dataset_kwargs 使用。
from typing import Any

# NumPy 需要在 dataloader worker 中设置随机种子。
import numpy as np
# PyTorch 用于随机数生成器和 DataLoader。
import torch
# DataLoader 用于按 batch 迭代数据集。
from torch.utils.data import DataLoader

# 从同目录的 avenue_dataset 模块导入数据集类和相关类型别名。
from .avenue_dataset import AvenueDataset, VideoBackend


# 这个函数会在每个 dataloader worker 进程启动时运行一次。
def seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python RNGs from PyTorch worker state."""

    del worker_id  # 当前函数里并不需要直接使用 worker_id，所以显式删除避免误会。
    worker_seed = torch.initial_seed() % (2**32)  # 取 PyTorch 分配给当前 worker 的种子，并限制到 NumPy 可接受的范围。
    np.random.seed(worker_seed)  # 设置 NumPy 的随机种子。
    random.seed(worker_seed)  # 设置 Python 标准库 random 的随机种子。


# 这个函数负责创建并返回 AvenueDataset 实例。
def build_avenue_dataset(
    root: str | Path,  # 数据集根目录。
    split: str = "training",  # 使用 training 还是 testing。
    clip_length: int = 16,  # 每个 clip 采样多少帧。
    clip_stride: int = 1,  # clip 内部采样帧之间的步长。
    clip_step: int | None = None,  # 相邻 clip 起始位置的步长。
    image_size: tuple[int, int] | None = (224, 224),  # 是否把视频帧缩放到固定大小。
    video_backend: VideoBackend = "auto",  # 视频读取后端。
    resize_mask_to_video: bool = False,  # 是否把 mask 缩放到视频帧大小。
    include_last_clip: bool = True,  # 是否补上最后一个贴近视频结尾的 clip。
    pad_short_clips: bool = False,  # 视频过短时是否仍然构造一个 clip。
) -> AvenueDataset:  # 返回 AvenueDataset 对象。
    """Construct an Avenue clip dataset with a reproducible interface."""

    return AvenueDataset(  # 直接把参数透传给 AvenueDataset，保持接口清晰一致。
        root=root,  # 数据集根目录。
        split=split,  # 训练/测试划分。
        clip_length=clip_length,  # 每个 clip 采样的帧数。
        clip_stride=clip_stride,  # clip 内部帧间隔。
        clip_step=clip_step,  # 相邻 clip 的移动步长。
        image_size=image_size,  # 输出图像大小。
        video_backend=video_backend,  # 视频读取后端。
        resize_mask_to_video=resize_mask_to_video,  # 是否同步缩放 mask。
        include_last_clip=include_last_clip,  # 是否补最后一个 clip。
        pad_short_clips=pad_short_clips,  # 短视频是否补 clip。
    )


# 这个函数负责创建 DataLoader，供训练或评估循环按 batch 读取数据。
def build_avenue_dataloader(
    root: str | Path,  # 数据集根目录。
    split: str = "training",  # 选择训练集或测试集。
    batch_size: int = 4,  # 每个 batch 包含多少个样本。
    shuffle: bool = True,  # 是否在每个 epoch 打乱样本顺序。
    num_workers: int = 0,  # 使用多少个子进程并行读取数据。
    pin_memory: bool = True,  # 是否启用固定内存，通常对 GPU 训练更友好。
    drop_last: bool = False,  # 最后一个 batch 不足 batch_size 时是否丢弃。
    seed: int | None = None,  # 如果提供种子，就让采样顺序和 worker 随机行为可复现。
    **dataset_kwargs: Any,  # 其余参数继续传给 build_avenue_dataset。
) -> DataLoader:  # 返回构建好的 PyTorch DataLoader。
    """Build a DataLoader for the Avenue dataset."""

    dataset = build_avenue_dataset(root=root, split=split, **dataset_kwargs)  # 先构建底层数据集对象。

    generator = None  # 默认不显式设置 PyTorch 随机数生成器。
    worker_init_fn = None  # 默认不设置 worker 初始化函数。
    if seed is not None:  # 如果用户希望可复现。
        generator = torch.Generator()  # 创建一个独立的 PyTorch 随机数生成器。
        generator.manual_seed(seed)  # 给这个生成器设置固定种子。
        worker_init_fn = seed_worker  # 同时为每个 worker 设置 NumPy 和 random 的种子。

    return DataLoader(  # 构建并返回 DataLoader。
        dataset,  # 要被按 batch 读取的数据集对象。
        batch_size=batch_size,  # 每个 batch 的样本数量。
        shuffle=shuffle,  # 是否打乱样本顺序。
        num_workers=num_workers,  # 并行加载数据的子进程数量。
        pin_memory=pin_memory,  # 是否使用固定内存以提升 GPU 搬运效率。
        drop_last=drop_last,  # 最后一个不完整 batch 是否丢弃。
        persistent_workers=num_workers > 0,  # 只要启用了 worker 进程，就让它们在 epoch 之间常驻，减少反复创建的开销。
        worker_init_fn=worker_init_fn,  # 每个 worker 启动时如何设置随机种子。
        generator=generator,  # 控制 shuffle 等随机行为的生成器。
    )
