#!/usr/bin/env python3
# 这个脚本离线计算所有视频的运动伪标签（frame-level），供 FeatureDataset 快速路径使用。
"""Extract motion-based frame labels for all videos in a split.

背景：FeatureDataset 只读 CLIP 特征（没有像素），无法自己算运动。此脚本
提前把每帧的运动标签算好存成 .pt，FeatureDataset 通过 ``label_dir`` 直接加载。

用法::

    python tools/extract_motion_labels.py \\
        --root ./data --split testing \\
        --out ./data/labels/testing --threshold 3.0

数学（motion-based anomaly baseline）：
    motion_t = (1/HW) · Σ|I_t - I_{t-1}|          # 帧间平均绝对差
    label_t  = 1[motion_t > threshold]             # 超过阈值视为异常
异常事件（投掷/奔跑/打架）通常伴随剧烈运动，这是 VAD 最经典的 baseline 之一。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from datasets.video_dataset import (
    VideoDataset,
    _compute_motion_labels,
    _read_frames_opencv,
)


def extract(
    root: str | Path,
    split: str = "testing",
    threshold: float = 3.0,
    out_dir: str | Path = "./data/labels",
) -> None:
    """为 split 下所有视频生成运动伪标签并保存。

    Args:
        root: Avenue 数据集根目录。
        split: training / testing。
        threshold: 运动阈值（0-255 灰度差尺度）。
        out_dir: 输出目录，每个视频存为 {video_id}.pt（形状 (N,) int64）。
    """
    out_path = Path(out_dir) / split
    out_path.mkdir(parents=True, exist_ok=True)

    # 用 VideoDataset 的元数据能力（视频列表、帧数），不切 clip
    ds = VideoDataset(root=root, split=split, clip_length=16, image_size=None)
    for vr in ds.video_records:
        # 逐段读取所有帧并计算运动标签
        labels: list[np.ndarray] = []
        n = vr.num_frames
        step = 256  # 每次最多读 256 帧，避免一次性占满内存
        for start in range(0, n, step):
            end = min(start + step, n)
            indices = np.arange(start, end, dtype=np.int64)
            frames = _read_frames_opencv(vr.video_path, indices)  # (T,H,W,C) uint8
            labels.append(_compute_motion_labels(frames, threshold))

        label = np.concatenate(labels)  # (N,)
        torch.save(torch.from_numpy(label), out_path / f"{vr.video_id}.pt")
        print(f"saved {split}/{vr.video_id}.pt  shape={label.shape}  "
              f"#anomaly={int(label.sum())}/{n}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract motion-based frame labels")
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", default="testing")
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--out", default="./data/labels")
    args = parser.parse_args()
    extract(root=args.root, split=args.split,
            threshold=args.threshold, out_dir=args.out)


if __name__ == "__main__":
    main()
