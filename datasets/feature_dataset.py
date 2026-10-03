# 这个文件负责读取预提取的 CLIP 视觉特征——跳过视频解码和 ViT 编码。
"""Feature dataset — reads pre-extracted CLIP features from .pt files.

搭配 :func:`tools.extract_video_features` 使用：
先离线把所有视频帧过一遍 ViT，存为 per-video ``(N, D)`` tensor，
然后用 FeatureDataset 直接加载特征，训练速度提升 10×。

与 :class:`VideoDataset` 的对比::

    VideoDataset   → 读视频 → 解码 → ViT encode → (T, D)   ← 慢，可训练 backbone
    FeatureDataset → 读 .pt → 切片      →           (T, D)   ← 快，backbone 冻结时用
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .video_dataset import (
    ClipRecord,
    VideoRecord,
    _build_frame_indices,
    _compute_clip_starts,
    _load_mask_volume,
    _normalize_split,
    _resolve_avenue_root,
)


# ═════════════════════════════════════════════════════════════════
# FeatureDataset
# ═════════════════════════════════════════════════════════════════

class FeatureDataset(Dataset[dict[str, Any]]):
    """读取预提取的帧级 CLIP 特征 + 原始标注。

    每个 .pt 文件对应一个视频，shape ``(N, D)`` — 逐帧按顺序排列。
    对每个 clip 从中切片出 T 帧，完全跳过视频解码和 ViT 前向。

    特征文件命名约定::

        {feature_dir}/{split}/{video_id}.pt   →  torch.Size([N, D])

    每个文件就是一个 ``(N, D)`` float32 tensor，按帧顺序排列。
    N 必须与对应视频的总帧数一致。

    Usage::

        ds = FeatureDataset(
            feature_dir="./data/features/training",
            root="./data",
            split="training",
            clip_length=16,
        )
        sample = ds[0]  # {"vis_feat": (16, 512), "frame_label": (16,), ...}
    """

    def __init__(
        self,
        feature_dir: str | Path,
        root: str | Path,
        split: str = "training",
        clip_length: int = 16,
        clip_stride: int = 1,
        clip_step: int | None = None,
        include_last_clip: bool = True,
        pad_short_clips: bool = False,
        preload: bool = True,
        allow_missing: bool = False,
        label_dir: str | Path | None = None,
    ) -> None:
        """初始化 FeatureDataset。

        Args:
            feature_dir: 提取好的 .pt 特征文件目录。
            root: Avenue 数据集根目录（用于读取 mask 标注）。
            split: training 或 testing。
            clip_length / clip_stride / clip_step: clip 切片参数，
                与 VideoDataset 保持一致。
            include_last_clip / pad_short_clips: 同 VideoDataset。
            preload: True 时一次性将所有 .pt 加载到内存（推荐），
                False 时按需读取（内存紧张时用）。
            allow_missing: True 时跳过没有对应 .pt 特征文件的视频
                （支持"只提取了一部分视频"的场景，如冒烟测试/增量提取）。
            label_dir: 可选——预计算的帧级标签目录 {label_dir}/{video_id}.pt。
                用于无法从特征反推标签、且真实 pixel mask 不可用时的
                运动伪标签路径（配合 tools/extract_motion_labels.py）。
        """
        if clip_length <= 0:
            raise ValueError("clip_length must be positive")
        if clip_stride <= 0:
            raise ValueError("clip_stride must be positive")

        self.root = _resolve_avenue_root(root)
        self.split = _normalize_split(split)
        self.feature_dir = Path(feature_dir)
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.clip_step = clip_step if clip_step is not None else clip_length
        self.include_last_clip = include_last_clip
        self.pad_short_clips = pad_short_clips
        self.preload = preload
        self.allow_missing = allow_missing
        self.label_dir = Path(label_dir) if label_dir is not None else None

        # 构建与 VideoDataset 完全相同的视频索引和 clip 索引
        self.video_records = self._build_video_records()
        if self.allow_missing:
            # 过滤掉没有特征文件的视频（特征只提取了一部分时）
            self.video_records = [
                r for r in self.video_records
                if (self.feature_dir / f"{r.video_id}.pt").is_file()
            ]
            if not self.video_records:
                raise RuntimeError(
                    f"No feature files found under {self.feature_dir} for split={split!r}"
                )
        self._features = self._load_features()

        # 验证特征帧数与视频帧数一致
        self._validate_feature_counts()

        self.clip_records = self._build_clip_records()
        if not self.clip_records:
            raise RuntimeError(
                "No clips generated. Check clip_length/clip_stride/clip_step."
            )

    # ── Build indices (mirrors VideoDataset) ──────────────────

    def _build_video_records(self) -> list[VideoRecord]:
        from .video_dataset import _read_video_metadata, _read_mask_metadata

        video_dir = self.root / f"{self.split}_videos"
        mask_dir = self.root / f"{self.split}_vol"
        video_paths = sorted(video_dir.glob("*.avi"))
        if not video_paths:
            raise FileNotFoundError(f"No AVI videos found in {video_dir}")

        records: list[VideoRecord] = []
        for vp in video_paths:
            vid = vp.stem
            mp = mask_dir / f"vol{vid}.mat"
            if not mp.is_file():
                raise FileNotFoundError(f"Missing annotation: {mp.name}")

            n, fps, fh, fw = _read_video_metadata(vp)
            mh, mw, mf = _read_mask_metadata(mp)
            if n != mf:
                raise RuntimeError(
                    f"Frame count mismatch: {vp.name}={n}, {mp.name}={mf}"
                )
            records.append(VideoRecord(
                video_id=vid, split=self.split, video_path=vp, mask_path=mp,
                num_frames=n, fps=fps, frame_height=fh, frame_width=fw,
                mask_height=mh, mask_width=mw,
            ))
        return records

    def _load_features(self) -> dict[str, torch.Tensor]:
        """加载所有 .pt 特征文件。

        Returns:
            dict[video_id] → Tensor (N, D)，按原始帧顺序排列。
        """
        features: dict[str, torch.Tensor] = {}
        for vr in self.video_records:
            pt_path = self.feature_dir / f"{vr.video_id}.pt"
            if not pt_path.is_file():
                raise FileNotFoundError(
                    f"Feature file not found: {pt_path}. "
                    f"Run tools/extract_video_features.py first."
                )
            if self.preload:
                feat = torch.load(pt_path, map_location="cpu", weights_only=True)
                if not isinstance(feat, torch.Tensor):
                    # 兼容 dict 格式：{"features": ..., "video_id": ...}
                    if isinstance(feat, dict):
                        feat = feat["features"]
                    else:
                        raise TypeError(
                            f"Expected tensor or dict in {pt_path}, got {type(feat)}"
                        )
                features[vr.video_id] = feat
            else:
                features[vr.video_id] = pt_path  # 延迟加载，存路径
        return features

    def _validate_feature_counts(self) -> None:
        """确保每个视频的特征帧数 == 视频真实帧数。"""
        for vr in self.video_records:
            feat = self._resolve_feature(vr.video_id)
            if feat.shape[0] != vr.num_frames:
                raise RuntimeError(
                    f"Feature frame count mismatch for {vr.video_id}: "
                    f"features={feat.shape[0]}, video={vr.num_frames}"
                )

    def _resolve_feature(self, video_id: str) -> torch.Tensor:
        """获取某个视频的特征 tensor（处理 preload/延迟加载）。"""
        entry = self._features[video_id]
        if isinstance(entry, torch.Tensor):
            return entry
        # 延迟加载
        feat = torch.load(entry, map_location="cpu", weights_only=True)
        if isinstance(feat, dict):
            feat = feat["features"]
        return feat

    def _build_clip_records(self) -> list[ClipRecord]:
        span = (self.clip_length - 1) * self.clip_stride + 1
        records: list[ClipRecord] = []
        for vi, vr in enumerate(self.video_records):
            starts = _compute_clip_starts(
                vr.num_frames, span, self.clip_step,
                self.include_last_clip, self.pad_short_clips,
            )
            records.extend(ClipRecord(video_index=vi, start_frame=s) for s in starts)
        return records

    # ── Dataset API ───────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.clip_records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        clip = self.clip_records[index]
        vr = self.video_records[clip.video_index]

        # 帧下标 —— 与 VideoDataset 完全相同的逻辑
        indices = _build_frame_indices(
            clip.start_frame, self.clip_length, self.clip_stride, vr.num_frames,
        )

        # 从预提取特征中切片：(N, D) → (T, D)
        feat = self._resolve_feature(vr.video_id)
        vis_feat = feat[indices]   # (T, D) — 直接就是 CLIP 编码后的特征

        # 标注 —— 优先用预计算标签文件，否则用像素 mask
        if self.label_dir is not None:
            label_path = self.label_dir / f"{vr.video_id}.pt"
            if not label_path.is_file():
                raise FileNotFoundError(
                    f"Label file not found: {label_path}. "
                    f"Run tools/extract_motion_labels.py first."
                )
            labels = torch.load(label_path, map_location="cpu", weights_only=True)
            frame_label = labels[indices].to(torch.long)
            clip_masks = frame_label.unsqueeze(1).float()  # (T, 1) 占位
        else:
            mask_vol = _load_mask_volume(str(vr.mask_path))
            clip_masks = torch.from_numpy(mask_vol[indices].copy()).to(torch.float32)
            frame_label = (clip_masks.flatten(1).amax(dim=1) > 0).to(torch.long)

        return {
            "vis_feat": vis_feat,                # (T, D) — backbone 编码后的特征
            "frame_label": frame_label,          # (T,)
            "pixel_mask": clip_masks,            # (T, H, W)
            "clip_label": frame_label.amax(),    # scalar
            "video_id": vr.video_id,
            "start_frame": clip.start_frame,
        }
