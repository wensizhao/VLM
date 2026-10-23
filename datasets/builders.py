# 这个文件为指定 split 构建 DataLoader——按 data.<split> 段解析路径，自动切换 Dataset。
"""Dataloader builders — auto-select VideoDataset (pixels) or FeatureDataset (.pt).

路径解析（Design A）
-------------------
``configs/experiment.yaml`` 里每个 split 显式声明自己的路径：

    data:
      train: {video_path: ..., mask_path: null}
      test:  {video_path: ..., mask_path: ...}

``build_split_dataloader(data_cfg, split)`` 先把 ``split`` 归一化成
``training`` / ``testing``，再映射到 ``train`` / ``test`` 段读取
``video_path`` / ``mask_path``。找不到对应段时回退到顶层
``data.video_path`` / ``data.mask_path``（向后兼容）。

Dataset 类型判断（按优先级）
----------------------------
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
from datasets.video_dataset import _normalize_split


# 归一化 split 名 → 允许的配置段名（Design A 用 train/test，兼容 training/testing）
_SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "training": ("train", "training"),
    "testing": ("test", "testing"),
}


def _feature_dir_exists(data_cfg: DictConfig, split: str) -> bool:
    """判断指定 split 的预提取特征目录是否存在。"""
    fd = data_cfg.get("feature_dir", None)
    if not fd:
        return False
    return any((Path(fd) / split).glob("*.pt"))


def _resolve_split_paths(
    data_cfg: DictConfig, split: str
) -> tuple[str, str, str | None]:
    """把 split 解析成 (归一化 split 名, video_path, mask_path)。

    解析顺序：
        1. ``data.<train|test>`` 段（Design A，推荐）
        2. 顶层 ``data.video_path`` / ``data.mask_path``（向后兼容）

    Args:
        data_cfg: ``cfg.data`` 配置段。
        split: "training" / "testing"（或其别名 train / test / eval）。

    Returns:
        (normalized_split, video_path, mask_path)，mask_path 缺省为 None。

    Raises:
        ValueError: 对应 split 段缺失，或段内未填写 ``video_path`` 时。
    """
    normalized = _normalize_split(split)

    # 1) Design A：显式 per-split 段
    for key in _SECTION_ALIASES[normalized]:
        section = data_cfg.get(key, None)
        if section is None:
            continue
        video_path = section.get("video_path", None)
        if video_path is None or str(video_path).strip() == "":
            raise ValueError(
                f"config.data.{key}.video_path 未填写：请显式提供视频目录路径，"
                f"例如 data/Avenue_Dataset/testing_videos。"
            )
        mask_path = section.get("mask_path", None)
        mask_path = str(mask_path) if mask_path and str(mask_path).strip() else None
        return normalized, str(video_path), mask_path

    # 2) 向后兼容：顶层 data.video_path / data.mask_path
    video_path = data_cfg.get("video_path", None)
    if video_path is not None and str(video_path).strip() != "":
        mask_path = data_cfg.get("mask_path", None)
        mask_path = str(mask_path) if mask_path and str(mask_path).strip() else None
        return normalized, str(video_path), mask_path

    raise ValueError(
        f"config.data 缺少 {normalized!r} 段（也没有顶层 video_path）："
        f"请在 experiment.yaml 的 data 下声明 train/test 段及其 video_path。"
    )


def build_split_dataloader(
    data_cfg: DictConfig,
    split: str,
    shuffle: bool,
    seed: int | None = None,
    backbone_cfg: DictConfig | None = None,
) -> tuple[DataLoader, str]:
    """为某个 split 构建 DataLoader。

    Args:
        data_cfg: ``cfg.data`` 配置段（含 ``train`` / ``test`` 子段）。
        split: "training" 或 "testing"（或其别名）。
        shuffle: 是否打乱（训练 True，评估 False）。
        seed: 传给 dataloader 固定 shuffle 顺序（可复现）。

    Returns:
        (dataloader, source_type)，source_type ∈ {"feature", "video"}，
        供上层决定走 forward_from_visual 还是 forward。
    """
    normalized, video_path, mask_path = _resolve_split_paths(data_cfg, split)
    if data_cfg.get("frame_label_mode") == "motion_diff" and data_cfg.get("label_dir"):
        from .cache import validate_motion_cache
        validate_motion_cache(Path(data_cfg.label_dir) / normalized,
                              float(data_cfg.get("motion_threshold", 3.0)))
    common = dict(
        clip_length=int(data_cfg.get("clip_length", 16)),
        clip_stride=int(data_cfg.get("clip_stride", 1)),
        clip_step=int(data_cfg.get("clip_step", 16)),
        include_last_clip=bool(data_cfg.get("include_last_clip", True)),
        pad_short_clips=bool(data_cfg.get("pad_short_clips", False)),
    )

    # use_feature: "auto"（默认，看目录是否存在）/ true / false
    requested = data_cfg.get("use_feature", "auto")
    if requested not in ("auto", True, False):
        raise ValueError("data.use_feature must be auto, true or false")
    use_feature = _feature_dir_exists(data_cfg, normalized) if requested == "auto" else bool(requested)

    if use_feature:
        feature_dir = Path(data_cfg.feature_dir) / normalized
        if backbone_cfg is not None:
            from .cache import validate_feature_cache
            validate_feature_cache(feature_dir, backbone_cfg)
        loader = build_feature_dataloader(
            feature_dir=feature_dir,
            video_dir=video_path,
            mask_dir=mask_path,
            split=normalized,
            batch_size=int(data_cfg.get("batch_size", 2)),
            shuffle=shuffle,
            num_workers=int(data_cfg.get("num_workers", 0)),
            seed=seed,
            allow_missing=bool(data_cfg.get("allow_missing", False)),
            frame_label_mode=str(data_cfg.get("frame_label_mode", "pixel")),
            label_dir=(
                Path(data_cfg.label_dir) / normalized
                if data_cfg.get("label_dir", None) is not None else None
            ),
            **common,
        )
        return loader, "feature"

    loader = build_video_dataloader(
        video_dir=video_path,
        mask_dir=mask_path,
        split=normalized,
        batch_size=int(data_cfg.get("batch_size", 2)),
        shuffle=shuffle,
        num_workers=int(data_cfg.get("num_workers", 0)),
        seed=seed,
        image_size=(tuple(data_cfg.image_size) if data_cfg.get("image_size") is not None else None),
        video_backend=str(data_cfg.get("video_backend", "auto")),
        frame_label_mode=str(data_cfg.get("frame_label_mode", "pixel")),
        motion_threshold=float(data_cfg.get("motion_threshold", 3.0)),
        label_dir=(Path(data_cfg.label_dir) / normalized
                   if data_cfg.get("label_dir", None) is not None else None),
        **common,
    )
    return loader, "video"
