"""Load encoder-verified training features without consuming anomaly labels."""
from __future__ import annotations

import math
from pathlib import Path
import random

import torch

from .cache import validate_feature_cache
from .video_dataset import _read_video_metadata, _resolve_avenue_root


def _resolve_dataset_root(data_cfg) -> Path:
    """按 Design A 的显式 split 段反推数据集根目录，缺省回退 data_cfg.root。

    configs/experiment.yaml 用 data.train.video_path 声明视频目录，没有顶层
    的 data.root；而 normality 协议需要遍历 training_videos，所以从 video_path
    的父目录得到根目录。兼容旧的 data.root 写法（测试里用 SimpleNamespace(root=...)）。
    """
    getter = getattr(data_cfg, "get", None)
    if callable(getter):
        try:
            from .builders import _resolve_split_paths

            _, video_path, _ = _resolve_split_paths(data_cfg, "training")
            return Path(video_path).expanduser().resolve().parent
        except Exception:
            pass
    root = getattr(data_cfg, "root", None)
    if root is None:
        raise ValueError(
            "Cannot resolve the Avenue dataset root: set data.train.video_path "
            "(Design A) or data.root."
        )
    return _resolve_avenue_root(root)

def load_normal_training_features(data_cfg, backbone_cfg, seed: int) -> tuple[dict, dict, dict]:
    root = _resolve_dataset_root(data_cfg)
    directory = Path(data_cfg.feature_dir) / "training"
    validate_feature_cache(directory, backbone_cfg)
    videos = sorted((root / "training_videos").glob("*.avi"))
    fraction = float(data_cfg.validation_fraction)
    if len(videos) < 2 or not 0 < fraction < 1:
        raise ValueError("Normality fitting needs at least two training videos and a valid holdout fraction")
    ids = [p.stem for p in videos]
    random.Random(seed).shuffle(ids)
    count = min(len(ids) - 1, max(1, math.ceil(len(ids) * fraction)))
    held_out = set(ids[:count])
    training, calibration = {}, {}
    for video in videos:
        n, _, _, _ = _read_video_metadata(video)
        feat = torch.load(directory / f"{video.stem}.pt", map_location="cpu", weights_only=True)
        if isinstance(feat, dict):
            feat = feat["features"]
        if not isinstance(feat, torch.Tensor) or feat.ndim != 2 or len(feat) != n or not torch.isfinite(feat).all():
            raise ValueError(f"Invalid normal-training features: {video.stem}")
        if (feat.norm(dim=-1) <= 0).any():
            raise ValueError(f"Zero training feature: {video.stem}")
        (calibration if video.stem in held_out else training)[video.stem] = feat.float()
    protocol = {
        "mode": "normal_feature_memory", "source_split": "training", "seed": seed,
        "fit_video_ids": sorted(training), "calibration_video_ids": sorted(calibration),
        "normality_assumption": "training videos are predominantly normal; outliers may be present",
        "training_frame_ground_truth_available": False, "motion_labels_used": False,
        "test_labels_used_for_training": False, "test_labels_used_for_calibration": False,
        "test_labels_used_for_model_selection": False,
    }
    return training, calibration, protocol
