"""Cache identities prevent comparisons using features from another encoder."""
import json
import re
from pathlib import Path
from typing import Any


FEATURE_VERSION = "clip_rgb_preprocess_v2"


def validate_feature_cache(directory: Path, backbone_cfg: Any) -> None:
    if backbone_cfg.get("provider", "clip") != "clip":
        raise ValueError("CLIP feature manifests cannot validate another encoder provider; use pixel data")
    manifest = directory / "manifest.json"
    if not manifest.is_file():
        raise ValueError(f"Unverified legacy feature cache: {directory}. "
                         "Extract a new cache with tools/extract_video_features.py")
    meta = json.loads(manifest.read_text())
    expected = {"model_name": str(backbone_cfg.name),
                "pretrained": str(backbone_cfg.pretrained), "preprocessing": FEATURE_VERSION}
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"Feature cache mismatch: {key}={meta.get(key)!r}, expected {value!r}")
    if not backbone_cfg.get("freeze_vision", True):
        raise ValueError("Feature caches cannot be used with trainable vision encoders")


def validate_motion_cache(directory: Path, threshold: float) -> None:
    manifest = directory / "manifest.json"
    if not manifest.is_file():
        raise ValueError(f"Unverified motion labels in {directory}; regenerate with extract_motion_labels.py")
    meta = json.loads(manifest.read_text())
    if meta.get("version") != 2 or meta.get("label_mode") != "motion_diff" or meta.get("threshold") != threshold:
        raise ValueError(f"Motion cache threshold/version mismatch in {manifest}")


def validate_official_cache(directory: Path) -> dict:
    manifest = directory / "manifest.json"
    if not manifest.is_file():
        raise ValueError(f"Missing official label manifest: {manifest}; run prepare_official_labels.py")
    meta = json.loads(manifest.read_text())
    if (meta.get("version") != 1 or meta.get("label_mode") != "official"
            or meta.get("split") != "testing" or meta.get("frame_index_base") != 0
            or meta.get("reduction") != "any_nonzero_pixel"
            or meta.get("annotation_type") != "pixel_mask_derived_frame_labels"
            or not re.fullmatch(r"[0-9a-f]{64}", str(meta.get("source", {}).get("sha256", "")))):
        raise ValueError(f"Invalid official label provenance: {manifest}")
    videos = meta.get("videos", {})
    if (not videos or meta.get("num_videos") != len(videos)
            or meta.get("num_frames") != sum(v.get("num_frames", 0) for v in videos.values())
            or meta.get("num_anomaly_frames") != sum(v.get("num_anomaly_frames", 0) for v in videos.values())):
        raise ValueError(f"Official manifest totals disagree: {manifest}")
    return meta
