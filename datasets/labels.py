"""Validated label IO: pseudo labels never silently replace pixel annotations."""
from pathlib import Path
import hashlib

import torch


def load_frame_labels(path: Path, num_frames: int) -> torch.Tensor:
    labels = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(labels, dict):
        labels = labels["labels"]
    if not isinstance(labels, torch.Tensor) or labels.shape != (num_frames,):
        raise ValueError(f"Label shape must be ({num_frames},) in {path}")
    if not torch.isfinite(labels).all() or not ((labels == 0) | (labels == 1)).all():
        raise ValueError(f"Labels must be finite binary values in {path}")
    return labels.long()


def official_video_records(root: Path, split: str, directory: Path | None) -> tuple[list, dict]:
    """Official frame annotations do not depend on Avenue's grayscale vol files."""
    from .cache import validate_official_cache
    from .video_dataset import VideoRecord, _read_video_metadata

    if split != "testing" or directory is None:
        raise ValueError("Official Avenue annotations require testing split and label_dir")
    meta = validate_official_cache(directory)
    videos = sorted((root / "testing_videos").glob("*.avi"))
    if not videos or {p.stem for p in videos} != set(meta.get("videos", {})):
        raise ValueError("Official manifest/video ID mismatch")
    records = []
    for video in videos:
        vid = video.stem
        n, fps, h, w = _read_video_metadata(video)
        annotation = meta["videos"][vid]
        label_path = directory / f"{vid}.pt"
        if annotation.get("num_frames") != n or annotation.get("mask_shape") != [h, w]:
            raise ValueError(f"Official annotation/video frame count or resolution mismatch: {vid}")
        if hashlib.sha256(label_path.read_bytes()).hexdigest() != annotation.get("label_sha256"):
            raise ValueError(f"Official label checksum mismatch: {vid}")
        labels = load_frame_labels(label_path, n)
        if int(labels.sum()) != annotation.get("num_anomaly_frames"):
            raise ValueError(f"Official anomaly count mismatch: {vid}")
        records.append(VideoRecord(video_id=vid, split=split, video_path=video, mask_path=label_path,
                                   num_frames=n, fps=fps, frame_height=h, frame_width=w,
                                   mask_height=h, mask_width=w))
    return records, meta
