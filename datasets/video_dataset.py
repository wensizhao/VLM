# 这个文件提供基于原始视频帧的数据集——每次 __getitem__ 从视频文件读取并解码帧。
"""Video frame dataset — reads and decodes raw frames from video files at query time."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader
from scipy.io import loadmat, whosmat
from torch.utils.data import Dataset

VideoBackend = Literal["auto", "decord", "opencv"]
Split = Literal["training", "testing"]
FrameLabelMode = Literal["pixel", "motion_diff"]

# 已提醒过的"可疑 mask"视频 id（避免每个 clip 都刷一次 warning）
_WARNED_SUSPECT_MASKS: set[str] = set()


# ═════════════════════════════════════════════════════════════════
# Metadata records
# ═════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class VideoRecord:
    """Metadata for a single video and its annotation file."""

    video_id: str
    split: Split
    video_path: Path
    mask_path: Path
    num_frames: int
    fps: float
    frame_height: int
    frame_width: int
    mask_height: int
    mask_width: int


@dataclass(frozen=True)
class ClipRecord:
    """A temporal clip extracted from a video."""

    video_index: int
    start_frame: int


# ═════════════════════════════════════════════════════════════════
# Helpers — path / metadata / index
# ═════════════════════════════════════════════════════════════════

def _normalize_split(split: str) -> Split:
    normalized = split.strip().lower()
    if normalized in {"train", "training"}:
        return "training"
    if normalized in {"test", "testing", "eval", "evaluation"}:
        return "testing"
    raise ValueError(f"Unsupported split: {split!r}")


def _resolve_avenue_root(root: str | Path) -> Path:
    root_path = Path(root).expanduser().resolve()
    if (root_path / "training_videos").is_dir():
        return root_path
    nested = root_path / "Avenue_Dataset"
    if (nested / "training_videos").is_dir():
        return nested
    raise FileNotFoundError(
        f"Could not locate Avenue dataset under {root_path}. "
        f"Expected training_videos/testing_videos directories."
    )


def _read_video_metadata(video_path: Path) -> tuple[int, float, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    if n <= 0:
        raise RuntimeError(f"Video has no readable frames: {video_path}")
    return n, fps, h, w


def _read_mask_metadata(mask_path: Path) -> tuple[int, int, int]:
    for name, shape, _ in whosmat(mask_path):
        if name == "vol":
            if len(shape) != 3:
                raise RuntimeError(
                    f"Expected 'vol' to have 3 dims in {mask_path}, got {shape}"
                )
            return int(shape[0]), int(shape[1]), int(shape[2])
    raise KeyError(f"Could not find 'vol' variable in {mask_path}")


@lru_cache(maxsize=8)
def _load_mask_volume(mask_path: str) -> np.ndarray:
    data = loadmat(mask_path)
    if "vol" not in data:
        raise KeyError(f"Could not find 'vol' variable in {mask_path}")
    volume = np.asarray(data["vol"])
    if volume.ndim != 3:
        raise RuntimeError(f"Expected 'vol' to have 3 dims, got {volume.shape}")

    # 在二值化之前检测：某些数据包的 vol 实为灰度视频帧而非 mask
    if mask_path not in _WARNED_SUSPECT_MASKS and mask_looks_like_frames(volume):
        _WARNED_SUSPECT_MASKS.add(mask_path)
        import logging
        logging.getLogger(__name__).warning(
            "Mask %s looks like grayscale frames (not a binary mask). "
            "Set frame_label_mode='motion_diff' or download the real masks.",
            mask_path,
        )

    binary = (volume > 0).astype(np.float32)
    return np.transpose(binary, (2, 0, 1))  # (H, W, T) → (T, H, W)


def mask_looks_like_frames(volume: np.ndarray) -> bool:
    """启发式检测：某些 Avenue 数据包把"灰度视频帧"误存成了 vol 变量。

    判断依据（两个特征同时成立则高度可疑）：
        1. 几乎所有像素都非零（帧是连续的灰度值，真 mask 大部分是 0）
        2. 值域很宽且连续（真 mask 只有 0 / 255 两个值）

    用户遇到此情况应：
        a) 重新下载官方 binary mask；或
        b) 设置 ``frame_label_mode: motion_diff`` 用运动伪标签跑通流程。
    """
    flat = volume.reshape(-1)
    if flat.size == 0:
        return False
    nonzero_frac = float((flat > 0).mean())
    # 帧图像是 uint8 灰度（~上百个不同值）；真二值 mask 只有 2 个值
    unique_frac = float(np.unique(flat).size) / min(flat.size, 1_000_000)
    return nonzero_frac > 0.99 and unique_frac > 1e-4


def _compute_motion_labels(frames: np.ndarray, threshold: float) -> np.ndarray:
    """基于帧间差的运动伪标签（classic motion-based VAD baseline）。

    数学：帧间运动强度 = 相邻两帧灰度图的平均绝对差（Mean Absolute Difference）：
        motion_t = (1/HW) · Σ_{i,j} |I_t(i,j) - I_{t-1}(i,j)|
    异常通常伴随剧烈运动（投掷、奔跑、打架），所以
        label_t = 1[motion_t > threshold]
    第一帧没有前帧，用第二帧的 motion 补齐。

    注意：这是一个**基准近似**（pseudo label），学术结论请使用真实的
    pixel mask（frame_label_mode="pixel"）。
    """
    gray = frames.mean(axis=-1).astype(np.float32)          # (T, H, W) in [0,255]
    diff = np.abs(np.diff(gray, axis=0)).mean(axis=(1, 2))  # (T-1,)
    motion = np.concatenate([[diff[0]], diff])              # (T,)
    return (motion > threshold).astype(np.int64)


def _compute_clip_starts(
    num_frames: int,
    clip_span: int,
    clip_step: int,
    include_last_clip: bool,
    pad_short_clips: bool,
) -> list[int]:
    if num_frames < clip_span:
        return [0] if pad_short_clips else []
    starts = list(range(0, num_frames - clip_span + 1, clip_step))
    last = num_frames - clip_span
    if include_last_clip and starts and starts[-1] != last:
        starts.append(last)
    return starts


def _build_frame_indices(
    start_frame: int,
    clip_length: int,
    clip_stride: int,
    num_frames: int,
) -> np.ndarray:
    indices = start_frame + np.arange(clip_length, dtype=np.int64) * clip_stride
    if num_frames <= 0:
        raise RuntimeError("num_frames must be positive")
    return np.clip(indices, 0, num_frames - 1)


# ═════════════════════════════════════════════════════════════════
# Frame readers
# ═════════════════════════════════════════════════════════════════

def _read_frames_decord(video_path: Path, indices: np.ndarray) -> np.ndarray:
    reader = VideoReader(str(video_path), num_threads=1)
    return reader.get_batch(indices.tolist()).asnumpy()  # (T, H, W, C)


def _read_frames_opencv(video_path: Path, indices: np.ndarray) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frames: list[np.ndarray] = []
    cursor = -1
    try:
        for idx in indices:
            if cursor != int(idx):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Failed to read frame {int(idx)} from {video_path}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cursor = int(idx) + 1
    finally:
        cap.release()
    return np.stack(frames, axis=0)


# ═════════════════════════════════════════════════════════════════
# VideoDataset
# ═════════════════════════════════════════════════════════════════

class VideoDataset(Dataset[dict[str, Any]]):
    """Clip-based dataset — reads raw video frames from disk at query time.

    Each ``__getitem__`` opens the video file, decodes T frames, normalizes,
    and returns ``{video, frame_label, pixel_mask, clip_label, video_id, start_frame}``.

    Use when:
    - Backbone is trainable (needs raw pixels for gradient flow)
    - You haven't pre-extracted features yet
    - You're running a quick sanity check without caching

    Switch to :class:`FeatureDataset` when the backbone is frozen for fast iteration.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "training",
        clip_length: int = 16,
        clip_stride: int = 1,
        clip_step: int | None = None,
        image_size: tuple[int, int] | None = (224, 224),
        video_backend: VideoBackend = "auto",
        resize_mask_to_video: bool = False,
        include_last_clip: bool = True,
        pad_short_clips: bool = False,
        frame_label_mode: FrameLabelMode = "pixel",
        motion_threshold: float = 3.0,
    ) -> None:
        if clip_length <= 0:
            raise ValueError("clip_length must be positive")
        if clip_stride <= 0:
            raise ValueError("clip_stride must be positive")
        if clip_step is not None and clip_step <= 0:
            raise ValueError("clip_step must be positive")
        if image_size is not None and (
            len(image_size) != 2 or image_size[0] <= 0 or image_size[1] <= 0
        ):
            raise ValueError("image_size must be a (height, width) tuple")
        if video_backend not in {"auto", "decord", "opencv"}:
            raise ValueError(f"Unsupported video_backend: {video_backend!r}")
        if frame_label_mode not in {"pixel", "motion_diff"}:
            raise ValueError(
                f"Unsupported frame_label_mode: {frame_label_mode!r} "
                "(expected 'pixel' or 'motion_diff')"
            )

        self.root = _resolve_avenue_root(root)
        self.split = _normalize_split(split)
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.clip_step = clip_step if clip_step is not None else clip_length
        self.image_size = image_size
        self.video_backend = video_backend
        self.resize_mask_to_video = resize_mask_to_video
        self.include_last_clip = include_last_clip
        self.pad_short_clips = pad_short_clips
        self.frame_label_mode = frame_label_mode
        self.motion_threshold = motion_threshold

        self.video_records = self._build_video_records()
        self.clip_records = self._build_clip_records()

        if not self.clip_records:
            raise RuntimeError(
                "No clips generated. Check clip_length/clip_stride/clip_step "
                "against dataset frame counts."
            )

    # ── Build indices ─────────────────────────────────────────

    def _build_video_records(self) -> list[VideoRecord]:
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

    # ── Frame reading ─────────────────────────────────────────

    def _read_video_frames(self, path: Path, indices: np.ndarray) -> np.ndarray:
        if self.video_backend in {"auto", "decord"}:
            try:
                return _read_frames_decord(path, indices)
            except Exception:
                if self.video_backend == "decord":
                    raise
        return _read_frames_opencv(path, indices)

    # ── Dataset API ───────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.clip_records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        clip = self.clip_records[index]
        vr = self.video_records[clip.video_index]

        indices = _build_frame_indices(
            clip.start_frame, self.clip_length, self.clip_stride, vr.num_frames,
        )
        raw = self._read_video_frames(vr.video_path, indices)          # (T,H,W,C)
        video = torch.from_numpy(raw.copy()).permute(0, 3, 1, 2).contiguous()  # (T,C,H,W)
        video = video.to(torch.float32) / 255.0

        if self.image_size is not None:
            video = F.interpolate(video, size=self.image_size,
                                  mode="bilinear", align_corners=False)

        mask_vol = _load_mask_volume(str(vr.mask_path))                # (T_m, H, W)
        if self.frame_label_mode == "pixel":
            # 可疑 mask（实为视频帧）的提醒已在 _load_mask_volume 内完成
            clip_masks = torch.from_numpy(mask_vol[indices].copy()).to(torch.float32)
            frame_label = (clip_masks.flatten(1).amax(dim=1) > 0).to(torch.long)
        else:  # motion_diff —— 基于帧间差的运动伪标签
            frame_label = torch.from_numpy(
                _compute_motion_labels(raw, self.motion_threshold)
            )
            # 无像素级标注：pixel_mask 退化为帧标签占位，保持返回 dict 结构一致
            clip_masks = frame_label.unsqueeze(1).float()  # (T, 1)

        if self.resize_mask_to_video and self.image_size is not None:
            clip_masks = F.interpolate(
                clip_masks.unsqueeze(1), size=self.image_size, mode="nearest",
            ).squeeze(1)

        return {
            "video": video,                      # (T, C, H, W)
            "frame_label": frame_label,          # (T,)
            "pixel_mask": clip_masks,            # (T, H, W)
            "clip_label": frame_label.amax(),    # scalar
            "video_id": vr.video_id,
            "start_frame": clip.start_frame,
        }
