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
from scipy.io import loadmat
from torch.utils.data import Dataset

VideoBackend = Literal["auto", "decord", "opencv"]
Split = Literal["training", "testing"]
# Metadata records
# ═════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class VideoRecord:
    """Metadata for a single video and its annotation file."""

    video_id: str
    split: Split
    video_path: Path
    mask_path: Path | None  # 该视频的 GT mask；未配置/不存在时为 None（正常视频）
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


def _resolve_dir(path: str | Path, what: str) -> Path:
    """把配置里的路径转成绝对路径，并校验目录存在。

    Args:
        path: 配置给出的目录路径（绝对或相对 cwd）。
        what: 出错信息里的名称（如 video_path / mask_path）。

    Returns:
        解析后的绝对路径。
    """
    if path is None or str(path).strip() == "":
        raise ValueError(
            f"{what} 不能为空：请在 config 中显式提供该路径"
        )
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{what} 目录不存在：{resolved}")
    return resolved


def _find_mask_path(mask_dir: Path | None, video_id: str) -> Path | None:
    """在 mask 目录里找视频对应的 GT mask 文件。

    约定（可自行调整命名）：``video 01.avi`` ↔ ``1_label.mat`` 或 ``01_label.mat``。
    匹配顺序：``{video_id}_label.mat`` → ``{int(video_id)}_label.mat``。
    找不到（含 mask_dir 未配置）→ 返回 None，表示该视频为正常视频。
    """
    if mask_dir is None:
        return None
    stems = [video_id]
    if video_id.isdigit():
        stems.append(str(int(video_id)))
    for stem in stems:
        candidate = mask_dir / f"{stem}_label.mat"
        if candidate.is_file():
            return candidate
    return None


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
    data = loadmat(mask_path)

    # 检查变量名
    if "volLabel" not in data:
        raise KeyError(f"Could not find 'volLabel' variable in {mask_path}")
    
    volLabel = data["volLabel"]
    
    # 检查数据结构
    if volLabel.ndim != 2 or volLabel.shape[0] != 1:
        raise RuntimeError(
            f"Expected volLabel to have shape (1, N), got {volLabel.shape}"
        )
    
    # 获取帧数、高度、宽度
    num_frames = volLabel.shape[1]
    height = volLabel[0, 0].shape[0]  # 第一帧的高度
    width = volLabel[0, 0].shape[1]   # 第一帧的宽度
    
    # 验证所有帧的尺寸一致
    for i in range(1, min(num_frames, 10)):  # 检查前10帧
        frame = volLabel[0, i]
        if frame.shape != (height, width):
            raise RuntimeError(
                f"Inconsistent frame sizes in {mask_path}: "
                f"frame 0 is {height}x{width}, frame {i} is {frame.shape}"
            )
    
    return height, width, num_frames

@lru_cache(maxsize=8)
def _load_mask_volume(mask_path: str) -> np.ndarray:
    data = loadmat(mask_path)
    if "volLabel" not in data:
        raise KeyError(f"Could not find 'volLabel' variable in {mask_path}")

    volLabel = data["volLabel"]
    # 将object数组转换为3D数组 (T, H, W)
    volume = np.stack([volLabel[0, i] for i in range(volLabel.shape[1])])

    # 官方 GT 是二值 mask（0/255）→ 归一化成 0/1
    return (volume > 0).astype(np.float32)  # (T, H, W)


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
    
    frames: list[np.ndarray] =[]
    current_pos = -1 
    
    try:
        for target_frame in indices:
            if current_pos != target_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                current_pos = target_frame 
            
            #（cap.read() 会自动移动到下一帧
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Failed to read frame {target_frame}")
            
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            current_pos += 1  
            
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

        # video_dir 必填：未提供直接报错；mask_dir 可选：未提供 = 全正常视频
        self.video_dir = _resolve_dir(video_dir, "video_path")
        self.mask_dir = (
            _resolve_dir(mask_dir, "mask_path")
            if mask_dir is not None and str(mask_dir).strip() != ""
            else None
        )
        self.split = _normalize_split(split)
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.clip_step = clip_step if clip_step is not None else clip_length
        self.image_size = image_size
        self.video_backend = video_backend
        self.resize_mask_to_video = resize_mask_to_video  # 调整标注 mask 到视频尺寸
        self.include_last_clip = include_last_clip        # 是否包含最后一个不完整的 clip
        self.pad_short_clips = pad_short_clips            # 是否填充短于 clip 长度的视频

        # 构建视频和剪辑的元数据索引
        self.video_records = self._build_video_records()   # 视频元数据列表
        self.clip_records = self._build_clip_records()     # 剪辑元数据列表

        if not self.clip_records:
            raise RuntimeError(
                "No clips generated. Check clip_length/clip_stride/clip_step "
                "against dataset frame counts."
            )

    # ── Build indices ─────────────────────────────────────────

    def _build_video_records(self) -> list[VideoRecord]:
        # 从显式配置的 video_dir 扫描视频；mask_dir 可选。
        # mask_dir 未配置（或某个视频没有对应 mask 文件）→ 该视频视为正常视频。
        video_paths = sorted(self.video_dir.glob("*.avi"))
        if not video_paths:
            raise FileNotFoundError(f"No AVI videos found in {self.video_dir}")

        records: list[VideoRecord] = []
        for vp in video_paths:
            vid = vp.stem
            n, fps, fh, fw = _read_video_metadata(vp)
            mp = _find_mask_path(self.mask_dir, vid)
            if mp is not None:
                mh, mw, mf = _read_mask_metadata(mp)
                if n != mf:
                    raise RuntimeError(
                        f"Frame count mismatch: {vp.name}={n}, {mp.name}={mf}"
                    )
            else:
                mh, mw, mf = fh, fw, n
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

        if vr.mask_path is None:
            # 无 GT mask（mask_dir 未配置或该视频无 mask 文件）→ 视为正常视频
            frame_label = torch.zeros(len(indices), dtype=torch.long)
            clip_masks = frame_label.unsqueeze(1).float()  # (T, 1) 占位
        else:
            mask_vol = _load_mask_volume(str(vr.mask_path))            # (T_m, H, W)
            clip_masks = torch.from_numpy(mask_vol[indices].copy()).to(torch.float32)
            frame_label = (clip_masks.flatten(1).amax(dim=1) > 0).to(torch.long)

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
