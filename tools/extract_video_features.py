#!/usr/bin/env python3
# 这个脚本离线提取所有视频帧的 CLIP 视觉特征——一次性跑完，后续训练直接用 FeatureDataset 加载。
"""Extract per-frame CLIP visual features for all videos in a dataset split.

Run once before training with a frozen backbone::

    python tools/extract_video_features.py \\
        --root ./data \\
        --split training \\
        --model ViT-B-32 \\
        --out ./data/features/training \\
        --batch-size 64

Output: one ``{video_id}.pt`` per video, shape ``(N, D)``, float32, L2-normalized.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.backbone import CLIPBackbone
from datasets.video_dataset import VideoDataset


# ═════════════════════════════════════════════════════════════════
# Frame-only dataset wrapper (no clip slicing — all frames)
# ═════════════════════════════════════════════════════════════════

class _FrameIterator:
    """遍历某个 split 下所有视频的全部帧，不做 clip 切片。

    每个视频作为一个"样本"，返回全部帧 tensor。
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "training",
        image_size: tuple[int, int] = (224, 224),
    ) -> None:
        # 用 VideoDataset 的元数据能力（视频列表、帧数），但不用它的 clip 索引
        from datasets.video_dataset import (
            _resolve_avenue_root,
            _normalize_split,
            VideoRecord,
            _read_video_metadata,
            _read_mask_metadata,
        )
        self.root = _resolve_avenue_root(root)
        self.split = _normalize_split(split)
        self.image_size = image_size

        # 构建视频元数据列表
        video_dir = self.root / f"{self.split}_videos"
        mask_dir = self.root / f"{self.split}_vol"
        paths = sorted(video_dir.glob("*.avi"))
        if not paths:
            raise FileNotFoundError(f"No videos in {video_dir}")

        self.records: list[VideoRecord] = []
        for vp in paths:
            vid = vp.stem
            mp = mask_dir / f"vol{vid}.mat"
            if not mp.is_file():
                raise FileNotFoundError(f"Missing annotation: {mp.name}")
            n, fps, fh, fw = _read_video_metadata(vp)
            mh, mw, mf = _read_mask_metadata(mp)
            if n != mf:
                raise RuntimeError(f"Frame mismatch: {vp.name}={n}, {mp.name}={mf}")
            self.records.append(VideoRecord(
                video_id=vid, split=self.split, video_path=vp, mask_path=mp,
                num_frames=n, fps=fps, frame_height=fh, frame_width=fw,
                mask_height=mh, mask_width=mw,
            ))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor]:
        """返回 (video_id, frames_tensor (N, C, H, W))。"""
        from datasets.video_dataset import _read_frames_decord, _read_frames_opencv

        vr = self.records[idx]
        indices = np.arange(vr.num_frames, dtype=np.int64)
        try:
            raw = _read_frames_decord(vr.video_path, indices)
        except Exception:
            raw = _read_frames_opencv(vr.video_path, indices)

        video = torch.from_numpy(raw.copy()).permute(0, 3, 1, 2).contiguous()
        video = video.to(torch.float32) / 255.0
        if self.image_size is not None:
            import torch.nn.functional as F
            video = F.interpolate(
                video, size=self.image_size, mode="bilinear", align_corners=False,
            )
        return vr.video_id, video


# ═════════════════════════════════════════════════════════════════
# Main extraction
# ═════════════════════════════════════════════════════════════════

def extract(
    root: str | Path,
    split: str,
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    out_dir: str | Path = "./data/features",
    batch_size: int = 64,
    image_size: tuple[int, int] = (224, 224),
    device: str = "cuda",
) -> None:
    """对所有视频的所有帧逐帧编码并保存为 .pt 文件。"""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. 加载 backbone (eval, no grad)
    print(f"[1/4] Loading backbone: {model_name} ({pretrained})")
    backbone = CLIPBackbone(model_name=model_name, pretrained=pretrained)
    backbone.eval()
    backbone.to(device)

    # 2. 构建逐帧数据集
    print(f"[2/4] Scanning videos from {root}/{split}_videos")
    iterator = _FrameIterator(root=root, split=split, image_size=image_size)
    print(f"       Found {len(iterator)} videos, "
          f"{sum(r.num_frames for r in iterator.records):,} total frames")

    # 3. 逐视频编码
    print(f"[3/4] Extracting features (batch_size={batch_size})...")
    with torch.no_grad():
        for idx in tqdm(range(len(iterator)), desc="Videos", unit="video"):
            video_id, frames = iterator[idx]  # frames: (N, C, H, W)
            N = frames.shape[0]

            # 分批过 ViT
            all_feats: list[torch.Tensor] = []
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                chunk = frames[start:end].to(device)  # (B, C, H, W)
                feats = backbone._clip.encode_image(chunk, normalize=True)  # (B, D)
                all_feats.append(feats.cpu())

            features = torch.cat(all_feats, dim=0)  # (N, D)

            # 保存
            save_path = out_path / f"{video_id}.pt"
            torch.save(features, save_path)

    # 4. 完成
    total_size = sum(
        f.stat().st_size for f in out_path.glob("*.pt")
    )
    print(f"[4/4] Done! Saved {len(iterator)} files to {out_path} "
          f"({total_size / 1024 / 1024:.1f} MB)")


# ═════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-frame CLIP visual features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--root", required=True, help="Dataset root directory")
    parser.add_argument("--split", default="training", help="training | testing")
    parser.add_argument("--model", default="ViT-B-32", help="CLIP model name")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--out", default="./data/features", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()
    extract(
        root=args.root,
        split=args.split,
        model_name=args.model,
        pretrained=args.pretrained,
        out_dir=args.out,
        batch_size=args.batch_size,
        image_size=tuple(args.image_size),
        device=args.device,
    )


if __name__ == "__main__":
    main()
