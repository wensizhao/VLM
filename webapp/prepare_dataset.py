#!/usr/bin/env python3
"""Prepare an uploaded Avenue-style dataset for the existing training pipeline.

Keep legacy Studio metadata for video-only uploads, then generate motion labels
and optional CLIP features. The upstream loaders use explicit video/mask paths.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.io import savemat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SPLITS = ("training", "testing")


def _video_metadata(path: Path) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: " + str(path))
    try:
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if frames <= 0:
        raise RuntimeError("Video has no readable frames: " + str(path))
    return frames, height, width


def ensure_placeholder_volumes(root: Path) -> None:
    """Create a tiny (1, 1, N) vol variable when an upload has no annotations.

    Legacy Studio readiness checks use this metadata. It is not ground truth
    and is never passed as mask_path to the upstream loaders. Existing real
    masks are never modified.
    """
    for split in SPLITS:
        video_dir = root / (split + "_videos")
        if not video_dir.is_dir():
            continue
        mask_dir = root / (split + "_vol")
        mask_dir.mkdir(parents=True, exist_ok=True)
        for video_path in sorted(video_dir.glob("*.avi")):
            mask_path = mask_dir / ("vol" + video_path.stem + ".mat")
            if mask_path.is_file():
                continue
            frames, _height, _width = _video_metadata(video_path)
            savemat(mask_path, {"vol": np.zeros((1, 1, frames), dtype=np.uint8)})
            print(
                "[prep] placeholder vol " + split + "/" + video_path.stem
                + " frames=" + str(frames),
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an uploaded dataset")
    parser.add_argument("--root", required=True, help="Avenue-style dataset root")
    parser.add_argument("--labels", dest="labels", action="store_true", default=True)
    parser.add_argument("--no-labels", dest="labels", action="store_false")
    parser.add_argument("--features", action="store_true", default=False)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--threshold", type=float, default=3.0)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not (root / "training_videos").is_dir():
        raise SystemExit("Not an Avenue-style dataset: " + str(root))
    if not sorted((root / "training_videos").glob("*.avi")):
        raise SystemExit(
            "No .avi videos found under training_videos. The current training "
            "pipeline follows Avenue's .avi naming; transcode other formats first."
        )

    print("[prep] dataset root " + str(root), flush=True)
    ensure_placeholder_volumes(root)

    if args.labels:
        from tools.extract_motion_labels import extract as extract_labels
        for split in SPLITS:
            if not (root / (split + "_videos")).is_dir():
                continue
            print("[prep] motion labels " + split, flush=True)
            extract_labels(
                root=root,
                split=split,
                threshold=float(args.threshold),
                out_dir=root / "labels_v2",
            )

    if args.features:
        import torch
        from tools.extract_video_features import extract as extract_features
        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        for split in SPLITS:
            if not (root / (split + "_videos")).is_dir():
                continue
            print("[prep] CLIP features " + split + " on " + device, flush=True)
            extract_features(
                video_dir=root / (split + "_videos"),
                mask_dir=None,
                split=split,
                out_dir=root / "features_v2" / split,
                batch_size=32,
                device=device,
            )

    print("[prep] done", flush=True)


if __name__ == "__main__":
    main()
