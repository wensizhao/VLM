#!/usr/bin/env python3
"""Verify Avenue's official ZIP and derive binary frame labels from its masks."""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from datasets.annotations import frame_labels_from_mat
from datasets.video_dataset import _read_video_metadata, _resolve_avenue_root
from utils.io import save_json

SOURCE_URL = "https://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/ground_truth_demo.zip"


def prepare_labels(archive: Path, root: Path, output: Path) -> dict:
    """Validate every member before publishing a new testing-label directory."""
    target = output / "testing"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite labels: {target}")
    root = _resolve_avenue_root(root)
    videos = {p.stem: p for p in sorted((root / "testing_videos").glob("*.avi"))}
    if not videos:
        raise ValueError("No testing videos found")
    parsed = {}
    with ZipFile(archive) as package:
        if package.testzip() is not None:
            raise ValueError("Corrupt ground-truth archive")
        members = {}
        for member in package.infolist():
            match = re.search(r"(?:^|/)testing_label_mask/(\d+)_label\.mat$", member.filename)
            if match:
                vid = f"{int(match.group(1)):02d}"
                if vid in members:
                    raise ValueError(f"Duplicate annotation ID: {vid}")
                members[vid] = member
        if set(members) != set(videos):
            raise ValueError(f"Annotation/video ID mismatch: missing={sorted(set(videos)-set(members))}, "
                             f"extra={sorted(set(members)-set(videos))}")
        for vid, video in videos.items():
            n, _, h, w = _read_video_metadata(video)
            content = package.read(members[vid])
            labels, details = frame_labels_from_mat(content, n)
            if details["mask_shape"] != [h, w]:
                raise ValueError(f"Mask/video resolution mismatch for {vid}: {details['mask_shape']} != {[h,w]}")
            parsed[vid] = (torch.from_numpy(labels), {
                **details, "num_frames": n, "num_anomaly_frames": int(labels.sum()),
                "archive_member": members[vid].filename,
                "mat_sha256": hashlib.sha256(content).hexdigest(),
            })

    meta = {
        "version": 1, "label_mode": "official", "split": "testing",
        "annotation_type": "pixel_mask_derived_frame_labels",
        "frame_index_base": 0, "reduction": "any_nonzero_pixel",
        "frame_mapping": "MATLAB volLabel{ii} -> Python frame ii-1",
        "source": {"url": SOURCE_URL, "archive_name": archive.name,
                   "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()},
        "num_videos": len(parsed), "num_frames": sum(v[1]["num_frames"] for v in parsed.values()),
        "num_anomaly_frames": sum(v[1]["num_anomaly_frames"] for v in parsed.values()),
        "videos": {},
    }
    target.mkdir(parents=True)
    for vid, (labels, details) in parsed.items():
        path = target / f"{vid}.pt"
        torch.save(labels, path)
        meta["videos"][vid] = {**details, "label_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    save_json(meta, target / "manifest.json")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    meta = prepare_labels(args.archive, args.root, args.out)
    print(f"Verified {meta['num_videos']} videos / {meta['num_frames']} frames / "
          f"{meta['num_anomaly_frames']} abnormal frames. Manifest: {args.out}/testing/manifest.json")


if __name__ == "__main__":
    main()
