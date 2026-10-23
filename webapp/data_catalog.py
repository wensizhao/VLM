"""Dataset discovery, inspection, upload and preparation commands."""
from __future__ import annotations

import json
import re
import shutil
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .config import (
    DATA_ROOT,
    OFFICIAL_LABEL_ROOT,
    PROJECT_ROOT,
    UPLOAD_ROOT,
    ensure_runtime_dirs,
    is_within,
    rel_to_project,
    sanitize_dataset_name,
)

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_VIDEO_SUFFIXES = {".avi", ".mp4", ".mkv", ".mov"}

_meta_lock = threading.Lock()
_meta_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
_label_lock = threading.Lock()
_label_cache: dict[tuple[str, str, int], dict[str, Any] | None] = {}
_thumb_lock = threading.Lock()
_thumb_cache: dict[tuple[str, str, str, int, int], bytes] = {}
_dataset_cache_lock = threading.Lock()
_dataset_cache: tuple[float, list[dict[str, Any]]] | None = None


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _candidate_roots() -> list[Path]:
    roots: set[Path] = set()
    if not DATA_ROOT.is_dir():
        return []
    for split_dir in list(DATA_ROOT.rglob("training_videos")) + list(DATA_ROOT.rglob("testing_videos")):
        if not split_dir.is_dir():
            continue
        root = split_dir.parent.resolve()
        try:
            rel_parts = root.relative_to(DATA_ROOT.resolve()).parts
        except ValueError:
            continue
        if any(part.startswith(".") for part in rel_parts):
            continue
        if any(part in {"__pycache__", ".git"} for part in rel_parts):
            continue
        roots.add(root)
    return sorted(roots, key=lambda p: str(p))


def _dataset_id(root: Path) -> str:
    return root.resolve().relative_to(DATA_ROOT.resolve()).as_posix()


def _find_dataset_root(target: Path) -> Path | None:
    if (target / "training_videos").is_dir():
        return target
    for child in target.iterdir():
        if child.is_dir() and (child / "training_videos").is_dir():
            return child
    for child in target.rglob("training_videos"):
        if not child.is_dir():
            continue
        try:
            depth = len(child.relative_to(target).parts)
        except ValueError:
            continue
        if depth <= 4:
            return child.parent
    return None


def _is_builtin_avenue(root: Path) -> bool:
    """Only the bundled data/Avenue_Dataset may inherit the shared caches."""
    try:
        return root.resolve() == (DATA_ROOT / "Avenue_Dataset").resolve()
    except Exception:
        return False


def _label_root(root: Path) -> Path | None:
    local = root / "labels_v2"
    if local.is_dir():
        return local
    if _is_builtin_avenue(root):
        shared = DATA_ROOT / "labels_v2"
        if shared.is_dir():
            return shared
    return None


def _feature_root(root: Path) -> Path | None:
    local = root / "features_v2"
    if local.is_dir():
        return local
    if _is_builtin_avenue(root):
        shared = DATA_ROOT / "features_v2"
        if shared.is_dir():
            return shared
    return None


def _official_root(root: Path) -> Path | None:
    if not _is_builtin_avenue(root):
        return None
    return OFFICIAL_LABEL_ROOT if OFFICIAL_LABEL_ROOT.is_dir() else None


def _split_videos(root: Path, split: str) -> list[Path]:
    directory = root / (split + "_videos")
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in _VIDEO_SUFFIXES
    )


def _has_volumes(root: Path, split: str, video_count: int) -> bool:
    if video_count == 0:
        return False
    mask_dir = root / (split + "_vol")
    if not mask_dir.is_dir():
        return False
    return len(list(mask_dir.glob("vol*.mat"))) >= video_count


def _split_labels(root: Path, split: str) -> Path | None:
    label_root = _label_root(root)
    if label_root is not None and (label_root / split).is_dir():
        return label_root / split
    official = _official_root(root)
    if official is not None and split == "testing" and (official / split).is_dir():
        return official / split
    return None


def _split_features(root: Path, split: str) -> Path | None:
    feature_root = _feature_root(root)
    if feature_root is not None and (feature_root / split).is_dir():
        return feature_root / split
    return None


def _video_meta(path: Path) -> dict[str, Any]:
    stat = path.stat()
    key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    with _meta_lock:
        cached = _meta_cache.get(key)
    if cached is not None:
        return dict(cached)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: " + str(path))
    try:
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    info = {
        "num_frames": frames,
        "fps": round(fps, 3),
        "width": width,
        "height": height,
        "size_mb": round(stat.st_size / 1024 / 1024, 2),
    }
    with _meta_lock:
        _meta_cache[key] = dict(info)
    return info


def _label_info(root: Path, split: str, video_id: str, num_frames: int) -> dict[str, Any] | None:
    label_dir = _split_labels(root, split)
    if label_dir is None:
        return None
    path = label_dir / (video_id + ".pt")
    if not path.is_file():
        return None
    key = (str(path.resolve()), split, path.stat().st_mtime_ns)
    with _label_lock:
        cached = _label_cache.get(key)
    if cached is not None:
        return dict(cached)
    try:
        import torch
        value = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(value, dict):
            value = value.get("labels", value)
        count = int(value.sum().item()) if hasattr(value, "sum") else None
        total = int(value.numel()) if hasattr(value, "numel") else None
        result = {
            "source": "official" if "labels_official" in str(path) else "motion_diff",
            "anomaly_frames": count,
            "num_frames": total,
            "matches_video": total == num_frames,
        }
    except Exception:
        result = {"source": "unknown", "anomaly_frames": None, "num_frames": None, "matches_video": False}
    with _label_lock:
        _label_cache[key] = dict(result)
    return result


# ---------------------------------------------------------------------------
# Public summaries
# ---------------------------------------------------------------------------

def summarize_dataset(root: Path) -> dict[str, Any]:
    root = root.resolve()
    splits: dict[str, Any] = {}
    total_frames = 0
    for split in ("training", "testing"):
        videos = _split_videos(root, split)
        frames = 0
        for path in videos:
            try:
                frames += int(_video_meta(path)["num_frames"])
            except Exception:
                pass
        total_frames += frames
        has_labels = _split_labels(root, split) is not None
        has_features = _split_features(root, split) is not None
        splits[split] = {
            "count": len(videos),
            "frames": frames,
            "has_volumes": _has_volumes(root, split, len(videos)),
            "has_labels": has_labels,
            "has_features": has_features,
        }

    training = splits["training"]
    has_masks = training["has_volumes"]
    has_labels = training["has_labels"]
    has_features = training["has_features"]
    ready = training["count"] >= 2 and has_masks
    needs_prep = training["count"] >= 2 and (not has_masks or not has_labels)
    feature_ready = ready and has_labels and has_features

    dataset_id = _dataset_id(root)
    official_ready = _official_root(root) is not None and splits["testing"]["count"] > 0
    label_root = _label_root(root)
    feature_root = _feature_root(root)

    return {
        "id": dataset_id,
        "name": root.name,
        "path": rel_to_project(root),
        "config_root": rel_to_project(root),
        "splits": splits,
        "total_videos": training["count"] + splits["testing"]["count"],
        "total_frames": total_frames,
        "has_masks": has_masks,
        "has_labels": has_labels,
        "has_features": has_features,
        "ready": ready,
        "needs_prep": needs_prep,
        "feature_ready": feature_ready,
        "official_ready": official_ready,
        "label_root": rel_to_project(label_root) if label_root else None,
        "feature_root": rel_to_project(feature_root) if feature_root else None,
        "label_mode": "motion_diff",
        "updated_at": time.time(),
    }


def list_datasets(force: bool = False) -> list[dict[str, Any]]:
    global _dataset_cache
    now = time.time()
    with _dataset_cache_lock:
        if not force and _dataset_cache is not None and now - _dataset_cache[0] < 5.0:
            return [dict(item) for item in _dataset_cache[1]]
    datasets: list[dict[str, Any]] = []
    for root in _candidate_roots():
        try:
            datasets.append(summarize_dataset(root))
        except Exception as exc:
            datasets.append({
                "id": _dataset_id(root),
                "name": root.name,
                "path": rel_to_project(root),
                "error": str(exc),
            })
    datasets.sort(key=lambda item: (not item.get("ready", False), item.get("name", "")))
    with _dataset_cache_lock:
        _dataset_cache = (now, datasets)
    return [dict(item) for item in datasets]


def get_dataset(dataset_id: str) -> tuple[Path, dict[str, Any]]:
    target = (DATA_ROOT / dataset_id).resolve()
    if not is_within(target, DATA_ROOT):
        raise ValueError("Dataset path escapes data root")
    if not (target / "training_videos").is_dir():
        found = _find_dataset_root(target) if target.is_dir() else None
        if found is None:
            raise FileNotFoundError("Dataset not found: " + dataset_id)
        target = found
    return target, summarize_dataset(target)


def dataset_detail(dataset_id: str, limit_per_split: int = 500) -> dict[str, Any]:
    root, summary = get_dataset(dataset_id)
    videos: list[dict[str, Any]] = []
    for split in ("training", "testing"):
        for path in _split_videos(root, split):
            if sum(1 for item in videos if item["split"] == split) >= limit_per_split:
                break
            try:
                meta = _video_meta(path)
            except Exception as exc:
                videos.append({"id": path.stem, "split": split, "error": str(exc)})
                continue
            label = _label_info(root, split, path.stem, int(meta["num_frames"]))
            videos.append({
                "id": path.stem,
                "split": split,
                "num_frames": meta["num_frames"],
                "fps": meta["fps"],
                "width": meta["width"],
                "height": meta["height"],
                "size_mb": meta["size_mb"],
                "label": label,
                "thumbnail": (
                    "/api/datasets/frame?id=" + dataset_id
                    + "&split=" + split + "&video=" + path.stem + "&index=0&w=360"
                ),
            })
    summary["videos"] = videos
    summary["video_limit"] = limit_per_split
    return summary


def read_frame(dataset_id: str, split: str, video_id: str, index: int, width: int = 640) -> bytes:
    if split not in ("training", "testing"):
        raise ValueError("split must be training or testing")
    if not VIDEO_ID_RE.fullmatch(video_id):
        raise ValueError("Invalid video id")
    root, _ = get_dataset(dataset_id)
    path = root / (split + "_videos") / (video_id + ".avi")
    if not path.is_file():
        for candidate in (root / (split + "_videos")).glob(video_id + ".*"):
            if candidate.suffix.lower() in _VIDEO_SUFFIXES:
                path = candidate
                break
    if not path.is_file():
        raise FileNotFoundError("Video not found: " + video_id)

    width = max(64, min(int(width), 1600))
    key = (str(path.resolve()), split, video_id, int(index), width)
    with _thumb_lock:
        cached = _thumb_cache.get(key)
    if cached is not None:
        return cached

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: " + str(path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = max(0, min(int(index), max(0, total - 1)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("Cannot read frame " + str(index) + " from " + str(path))
    finally:
        cap.release()

    height = max(1, int(frame.shape[0] * width / frame.shape[1]))
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        raise RuntimeError("Cannot encode preview frame")
    data = encoded.tobytes()
    with _thumb_lock:
        if len(_thumb_cache) > 512:
            _thumb_cache.clear()
        _thumb_cache[key] = data
    return data


# ---------------------------------------------------------------------------
# Upload / register
# ---------------------------------------------------------------------------

def _safe_relative(raw: str, fallback: str) -> Path:
    value = (raw or fallback or "").replace("\\", "/").strip()
    parts: list[str] = []
    for part in value.split("/"):
        if part in ("", "."):
            continue
        if part == ".." or part.startswith("/"):
            raise ValueError("Unsafe upload path: " + value)
        parts.append(part)
    if not parts:
        raise ValueError("Empty upload path")
    return Path(*parts)


def _unique_target(name: str) -> Path:
    base = UPLOAD_ROOT / name
    if not base.exists():
        return base
    for index in range(2, 1000):
        candidate = UPLOAD_ROOT / (name + "_" + str(index))
        if not candidate.exists():
            return candidate
    raise ValueError("Too many datasets with the same name")


def _extract_zip(archive: Path, target: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            name = member.filename.replace("\\", "/")
            if name.startswith("/") or ".." in Path(name).parts:
                raise ValueError("Unsafe zip member: " + name)
            if member.is_dir():
                continue
            destination = (target / name).resolve()
            if not is_within(destination, target):
                raise ValueError("Zip member escapes target: " + name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, open(destination, "wb") as sink:
                shutil.copyfileobj(source, sink)


def upload_dataset(files: Iterable[Any], rel_paths: list[str], name: str) -> dict[str, Any]:
    ensure_runtime_dirs()
    files = list(files)
    if not files:
        raise ValueError("No files uploaded")
    clean_name = sanitize_dataset_name(name or "dataset_" + time.strftime("%Y%m%d_%H%M%S"))
    target = _unique_target(clean_name)
    target.mkdir(parents=True, exist_ok=False)

    try:
        if len(files) == 1 and str(files[0].filename or "").lower().endswith(".zip"):
            archive = target / "_upload.zip"
            files[0].save(archive)
            _extract_zip(archive, target)
            archive.unlink(missing_ok=True)
        else:
            for index, storage in enumerate(files):
                raw = rel_paths[index] if index < len(rel_paths) else (storage.filename or "")
                relative = _safe_relative(raw, storage.filename or ("file_" + str(index)))
                destination = (target / relative).resolve()
                if not is_within(destination, target):
                    raise ValueError("Upload path escapes target")
                destination.parent.mkdir(parents=True, exist_ok=True)
                storage.save(destination)
        root = _find_dataset_root(target)
        if root is None:
            raise ValueError(
                "Uploaded files do not contain training_videos/testing_videos. "
                "Expected an Avenue-style folder."
            )
        return summarize_dataset(root)
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise


def register_dataset(raw_path: str) -> dict[str, Any]:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()
    if not is_within(candidate, DATA_ROOT):
        raise ValueError("Dataset path must stay inside the project data directory")
    root = candidate if (candidate / "training_videos").is_dir() else _find_dataset_root(candidate)
    if root is None:
        raise FileNotFoundError("No training_videos directory found under: " + str(candidate))
    return summarize_dataset(root)


def prepare_command(dataset_id: str, labels: bool = True, features: bool = False, device: str = "auto") -> list[str]:
    root, summary = get_dataset(dataset_id)
    if summary["splits"]["training"]["count"] < 2:
        raise ValueError("At least two training videos are required")
    command = [
        sys.executable, "-u", "webapp/prepare_dataset.py",
        "--root", str(root),
        "--device", device,
    ]
    command.append("--labels" if labels else "--no-labels")
    if features:
        command.append("--features")
    return command
