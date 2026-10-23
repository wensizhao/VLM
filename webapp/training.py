"""Build reproducible experiment configs and subprocess commands for the UI."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from utils.config import load_experiment_config, save_config_snapshot

from . import data_catalog
from .config import (
    BASE_TRAIN_CONFIG,
    CHECKPOINT_ROOT,
    GENERATED_CONFIG_ROOT,
    OFFICIAL_EVAL_CONFIG,
    OFFICIAL_LABEL_ROOT,
    PROJECT_ROOT,
    RESULTS_ROOT,
    ensure_runtime_dirs,
    is_within,
    rel_to_project,
    sanitize_run_name,
)

PROMPT_TYPES = {"label", "scene", "contrast"}
FUSIONS = {"concat", "gated", "crossattn"}
TEMPORALS = {"identity", "transformer"}
DEVICES = {"auto", "cpu", "cuda"}
LABEL_MODES = {"motion_diff", "pixel"}
USE_FEATURE = {"auto", "true", "false"}


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _clamp_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, parsed))


def _configure_dataset_paths(cfg: DictConfig, root: Path) -> None:
    """Point both splits at the selected upload using the upstream dataset API."""
    for section, split in (("train", "training"), ("test", "testing")):
        masks = root / "ground_truth" / (split + "_label_mask")
        cfg.data[section] = {
            "video_path": rel_to_project(root / (split + "_videos")),
            "mask_path": rel_to_project(masks) if masks.is_dir() else None,
        }
    if "root" in cfg.data:
        del cfg.data["root"]


def build_training_config(payload: dict[str, Any]) -> dict[str, Any]:
    ensure_runtime_dirs()
    run_name = sanitize_run_name(str(payload.get("run_name", "")))
    dataset_id = str(payload.get("dataset_id", "")).strip()
    if not dataset_id:
        raise ValueError("dataset_id is required")
    root, summary = data_catalog.get_dataset(dataset_id)
    if summary["splits"]["training"]["count"] < 2:
        raise ValueError("At least two training videos are required")
    if not summary["has_masks"]:
        raise ValueError(
            "This dataset has no *_vol mask metadata. Run the one-click preparation first."
        )
    if (CHECKPOINT_ROOT / run_name / "last.pt").exists():
        raise ValueError("Run already exists: " + run_name + ". Choose another run name.")

    epochs = _clamp_int(payload.get("epochs"), 1, 100, 10)
    batch_size = _clamp_int(payload.get("batch_size"), 1, 64, 4)
    num_workers = _clamp_int(payload.get("num_workers"), 0, 16, 0)
    device = str(payload.get("device", "auto")).lower()
    if device not in DEVICES:
        raise ValueError("device must be auto, cpu or cuda")
    fusion = str(payload.get("fusion", "concat")).lower()
    if fusion not in FUSIONS:
        raise ValueError("fusion must be concat, gated or crossattn")
    temporal = str(payload.get("temporal", "identity")).lower()
    if temporal not in TEMPORALS:
        raise ValueError("temporal must be identity or transformer")
    label_mode = str(payload.get("label_mode", "motion_diff")).lower()
    if label_mode not in LABEL_MODES:
        raise ValueError("label_mode must be motion_diff or pixel")
    use_feature = str(payload.get("use_feature", "auto")).lower()
    if use_feature not in USE_FEATURE:
        raise ValueError("use_feature must be auto, true or false")

    prompt_types = payload.get("prompt_types", ["label", "scene", "contrast"])
    if isinstance(prompt_types, str):
        prompt_types = [prompt_types]
    prompt_types = [str(item).lower() for item in prompt_types if str(item).strip()]
    unknown = set(prompt_types) - PROMPT_TYPES
    if unknown:
        raise ValueError("Unknown prompt types: " + ", ".join(sorted(unknown)))
    if not prompt_types:
        prompt_types = ["label", "scene", "contrast"]

    cfg = load_experiment_config(BASE_TRAIN_CONFIG)
    cfg.paths.run_name = run_name
    cfg.device = device
    _configure_dataset_paths(cfg, root)
    cfg.data.frame_label_mode = label_mode
    if use_feature == "true":
        cfg.data.use_feature = True
    elif use_feature == "false":
        cfg.data.use_feature = False
    else:
        cfg.data.use_feature = "auto"
    cfg.data.batch_size = batch_size
    cfg.data.num_workers = num_workers
    cfg.train.epochs = epochs
    cfg.eval.save_plots = True

    label_root = summary.get("label_root")
    cfg.data.label_dir = label_root if label_root else None
    feature_root = summary.get("feature_root")
    if feature_root:
        cfg.data.feature_dir = feature_root
    else:
        cfg.data.feature_dir = rel_to_project(root / "features_v2")

    if set(prompt_types) == PROMPT_TYPES:
        if "types" in cfg.prompt:
            del cfg.prompt["types"]
    else:
        cfg.prompt.types = prompt_types

    cfg.model.fusion.type = fusion
    cfg.model.temporal.type = temporal

    config_path = GENERATED_CONFIG_ROOT / (run_name + ".yaml")
    save_config_snapshot(cfg, config_path)

    command = [
        sys.executable, "-u", "train.py",
        "--config", rel_to_project(config_path),
        "--run-name", run_name,
        "--epochs", str(epochs),
    ]
    if _as_bool(payload.get("skip_eval"), False):
        command.append("--skip-eval")

    return {
        "run_name": run_name,
        "config_path": str(config_path),
        "config_rel": rel_to_project(config_path),
        "command": command,
        "dataset": summary,
        "epochs": epochs,
        "prompt_types": sorted(prompt_types),
        "fusion": fusion,
        "temporal": temporal,
        "use_feature": use_feature,
        "label_mode": label_mode,
    }


def _load_run_config(run_id: str) -> tuple[Path, DictConfig]:
    run_dir = (RESULTS_ROOT / run_id).resolve()
    if not is_within(run_dir, RESULTS_ROOT):
        raise ValueError("Run id escapes results root")
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError("Run has no config snapshot: " + str(config_path))
    cfg = OmegaConf.load(str(config_path))
    if cfg.get("prompt") is not None and "config_path" in cfg.prompt:
        del cfg.prompt["config_path"]
    # Old Studio runs used data.root. Migrate the in-memory copy; retain snapshots.
    if cfg.data.get("root") is not None:
        root = Path(str(cfg.data.root))
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        _configure_dataset_paths(cfg, root)
    return config_path, cfg


def build_eval_config(run_id: str, mode: str) -> dict[str, Any]:
    mode = str(mode).lower()
    if mode not in {"motion", "official"}:
        raise ValueError("mode must be motion or official")
    _, run_cfg = _load_run_config(run_id)
    checkpoint = CHECKPOINT_ROOT / run_id / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError("No best.pt checkpoint for run: " + run_id)

    safe_run = run_id.replace("/", "_")
    base_id = safe_run + ("_official" if mode == "official" else "_eval")
    result_id = sanitize_run_name(base_id[:64])
    counter = 2
    while (RESULTS_ROOT / result_id).exists():
        suffix = "_" + str(counter)
        result_id = sanitize_run_name((base_id[:64 - len(suffix)] + suffix))
        counter += 1

    if mode == "official":
        if not (OFFICIAL_LABEL_ROOT / "testing" / "manifest.json").is_file():
            raise FileNotFoundError("Official testing labels are not available")
        cfg = load_experiment_config(OFFICIAL_EVAL_CONFIG)
        cfg.model = run_cfg.model
        cfg.prompt = run_cfg.prompt
        if "config_path" in cfg.prompt:
            del cfg.prompt["config_path"]
        cfg.data.train = run_cfg.data.train
        cfg.data.test = run_cfg.data.test
        cfg.data.feature_dir = run_cfg.data.feature_dir
        cfg.data.label_dir = rel_to_project(OFFICIAL_LABEL_ROOT)
        cfg.data.frame_label_mode = "official"
        cfg.data.use_feature = "auto"
        cfg.data.batch_size = 4
        cfg.data.num_workers = 0
        cfg.paths.run_name = result_id
        cfg.device = "auto"
        cfg.eval.save_plots = True
    else:
        cfg = OmegaConf.create(OmegaConf.to_container(run_cfg, resolve=True))
        cfg.paths.run_name = result_id
        cfg.device = "auto"
        cfg.data.num_workers = 0
        cfg.eval.save_plots = True

    config_path = GENERATED_CONFIG_ROOT / (result_id + ".yaml")
    save_config_snapshot(cfg, config_path)
    command = [
        sys.executable, "-u", "eval.py",
        "--config", rel_to_project(config_path),
        "--ckpt", rel_to_project(checkpoint),
    ]
    return {
        "run_id": run_id,
        "result_id": result_id,
        "mode": mode,
        "config_path": str(config_path),
        "config_rel": rel_to_project(config_path),
        "checkpoint": rel_to_project(checkpoint),
        "command": command,
    }


def build_demo_job(run_id: str, videos: list[str]) -> dict[str, Any]:
    """Render an MP4/GIF demo for an existing run via scripts/visualize_demo.py."""
    import re

    _, run_cfg = _load_run_config(run_id)
    checkpoint = CHECKPOINT_ROOT / run_id / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError("No best.pt checkpoint for run: " + run_id)
    cleaned = []
    for video in videos:
        value = str(video).strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", value):
            raise ValueError("Invalid video id: " + value)
        cleaned.append(value)
    if not cleaned:
        raise ValueError("At least one video id is required")
    if len(cleaned) > 19:
        raise ValueError("At most 19 videos can be rendered at once")

    safe_run = run_id.replace("/", "_")
    result_id = (safe_run + "_demo")[:64]
    result_id = sanitize_run_name(result_id)
    cfg = OmegaConf.create(OmegaConf.to_container(run_cfg, resolve=True))
    cfg.paths.run_name = result_id
    cfg.device = "auto"
    cfg.eval.save_plots = False
    config_path = GENERATED_CONFIG_ROOT / (result_id + ".yaml")
    save_config_snapshot(cfg, config_path)
    command = [
        sys.executable, "-u", "scripts/visualize_demo.py",
        "--config", rel_to_project(config_path),
        "--ckpt", rel_to_project(checkpoint),
        "--videos",
    ] + cleaned + [
        "--out", "results/" + result_id,
    ]
    return {
        "run_id": run_id,
        "result_id": result_id,
        "config_path": str(config_path),
        "config_rel": rel_to_project(config_path),
        "checkpoint": rel_to_project(checkpoint),
        "videos": cleaned,
        "command": command,
    }


def build_zero_shot_job(prompt: str, official: bool = True) -> dict[str, Any]:
    prompt = str(prompt or "scene").lower()
    if prompt not in {"label", "scene", "contrast", "all"}:
        raise ValueError("prompt must be label, scene, contrast or all")
    config = OFFICIAL_EVAL_CONFIG if official and OFFICIAL_LABEL_ROOT.is_dir() else BASE_TRAIN_CONFIG
    stamp = time.strftime("%H%M%S")
    result_id = sanitize_run_name("web_zero_shot_" + prompt + "_" + stamp)
    command = [
        sys.executable, "-u", "scripts/zero_shot_test.py",
        "--config", rel_to_project(config),
        "--prompts", prompt,
        "--output", "results/" + result_id + "/summary.json",
        "--out-dir", "results/" + result_id,
    ]
    return {
        "result_id": result_id,
        "prompt": prompt,
        "config": rel_to_project(config),
        "command": command,
    }
