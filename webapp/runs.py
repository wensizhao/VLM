"""Run discovery and result aggregation for the validation dashboard."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from .config import (
    CHECKPOINT_ROOT,
    OFFICIAL_LABEL_ROOT,
    RESULTS_ROOT,
    is_within,
    rel_to_project,
)

_METRIC_FILES = ("metrics.json", "zero_shot_metrics.json")
_EXPLANATION_FILES = ("explanations.json", "zero_shot_explanations.json")


def _candidate_dirs() -> list[Path]:
    found: list[Path] = []
    if not RESULTS_ROOT.is_dir():
        return found
    for entry in sorted(RESULTS_ROOT.iterdir()):
        if not entry.is_dir():
            continue
        if any((entry / name).is_file() for name in _METRIC_FILES):
            found.append(entry.resolve())
            continue
        for child in sorted(entry.iterdir()):
            if child.is_dir() and any((child / name).is_file() for name in _METRIC_FILES):
                found.append(child.resolve())
    return found


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _metrics_path(entry: Path) -> tuple[Path, str] | None:
    for name in _METRIC_FILES:
        path = entry / name
        if path.is_file():
            return path, ("train" if name == "metrics.json" else "zero_shot")
    return None


def _plot_dir(entry: Path) -> Path | None:
    for name in ("plots", "zero_shot_plots"):
        candidate = entry / name
        if candidate.is_dir():
            return candidate
    return None


def _demo_dir(entry: Path) -> Path | None:
    candidate = entry.parent / (entry.name + "_demo")
    return candidate if candidate.is_dir() else None


def _explanations_path(entry: Path) -> Path | None:
    for name in _EXPLANATION_FILES:
        path = entry / name
        if path.is_file():
            return path
    return None


def _result_url(relative: str) -> str:
    from urllib.parse import quote
    return "/api/results/file?path=" + quote(relative, safe="")


def summarize_run(entry: Path, include_per_video: bool = False) -> dict[str, Any] | None:
    metrics_info = _metrics_path(entry)
    if metrics_info is None:
        return None
    metrics_path, kind = metrics_info
    try:
        data = _read_json(metrics_path)
    except Exception:
        return None
    metrics = data.get("metrics", data) if isinstance(data, dict) else {}
    run_id = entry.relative_to(RESULTS_ROOT.resolve()).as_posix()
    checkpoint = CHECKPOINT_ROOT / run_id / "best.pt"
    plot_dir = _plot_dir(entry)
    demo_dir = _demo_dir(entry)
    mtimes = [metrics_path.stat().st_mtime]
    if checkpoint.is_file():
        mtimes.append(checkpoint.stat().st_mtime)
    if plot_dir is not None:
        mtimes.append(plot_dir.stat().st_mtime)
    if demo_dir is not None:
        mtimes.append(demo_dir.stat().st_mtime)

    plots = []
    if plot_dir is not None:
        for path in sorted(plot_dir.glob("*.png")):
            relative = path.resolve().relative_to(RESULTS_ROOT.resolve()).as_posix()
            plots.append({"name": path.name, "path": relative, "url": _result_url(relative)})

    demo = []
    if demo_dir is not None:
        for path in sorted(demo_dir.iterdir()):
            if path.suffix.lower() in {".mp4", ".gif", ".md", ".png"}:
                relative = path.resolve().relative_to(RESULTS_ROOT.resolve()).as_posix()
                demo.append({
                    "name": path.name,
                    "kind": path.suffix.lower().lstrip("."),
                    "path": relative,
                    "url": _result_url(relative),
                })

    return {
        "id": run_id,
        "name": entry.name,
        "kind": kind,
        "metrics": metrics,
        "per_video": data.get("per_video", {}) if include_per_video and isinstance(data, dict) else {},
        "label_mode": data.get("label_mode") if isinstance(data, dict) else None,
        "benchmark_valid": data.get("benchmark_valid") if isinstance(data, dict) else None,
        "training_protocol": data.get("training_protocol") if isinstance(data, dict) else None,
        "has_checkpoint": checkpoint.is_file(),
        "checkpoint": rel_to_project(checkpoint) if checkpoint.is_file() else None,
        "official_available": checkpoint.is_file() and (OFFICIAL_LABEL_ROOT / "testing" / "manifest.json").is_file(),
        "num_plots": len(plots),
        "num_demo": len(demo),
        "modified": max(mtimes),
        "path": rel_to_project(entry),
    }


def list_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for entry in _candidate_dirs():
        try:
            summary = summarize_run(entry)
        except Exception:
            summary = None
        if summary is not None:
            runs.append(summary)
    runs.sort(key=lambda item: item["modified"], reverse=True)
    return runs


def _find_run_dir(run_id: str) -> Path:
    candidate = (RESULTS_ROOT / run_id).resolve()
    if not is_within(candidate, RESULTS_ROOT) or not candidate.is_dir():
        raise FileNotFoundError("Run not found: " + run_id)
    if _metrics_path(candidate) is None:
        raise FileNotFoundError("Run has no metrics: " + run_id)
    return candidate


def run_detail(run_id: str) -> dict[str, Any]:
    entry = _find_run_dir(run_id)
    summary = summarize_run(entry, include_per_video=True)
    if summary is None:
        raise FileNotFoundError("Run could not be summarized: " + run_id)
    metrics_path, kind = _metrics_path(entry)
    data = _read_json(metrics_path)

    config_summary: dict[str, Any] | None = None
    config_path = entry / "config.yaml"
    if config_path.is_file():
        try:
            cfg = OmegaConf.load(str(config_path))
            prompt = cfg.get("prompt", {}) or {}
            config_summary = {
                "seed": cfg.get("seed"),
                "device": cfg.get("device"),
                "run_name": cfg.get("paths", {}).get("run_name"),
                "data": OmegaConf.to_container(cfg.get("data", {}), resolve=True),
                "model": OmegaConf.to_container(cfg.get("model", {}), resolve=True),
                "train": OmegaConf.to_container(cfg.get("train", {}), resolve=True),
                "prompt": {
                    "types": list(prompt.get("types", ["label", "scene", "contrast"])),
                    "templates": OmegaConf.to_container(prompt.get("templates", {}), resolve=True),
                    "expansions": OmegaConf.to_container(prompt.get("expansions", {}), resolve=True),
                },
            }
        except Exception as exc:
            config_summary = {"error": str(exc)}

    explanations = {}
    explanation_path = _explanations_path(entry)
    if explanation_path is not None:
        try:
            explanations = _read_json(explanation_path)
        except Exception:
            explanations = {}

    split_manifest = None
    if (entry / "split_manifest.json").is_file():
        try:
            split_manifest = _read_json(entry / "split_manifest.json")
        except Exception:
            split_manifest = None

    artifacts = []
    for path in sorted(entry.iterdir()):
        if path.is_file():
            artifacts.append({
                "name": path.name,
                "size_kb": round(path.stat().st_size / 1024, 1),
                "path": rel_to_project(path),
            })

    summary["metrics_data"] = data
    summary["config"] = config_summary
    summary["explanations"] = explanations
    summary["split_manifest"] = split_manifest
    summary["artifacts"] = artifacts
    summary["plots"] = []
    plot_dir = _plot_dir(entry)
    if plot_dir is not None:
        for path in sorted(plot_dir.glob("*.png")):
            relative = path.resolve().relative_to(RESULTS_ROOT.resolve()).as_posix()
            summary["plots"].append({
                "name": path.name,
                "path": relative,
                "url": _result_url(relative),
            })
    summary["demo"] = []
    demo_dir = _demo_dir(entry)
    if demo_dir is not None:
        for path in sorted(demo_dir.iterdir()):
            if path.suffix.lower() in {".mp4", ".gif", ".md", ".png"}:
                relative = path.resolve().relative_to(RESULTS_ROOT.resolve()).as_posix()
                summary["demo"].append({
                    "name": path.name,
                    "kind": path.suffix.lower().lstrip("."),
                    "path": relative,
                    "url": _result_url(relative),
                })
    return summary


def result_file(relative_path: str) -> Path:
    candidate = (RESULTS_ROOT / relative_path).resolve()
    if not is_within(candidate, RESULTS_ROOT) or not candidate.is_file():
        raise FileNotFoundError("Result file not found: " + relative_path)
    return candidate


def compare_runs(run_ids: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_id in run_ids:
        try:
            entry = _find_run_dir(run_id)
            summary = summarize_run(entry)
        except Exception:
            summary = None
        if summary is not None:
            rows.append(summary)
    return rows
