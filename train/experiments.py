"""Plan and execute isolated prompt/fusion/backbone comparisons."""
from __future__ import annotations

import itertools
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from utils.config import save_config_snapshot
from utils.io import load_json, save_json


def experiment_matrix(base: DictConfig, sweep: DictConfig) -> list[tuple[str, DictConfig]]:
    """Cartesian product with independent configurations and per-backbone caches."""
    result = []
    for prompt, fusion, backbone, seed in itertools.product(
        sweep.prompt_types, sweep.fusions, sweep.backbones, sweep.seeds,
    ):
        if prompt not in {"label", "scene", "contrast", "all"}:
            raise ValueError(f"Unknown prompt type: {prompt}")
        if fusion not in {"concat", "gated", "crossattn"}:
            raise ValueError(f"Unknown fusion: {fusion}")
        cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
        cfg.seed = int(seed)
        cfg.prompt.types = [prompt] if prompt != "all" else ["label", "scene", "contrast"]
        cfg.model.fusion.type = fusion
        cfg.model.backbone.name = backbone.name
        cfg.model.backbone.pretrained = backbone.pretrained
        cfg.data.feature_dir = backbone.feature_dir
        cfg.paths.feature_dir = backbone.feature_dir
        cfg.train.epochs = int(sweep.get("epochs", cfg.train.epochs))
        if sweep.get("num_workers") is not None:
            cfg.data.num_workers = int(sweep.num_workers)
        cfg.eval.save_plots = bool(sweep.get("save_plots", False))
        name = f"{sweep.name}_{backbone.id}_{prompt}_{fusion}_s{seed}"
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"Invalid experiment name: {name}")
        cfg.paths.run_name = name
        result.append((name, cfg))
    names = [n for n, _ in result]
    if not result or len(set(names)) != len(names):
        raise ValueError("Experiment matrix is empty or contains duplicate run names")
    return result


def run_matrix(base: DictConfig, sweep: DictConfig, output: Path,
               dry_run: bool = False) -> list[dict[str, Any]]:
    output.mkdir(parents=True, exist_ok=True)
    project = Path(__file__).resolve().parents[1]
    plan = experiment_matrix(base, sweep)
    if not dry_run:
        for name, cfg in plan:
            if (Path(cfg.paths.checkpoint_dir) / name / "last.pt").exists():
                raise FileExistsError(f"Experiment {name} already exists; choose a new sweep name")
    records = []
    for name, cfg in plan:
        path = save_config_snapshot(cfg, output / "configs" / f"{name}.yaml")
        records.append({"run_name": name, "status": "planned", "config": str(path),
                        "label_mode": str(cfg.data.frame_label_mode),
                        "benchmark_valid": cfg.data.frame_label_mode == "pixel"})
    save_json(records, output / "summary.json")
    if dry_run:
        return records
    for (name, cfg), record in zip(plan, records):
        started = time.monotonic()
        record["status"] = "running"
        print(f"Running {name}", flush=True)
        save_json(records, output / "summary.json")
        log_path = output / f"{name}.log"
        with log_path.open("w") as log:
            completed = subprocess.run(
                [sys.executable, str(project / "train.py"), "--config", record["config"]],
                cwd=project, stdout=log, stderr=subprocess.STDOUT, check=False,
            )
        record.update(status="completed" if completed.returncode == 0 else "failed",
                      returncode=completed.returncode, elapsed_s=round(time.monotonic() - started, 2),
                      log=str(log_path))
        metrics_path = Path(cfg.paths.result_dir) / name / "metrics.json"
        if completed.returncode == 0:
            if not metrics_path.is_file():
                record.update(status="failed", error="Training exited without final metrics")
            else:
                record["metrics"] = load_json(metrics_path)["metrics"]
                history = load_json(Path(cfg.paths.result_dir) / name / "history_from_epoch_1.json")
                best = max(history, key=lambda h: h["validation"]["clip_auc"])
                record["selection"] = {"source": "held_out_training_videos", "epoch": best["epoch"],
                                       "validation_clip_auc": best["validation"]["clip_auc"]}
        save_json(records, output / "summary.json")
        print(f"{name}: {record['status']}", flush=True)
    lines = ["# Experiment comparison", "", "Motion pseudo labels are engineering validation only."
             if base.data.frame_label_mode == "motion_diff" else "Pixel annotation evaluation.", "",
             "| Run | Status | Frame AUC | Frame AP |", "|---|---|---:|---:|"]
    for r in records:
        m = r.get("metrics", {})
        lines.append(f"| {r['run_name']} | {r['status']} | {m.get('frame_auc', '—')} | {m.get('frame_ap', '—')} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return records
