#!/usr/bin/env python3
"""Re-evaluate every completed sweep checkpoint against verified official labels."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf
from utils.config import load_experiment_config, save_config_snapshot
from utils.io import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sweep", type=Path, required=True)
    parser.add_argument("--official-config", default="configs/official_eval.yaml")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation results: {args.out_dir}")
    official = load_experiment_config(args.official_config)
    if official.data.frame_label_mode != "official":
        raise ValueError("Re-evaluation requires official labels")
    source = json.loads(args.source_sweep.read_text())
    runs = [r for r in source if r["status"] == "completed"]
    if not runs:
        raise ValueError("No completed source runs")
    planned = []
    for record in runs:
        cfg = load_experiment_config(record["config"])
        checkpoint = Path(cfg.paths.checkpoint_dir) / cfg.paths.run_name / "best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        cfg.data = OmegaConf.create(OmegaConf.to_container(official.data, resolve=True))
        cfg.paths.result_dir = str(args.out_dir)
        cfg.eval.save_plots = False
        planned.append((record, cfg, checkpoint))
    args.out_dir.mkdir(parents=True)
    reports = []
    for record, cfg, checkpoint in planned:
        name = record["run_name"]
        config = args.out_dir / "configs" / f"{name}.yaml"
        save_config_snapshot(cfg, config)
        log = args.out_dir / f"{name}.log"
        with log.open("w") as output:
            env = dict(os.environ)
            env["MKL_THREADING_LAYER"] = "GNU"
            proc = subprocess.run([sys.executable, "eval.py", "--config", str(config),
                                   "--ckpt", str(checkpoint)], stdout=output, stderr=subprocess.STDOUT, env=env)
        result = {"run_name": name, "checkpoint": str(checkpoint), "returncode": proc.returncode,
                  "training_label_mode": record["label_mode"], "evaluation_label_mode": "official",
                  "selection": record.get("selection"), "log": str(log)}
        if proc.returncode == 0:
            evaluated = json.loads((args.out_dir / name / "metrics.json").read_text())
            result["metrics"] = evaluated["metrics"]
        reports.append(result)
        save_json(reports, args.out_dir / "summary.json")
        print(name, "OK" if proc.returncode == 0 else "FAILED", result.get("metrics"), flush=True)
    lines = ["# Existing motion-pseudo-trained checkpoints on official frame labels", "",
             "Fixed runs; checkpoints selected on held-out training videos. No test-label model selection.", "",
             "| Run | Frame AUC | Frame AP |", "|---|---:|---:|"]
    for record in reports:
        m = record.get("metrics", {})
        lines.append(f"| {record['run_name']} | {m.get('frame_auc')} | {m.get('frame_ap')} |")
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    if any(r["returncode"] != 0 for r in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
