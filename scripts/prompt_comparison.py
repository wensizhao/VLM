#!/usr/bin/env python3
"""Prompt 消融对比实验——固定冻结 CLIP，只改 prompt 集合，比较真实 AUC/AP。

用法::

    python scripts/prompt_comparison.py --config configs/experiment.yaml
    python scripts/prompt_comparison.py --experiments label_only scene_only all_types

实验定义来自 ``configs/prompts.yaml`` 的 ``experiments`` 段（改 yaml 即可）。
输出：``results/prompt_comparison.json`` + ``results/prompt_comparison.md``。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch

from datasets.builders import build_split_dataloader
from eval.zero_shot import evaluate_zero_shot
from models.backbone import CLIPBackbone
from models.factory import build_prompt_processor
from utils.config import load_experiment_config, save_config_snapshot
from utils.io import ensure_dirs
from utils.logging import get_logger
from utils.reproducibility import set_seed

log = get_logger(__name__)

_DEFAULT_EXPERIMENTS = ["label_only", "scene_only", "contrast_only", "all_types"]


def _markdown_table(results: list[dict[str, Any]]) -> str:
    header = "| experiment | #prompts | Frame AUC | Frame AP | Video AUC | Video AP |\n"
    sep = "|---|---|---|---|---|---|\n"
    rows = []
    for r in results:
        m = r["metrics"]
        rows.append(
            "| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                r["experiment"], r["num_prompts"],
                m.get("frame_auc", 0.0), m.get("frame_ap", 0.0),
                m.get("video_auc", 0.0), m.get("video_ap", 0.0),
            )
        )
    best = max(results, key=lambda r: r["metrics"].get("frame_auc", 0.0)) if results else None
    footer = ""
    if best is not None:
        footer = (
            "\n\n**Best by Frame AUC**: `{}` ({:.4f})\n".format(
                best["experiment"], best["metrics"].get("frame_auc", 0.0))
        )
    return header + sep + "\n".join(rows) + footer


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt ablation experiment")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--output", default="results/prompt_comparison.json")
    parser.add_argument("--experiments", nargs="+", default=_DEFAULT_EXPERIMENTS)
    parser.add_argument("--out-dir", default="results/prompt_comparison")
    parser.add_argument("--save-plots", action="store_true",
                        help="为每个实验保存热力图（较慢）")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    set_seed(int(cfg.seed))
    device = torch.device(str(cfg.device))

    experiments_cfg = cfg.prompt.get("experiments", None)
    if experiments_cfg is None:
        raise KeyError(
            "No `experiments` section in the prompt config. "
            "Define it in configs/prompts.yaml or pass --experiments."
        )

    log.info("Prompt comparison | %d experiments | device=%s", len(args.experiments), device)
    ensure_dirs({"result": Path(args.output).parent, "out": Path(args.out_dir)})
    save_config_snapshot(cfg, Path(args.out_dir) / "config.yaml")

    backbone = CLIPBackbone(
        model_name=str(cfg.model.backbone.name),
        pretrained=str(cfg.model.backbone.pretrained),
    )
    processor = build_prompt_processor(cfg.prompt)
    loader, src = build_split_dataloader(
        cfg.data, split=cfg.data.eval_split, shuffle=False, seed=int(cfg.seed),
    )
    log.info("Eval data ready | split=%s source=%s batches=%d",
             cfg.data.eval_split, src, len(loader))

    results: list[dict[str, Any]] = []
    start = time.time()
    for name in args.experiments:
        if name not in experiments_cfg:
            log.error("Unknown experiment %r; skipping (available: %s)",
                      name, list(experiments_cfg.keys()))
            continue
        types = list(experiments_cfg[name].get("types", []))
        log.info("Running %s | types=%s", name, types)
        try:
            res = evaluate_zero_shot(
                backbone=backbone,
                processor=processor,
                cfg=cfg,
                dataset=loader.dataset,
                device=device,
                batch_size=int(cfg.eval.batch_size),
                top_k=int(cfg.eval.top_k_prompts),
                out_dir=Path(args.out_dir) / name,
                save_plots=bool(args.save_plots),
                prompt_types=types,
            )
            results.append({
                "experiment": name,
                "types": types,
                "num_prompts": res["num_prompts"],
                "metrics": res["metrics"],
            })
        except Exception as e:  # noqa: BLE001
            log.error("Experiment %s failed: %s", name, e)
            continue

    total = time.time() - start
    summary = {
        "config": args.config,
        "experiments": args.experiments,
        "total_time_sec": round(total, 2),
        "results": results,
        "best_by_frame_auc": (
            max(results, key=lambda r: r["metrics"].get("frame_auc", 0.0))["experiment"]
            if results else None
        ),
    }

    ensure_dirs({"result": Path(args.output).parent})
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    md_path = Path(args.output).with_suffix(".md")
    with open(md_path, "w") as f:
        f.write("# Prompt Comparison (zero-shot, frozen CLIP)\n\n")
        f.write("Dataset: {} | split: {}\n\n".format(
            cfg.data.get("dataset", "avenue"), cfg.data.eval_split))
        f.write(_markdown_table(results))

    log.info("=" * 60)
    log.info("PROMPT COMPARISON SUMMARY (%.1fs)", total)
    log.info("=" * 60)
    for r in results:
        m = r["metrics"]
        log.info("%-14s K=%-3d frame_auc=%.4f frame_ap=%.4f video_auc=%.4f video_ap=%.4f",
                 r["experiment"], r["num_prompts"], m.get("frame_auc", 0.0),
                 m.get("frame_ap", 0.0), m.get("video_auc", 0.0), m.get("video_ap", 0.0))
    if results:
        best = max(results, key=lambda r: r["metrics"].get("frame_auc", 0.0))
        log.info("Best by Frame AUC: %s (%.4f)", best["experiment"],
                 best["metrics"].get("frame_auc", 0.0))
    log.info("Saved: %s | %s", args.output, md_path)


if __name__ == "__main__":
    main()
