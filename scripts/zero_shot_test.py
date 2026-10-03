#!/usr/bin/env python3
"""Zero-shot 测试入口——冻结 CLIP 直接检测异常，无需训练。

用法::

    python scripts/zero_shot_test.py --config configs/experiment.yaml --prompts all
    python scripts/zero_shot_test.py --prompts label --output results/zero_shot_label.json

支持的 ``--prompts``：label | scene | contrast | all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

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

_PROMPT_TYPES = {
    "label": ["label"],
    "scene": ["scene"],
    "contrast": ["contrast"],
    "all": None,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot VAD with frozen CLIP")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--prompts", default="all", choices=list(_PROMPT_TYPES.keys()))
    parser.add_argument("--output", default="results/zero_shot_test.json")
    parser.add_argument("--out-dir", default=None, help="Directory for plots/explanations")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    set_seed(int(cfg.seed))
    device = torch.device(str(cfg.device))
    prompt_types = _PROMPT_TYPES[args.prompts]
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.output).parent / f"zero_shot_{args.prompts}"

    log.info("Zero-shot test | config=%s device=%s prompts=%s", args.config, device, args.prompts)
    ensure_dirs({"result": Path(args.output).parent, "out": out_dir})
    save_config_snapshot(cfg, out_dir / "config.yaml")

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

    start = time.time()
    results = evaluate_zero_shot(
        backbone=backbone,
        processor=processor,
        cfg=cfg,
        dataset=loader.dataset,
        device=device,
        batch_size=int(cfg.eval.batch_size),
        top_k=int(cfg.eval.top_k_prompts),
        out_dir=out_dir,
        save_plots=bool(cfg.eval.save_plots),
        prompt_types=prompt_types,
    )
    elapsed = time.time() - start

    ensure_dirs({"result": Path(args.output).parent})
    summary = {
        "config": args.config,
        "prompt_selection": args.prompts,
        "num_prompts": results["num_prompts"],
        "metrics": results["metrics"],
        "inference_time_sec": round(elapsed, 2),
        "out_dir": str(out_dir),
    }
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log.info("Zero-shot done in %.1fs | frame_auc=%.4f frame_ap=%.4f video_auc=%.4f video_ap=%.4f",
             elapsed, results["metrics"].get("frame_auc", 0.0),
             results["metrics"].get("frame_ap", 0.0),
             results["metrics"].get("video_auc", 0.0),
             results["metrics"].get("video_ap", 0.0))
    log.info("Summary saved to %s | plots/explanations in %s", args.output, out_dir)


if __name__ == "__main__":
    main()
