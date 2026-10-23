#!/usr/bin/env python3
"""Zero-shot 测试入口——冻结 CLIP 直接检测异常，无需训练。

用法::

    python scripts/zero_shot_test.py --config configs/experiment.yaml --prompts all
    python scripts/zero_shot_test.py --prompts label

产物默认全部落在 ``results/zero_shot_{prompts}/`` 下（``summary.json`` +
``zero_shot_metrics.json`` + ``zero_shot_explanations.json`` + ``zero_shot_plots/``）。

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
from utils.provenance import runtime_info
from utils.io import save_json

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
    parser.add_argument("--out-dir", default=None,
                        help="Output dir (default: results/zero_shot_{prompts})")
    parser.add_argument("--output", default=None,
                        help="Summary JSON path (default: <out-dir>/summary.json)")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    set_seed(int(cfg.seed))
    device = torch.device(str(cfg.device))
    prompt_types = _PROMPT_TYPES[args.prompts]

    # 本次运行的所有产物都收敛到同一个 per-prompt 目录，避免互相覆盖
    out_dir = (
        Path(args.out_dir) if args.out_dir
        else Path("results") / f"zero_shot_{args.prompts}"
    )
    output_path = Path(args.output) if args.output else out_dir / "summary.json"

    log.info("Zero-shot test | config=%s device=%s prompts=%s", args.config, device, args.prompts)
    ensure_dirs({"out": out_dir, "summary": output_path.parent})
    save_config_snapshot(cfg, out_dir / "config.yaml")
    save_json(runtime_info(), out_dir / "environment.json")

    backbone = CLIPBackbone(
        model_name=str(cfg.model.backbone.name),
        pretrained=str(cfg.model.backbone.pretrained),
    )
    processor = build_prompt_processor(cfg.prompt)

    loader, src = build_split_dataloader(
        cfg.data, split=cfg.data.eval_split, shuffle=False, seed=int(cfg.seed),
        backbone_cfg=cfg.model.backbone,
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

    ensure_dirs({"summary": output_path.parent})
    summary = {
        "config": args.config,
        "prompt_selection": args.prompts,
        "num_prompts": results["num_prompts"],
        "metrics": results["metrics"],
        "inference_time_sec": round(elapsed, 2),
        "out_dir": str(out_dir),
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log.info("Zero-shot done in %.1fs | frame_auc=%s frame_ap=%s video_auc=%s video_ap=%s",
             elapsed, results["metrics"].get("frame_auc", 0.0),
             results["metrics"].get("frame_ap", 0.0),
             results["metrics"].get("video_auc", 0.0),
             results["metrics"].get("video_ap", 0.0))
    log.info("Summary saved to %s | plots/explanations in %s", output_path, out_dir)


if __name__ == "__main__":
    main()
