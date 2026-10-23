#!/usr/bin/env python3
# 纯评估入口——只加载已训练好的 checkpoint 做视频级评估 + 可解释性。
"""Evaluate a trained VLM-VAD checkpoint on the test split.

用法::

    python eval.py --config configs/experiment.yaml --ckpt ./checkpoints/exp01/best.pt

    # 不带 --ckpt：下游模块为随机初始化，trained 指标无意义，
    # 但零样本 CLIP 保底（zs_* 指标）仍然有效——训练前的 baseline 参照
    python eval.py --config configs/experiment.yaml

输出（到 {result_dir}/{run_name}/）：
    - metrics.json      帧级/clip级/视频级 AUC & AP + zs_* 零样本保底指标
    - explanations.json 每个视频的异常原因（template-based 解释）
    - plots/*.png       时序热力图 + prompt 相似度条形图
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch

from datasets.builders import build_split_dataloader
from eval.inference import evaluate_videos
from models.factory import build_matcher, build_model
from utils.config import load_experiment_config, save_config_snapshot
from utils.io import ensure_dirs, load_checkpoint
from utils.logging import get_logger
from utils.reproducibility import set_seed

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VLM-VAD checkpoint")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--ckpt", default=None,
                        help="Checkpoint .pt 路径；省略则下游模块为随机初始化，"
                             "只看 zs_* 零样本保底指标")
    parser.add_argument("--eval-split", default=None,
                        help="Override eval split (default: cfg.data.eval_split)")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    set_seed(int(cfg.seed))
    device = torch.device(str(cfg.device))
    dirs = ensure_dirs({"result": cfg.paths.result_dir})
    save_config_snapshot(
        cfg, dirs["result"] / str(cfg.paths.run_name) / "config.yaml",
    )

    eval_split = args.eval_split if args.eval_split else str(cfg.data.eval_split)
    eval_loader, source = build_split_dataloader(
        cfg.data, split=eval_split, shuffle=False, seed=int(cfg.seed),
        backbone_cfg=cfg.model.backbone,
    )
    log.info("Eval data ready | %s(%d batches) via %s",
             eval_split, len(eval_loader), source)

    # 构建模型 + 加载权重
    model = build_model(cfg)
    matcher = build_matcher(cfg.model)
    if args.ckpt is not None:
        state = load_checkpoint(args.ckpt)
        from utils.provenance import validate_checkpoint
        validate_checkpoint(state, cfg, model.prompt_processor.process())
        model.load_state_dict(state["model_state"])
        if "matcher_state" in state:
            matcher.load_state_dict(state["matcher_state"])
        log.info("Checkpoint loaded (epoch=%s, best_val=%s)",
                 state.get("epoch", "?"), state.get("best_val_metric", "?"))
    else:
        # 未提供 --ckpt：下游模块随机初始化，trained 指标无意义；
        # 但 zs_* 零样本 CLIP 保底指标仍然有效（polarity 由 evaluate_videos 内部按 cfg 计算）
        log.warning(
            "未提供 --ckpt：下游模块为随机初始化，trained 指标无意义；"
            "本次运行请关注 zs_* 零样本 CLIP 保底指标（训练前的 baseline）。"
        )
        state = {"config": {}}

    # prompt 极性在 evaluate_videos 内部由 cfg 计算（含长度校验），无需在此重复
    results = evaluate_videos(
        model=model,
        matcher=matcher,
        dataset=eval_loader.dataset,
        device=device,
        batch_size=int(cfg.eval.batch_size),
        top_k=int(cfg.eval.top_k_prompts),
        out_dir=dirs["result"] / str(cfg.paths.run_name),
        save_plots=bool(cfg.eval.save_plots),
        cfg=cfg,
        training_protocol={"mode": "trained", "checkpoint": args.ckpt,
                           "label_mode": state.get("config", {}).get("data", {}).get("frame_label_mode", "unknown"),
                           "train_split": state.get("config", {}).get("data", {}).get("train_split", "unknown"),
                           "test_labels_used_for_training": (
                               False if state.get("config", {}).get("data", {}).get("train_split") == "training"
                               else None)},
    )
    log.info("Done. frame_auc=%s clip_auc=%s video_auc=%s",
             results["metrics"].get("frame_auc", 0.0),
             results["metrics"].get("clip_auc", 0.0),
             results["metrics"].get("video_auc", 0.0))


if __name__ == "__main__":
    main()
