#!/usr/bin/env python3
# 训练入口——一行命令完成"配置加载 → 数据 → 模型 → 训练 → 视频级评估"。
"""Train VLM-VAD end-to-end and evaluate on the test split.

用法::

    python train.py --config configs/experiment.yaml
    python train.py --config configs/experiment.yaml --resume ./checkpoints/exp01/last.pt

完整闭环（每条链路都是通的）:
    1. 读配置（experiment.yaml + prompts.yaml 合并）
    2. 固定随机种子（可复现）
    3. 构建数据（自动选 VideoDataset / FeatureDataset）
    4. 构建模型 + matcher + loss
    5. Trainer.fit() 训练并保存 checkpoint
    6. evaluate_videos() 做视频级评估，输出 metrics + 解释 + 可视化
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 保证从任意 cwd 都能 import 项目模块
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch

from datasets.builders import build_split_dataloader
from eval.inference import evaluate_videos
from models.factory import build_matcher, build_model
from train.trainer import Trainer, resume_from_checkpoint
from utils.config import load_experiment_config, save_config_snapshot
from utils.io import ensure_dirs
from utils.logging import get_logger
from utils.reproducibility import set_seed

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VLM-VAD")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override cfg.train.epochs (smoke tests)")
    parser.add_argument("--run-name", default=None,
                        help="Override cfg.paths.run_name")
    args = parser.parse_args()

    # ── 1. 配置 ──────────────────────────────────────────────
    cfg = load_experiment_config(args.config)
    if args.epochs is not None:
        cfg.train.epochs = int(args.epochs)
    if args.run_name is not None:
        cfg.paths.run_name = str(args.run_name)
    set_seed(int(cfg.seed))  # 先固定种子，再构建数据/模型 → 全部可复现
    log.info("Config loaded: %s (device=%s, seed=%d)", args.config, cfg.device, cfg.seed)

    device = torch.device(str(cfg.device))
    dirs = ensure_dirs({
        "log": cfg.paths.log_dir,
        "checkpoint": cfg.paths.checkpoint_dir,
        "result": cfg.paths.result_dir,
    })
    save_config_snapshot(
        cfg, dirs["result"] / str(cfg.paths.run_name) / "config.yaml",
    )

    # ── 2. 数据 ──────────────────────────────────────────────
    train_loader, train_src = build_split_dataloader(
        cfg.data, split=cfg.data.train_split, shuffle=True, seed=int(cfg.seed),
    )
    val_loader, val_src = build_split_dataloader(
        cfg.data, split=cfg.data.eval_split, shuffle=False, seed=int(cfg.seed),
    )
    log.info("Data ready | train=%s(%d batches), eval=%s(%d batches)",
             train_src, len(train_loader), val_src, len(val_loader))

    # ── 3. 模型 + matcher ────────────────────────────────────
    model = build_model(cfg)
    matcher = build_matcher(cfg.model)
    log.info("Model built | prompts(K=%d), trainable=%d params",
             model.prompt_processor.num_prompts,
             sum(1 for p in model.parameters() if p.requires_grad))

    # ── 4. 训练 ──────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        matcher=matcher,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        log_dir=dirs["log"],
        checkpoint_dir=dirs["checkpoint"],
        run_name=str(cfg.paths.run_name),
    )

    if args.resume:
        resumed_epoch = resume_from_checkpoint(trainer, args.resume)
        trainer.fit(start_epoch=resumed_epoch + 1)
    else:
        trainer.fit()

    # ── 5. 视频级评估（用最优 checkpoint 的权重）────────────────
    best_ckpt = Path(cfg.paths.checkpoint_dir) / str(cfg.paths.run_name) / "best.pt"
    if best_ckpt.is_file():
        from utils.io import load_checkpoint
        state = load_checkpoint(best_ckpt)
        model.load_state_dict(state["model_state"])
        log.info("Loaded best checkpoint for evaluation")

    eval_ds = val_loader.dataset
    results = evaluate_videos(
        model=model,
        matcher=matcher,
        dataset=eval_ds,
        device=device,
        batch_size=int(cfg.eval.batch_size),
        top_k=int(cfg.eval.top_k_prompts),
        out_dir=dirs["result"] / str(cfg.paths.run_name),
        save_plots=bool(cfg.eval.save_plots),
        cfg=cfg,
    )
    log.info("Results saved to %s", dirs["result"] / str(cfg.paths.run_name))


if __name__ == "__main__":
    main()
