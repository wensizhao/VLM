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
from datasets.splits import split_training_videos
from datasets.dataloader import seed_worker
from torch.utils.data import DataLoader
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
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip the final test-set evaluation")
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
    target_ckpt = Path(cfg.paths.checkpoint_dir) / str(cfg.paths.run_name) / "last.pt"
    if target_ckpt.exists() and not args.resume:
        raise FileExistsError(f"Run already exists: {target_ckpt}. Use --resume or a new --run-name")
    dirs = ensure_dirs({
        "log": cfg.paths.log_dir,
        "checkpoint": cfg.paths.checkpoint_dir,
        "result": cfg.paths.result_dir,
    })
    save_config_snapshot(
        cfg, dirs["result"] / str(cfg.paths.run_name) / "config.yaml",
    )
    from utils.io import save_json
    from utils.provenance import runtime_info
    save_json(runtime_info(), dirs["result"] / str(cfg.paths.run_name) / "environment.json")

    # ── 2. 数据 ──────────────────────────────────────────────
    train_loader, train_src = build_split_dataloader(
        cfg.data, split=cfg.data.train_split, shuffle=False, seed=int(cfg.seed),
        backbone_cfg=cfg.model.backbone,
    )
    train_ds, val_ds = split_training_videos(
        train_loader.dataset, float(cfg.data.get("validation_fraction", 0.25)), int(cfg.seed),
    )
    generator = torch.Generator().manual_seed(int(cfg.seed))
    loader_args = dict(batch_size=int(cfg.data.batch_size), num_workers=int(cfg.data.num_workers),
                       worker_init_fn=seed_worker, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    val_src = train_src
    from utils.io import save_json
    save_json({"train": [r.video_id for r in train_ds.video_records],
               "validation": [r.video_id for r in val_ds.video_records],
               "test_split": str(cfg.data.eval_split), "seed": int(cfg.seed),
               "label_mode": str(cfg.data.frame_label_mode)},
              dirs["result"] / str(cfg.paths.run_name) / "split_manifest.json")
    log.info("Data ready | train=%s(%d batches), eval=%s(%d batches)",
             train_src, len(train_loader), val_src, len(val_loader))

    # ── 3. 模型 + matcher ────────────────────────────────────
    model = build_model(cfg)
    matcher = build_matcher(cfg.model)
    log.info("Model built | prompts(K=%d), trainable=%d params",
             model.prompt_processor.num_prompts,
             sum(p.numel() for p in model.parameters() if p.requires_grad))

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
    if args.skip_eval:
        log.info("Skipping test-set evaluation (--skip-eval)")
        return

    best_ckpt = Path(cfg.paths.checkpoint_dir) / str(cfg.paths.run_name) / "best.pt"
    if best_ckpt.is_file():
        from utils.io import load_checkpoint
        state = load_checkpoint(best_ckpt)
        model.load_state_dict(state["model_state"])
        matcher.load_state_dict(state["matcher_state"])
        log.info("Loaded best checkpoint for evaluation")

    eval_loader, _ = build_split_dataloader(
        cfg.data, split=cfg.data.eval_split, shuffle=False, seed=int(cfg.seed),
        backbone_cfg=cfg.model.backbone,
    )
    eval_ds = eval_loader.dataset
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
