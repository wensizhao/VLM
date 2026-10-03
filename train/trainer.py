# 这个文件实现完整的训练循环——forward / loss / backward / step / logging / 验证 / checkpoint。
"""Trainer — the full training loop for VLM-VAD.

训练一个 epoch 的"最小闭环"（AGENTS.md §6.2）：
    for batch in dataloader:
        output = model(batch)              # 1. forward
        loss = criterion(output, labels)   # 2. loss
        loss.backward()                    # 3. backward（自动求导）
        optimizer.step()                   # 4. 参数更新
        optimizer.zero_grad()              # 5. 梯度清零
        log_metrics()                      # 6. 记录指标

自动求导的数学本质：
    ``loss.backward()`` 走反向模式自动微分（reverse-mode AD）。
    对每个参数 θ，它按链式法则累加出梯度：
        ∂loss/∂θ = Σ_layers (∂loss/∂h_L) · (∂h_L/∂h_{L-1}) · ... · (∂h_l/∂θ)
    存进 ``param.grad``。之后 optimizer 用梯度做一次更新，如 SGD：
        θ ← θ - lr · ∂loss/∂θ
    或 Adam（RMSProp 改良）：
        m ← β1·m + (1-β1)·g
        v ← β2·v + (1-β2)·g²
        θ ← θ - lr · m̂ / (√v̂ + ε)
    Adam 对每个参数自适应学习率：梯度大但震荡（v 大）的方向步子小，
    梯度一致（v 小）的方向步子大——这就是"自适应矩估计"名字的由来。
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from eval.metrics import clip_level_metrics, frame_level_metrics
from models import VLMModel
from models.matcher import Matcher
from train.losses import VLMVADLoss, build_prompt_polarity
from utils.io import load_checkpoint, save_checkpoint
from utils.logging import MetricLogger, TBLogger, get_logger

log = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 优化器 & 学习率调度
# ═══════════════════════════════════════════════════════════════════

def build_optimizer(model: VLMModel, cfg: DictConfig) -> torch.optim.Optimizer:
    """构造 AdamW 优化器，backbone 与下游模块使用不同学习率。

    为什么 backbone 用更小学习率？
        预训练 CLIP 的表示已经很好，微调时只需"轻触"（小 lr）以免灾难性遗忘；
        下游的 alignment/fusion/head 是随机初始化，需要大 lr 快速收敛。
        这就是**分层学习率（layer-wise learning rate / differential lr）**。

    AdamW 与 Adam 的区别：
        AdamW 把权重衰减（weight decay）从"梯度里"挪到"参数更新里"独立处理：
            θ ← θ - lr·(m̂/√v̂) - lr·λ·θ
        数学上等价于 L2 正则化的解耦版本，经验上泛化更好（所以叫 Adam with
        decoupled weight decay）。

    Args:
        model: VLMModel。
        cfg: ``cfg.train`` 配置段，含 ``lr`` / ``lr_backbone`` / ``weight_decay``。

    Returns:
        AdamW 优化器（只包含 requires_grad=True 的参数）。
    """
    lr = float(cfg.get("lr", 1e-3))
    lr_backbone = float(cfg.get("lr_backbone", lr * 0.1))
    wd = float(cfg.get("weight_decay", 1e-4))

    backbone_params: list[nn.Parameter] = []
    head_params: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone"):
            backbone_params.append(p)
        else:
            head_params.append(p)

    groups = [
        {"params": head_params, "lr": lr, "weight_decay": wd},
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": wd},
    ]
    # 过滤空组（backbone 全冻结时 backbone_params 为空）
    groups = [g for g in groups if g["params"]]
    return AdamW(groups)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
    total_steps: int,
) -> LambdaLR:
    """学习率调度：线性预热（warmup）→ 余弦退火（cosine decay）。

    数学定义：
        warmup 阶段（step < warmup_steps）：
            lr(t) = lr_max · (t / warmup_steps)          # 从 0 线性升到 lr_max
        余弦阶段：
            lr(t) = lr_min + 0.5·(lr_max - lr_min)·(1 + cos(π·t' / total))
        其中 t' = (t - warmup_steps) / (total_steps - warmup_steps)，t'∈[0,1]。

    为什么需要 warmup？
        训练初期参数远离最优点，梯度方向噪声大。一上来就用大学习率，
        模型可能直接"跳飞"（发散）。线性预热相当于先"试探"一个小步长，
        等梯度方向稳定后再加大。这是 Transformer 类模型的标准配置。

    为什么余弦退火？
        学习率太大时参数会在最优点附近震荡、无法收敛到极小值谷底。
        余弦退火让学习率按余弦曲线平滑降到接近 0，允许模型"微调落点"，
        经验上比 step decay 收敛得更好、更稳。
    """
    warmup = int(cfg.get("warmup_steps", 0))
    total = max(1, total_steps)
    lr_max = 1.0  # 比例制：LambdaLR 乘的是基础 lr，这里返回系数即可

    def lr_lambda(step: int) -> float:
        if step < warmup:
            # 线性预热：step/warmup ∈ (0, 1]
            return step / max(1, warmup)
        # 余弦：从 1 平滑降到 0（LambdaLR 会把结果乘回组里的 lr）
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# ═══════════════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════════════

class Trainer:
    """VLM-VAD 训练器——封装训练/验证/checkpoint/TensorBoard。

    Attributes:
        model: VLMModel（含 prompt_processor，prompt 的源头在模型内部）。
        matcher: Matcher——只用于对比 loss，属于模型外部（解耦设计）。
        criterion: VLMVADLoss。
        device: 计算设备。
    """

    def __init__(
        self,
        model: VLMModel,
        matcher: Matcher,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: DictConfig,
        device: torch.device,
        log_dir: str | Path,
        checkpoint_dir: str | Path,
        run_name: str = "exp01",
    ) -> None:
        self.model = model.to(device)
        self.matcher = matcher.to(device)
        self.criterion = VLMVADLoss(
            matcher=matcher,
            w_bce=float(cfg.train.loss.get("w_bce", 1.0)),
            w_contrastive=float(cfg.train.loss.get("w_contrastive", 0.5)),
            skip_bce_when_no_pos=bool(
                cfg.train.loss.get("skip_bce_when_no_pos", True)
            ),
        ).to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device

        # prompt 极性：由模型内部的 processor 决定 prompt 集合，训练/推理一致
        self.prompts = model.prompt_processor.process()
        self.polarity = build_prompt_polarity(self.prompts, cfg.prompt).to(device)
        log.info("Prompts (K=%d), polarity=%s", len(self.prompts),
                 [f"{p}:{pol}" for p, pol in
                  zip(self.prompts, self.polarity.tolist())])

        # 优化器 / 调度器
        total_steps = len(train_loader) * int(cfg.train.get("epochs", 1))
        self.optimizer = build_optimizer(model, cfg.train)
        self.scheduler = build_scheduler(self.optimizer, cfg.train, total_steps)

        # 路径 & 日志
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.run_name = run_name
        self.tb = TBLogger(self.log_dir / run_name)
        self.metric_logger = MetricLogger()
        self.best_val_metric = -1.0

        self.epochs = int(cfg.train.get("epochs", 1))
        self.grad_clip = float(cfg.train.get("grad_clip", 1.0))
        self.log_every = int(cfg.train.get("log_every", 10))
        self.amp_enabled = bool(cfg.train.get("amp", False)) and torch.cuda.is_available()
        if cfg.train.get("amp", False) and not torch.cuda.is_available():
            log.warning("amp requested but CUDA unavailable; falling back to fp32.")

    # ── 数据适配 ──────────────────────────────────────────────────

    @staticmethod
    def _to_device(batch: dict, device: torch.device) -> dict:
        return {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()}

    def _forward_batch(self, batch: dict) -> dict:
        """根据 batch 内容选择 forward 路径（原始像素 or 预提取特征）。"""
        if "vis_feat" in batch and batch["vis_feat"] is not None:
            # FeatureDataset 路径：跳过 ViT，训练提速 ~10×
            return self.model.forward_from_visual(batch["vis_feat"])
        return self.model.forward(batch["video"])

    # ── 训练步 ────────────────────────────────────────────────────

    def _train_step(self, batch: dict, step: int) -> dict[str, float]:
        batch = self._to_device(batch, self.device)
        output = self._forward_batch(batch)

        losses = self.criterion(
            output,
            frame_label=batch["frame_label"],
            clip_label=batch["clip_label"],
            polarity=self.polarity,
        )
        total = losses["total"]

        # backward：链式法则自动求导，梯度累加到各 param.grad
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()

        # 梯度裁剪：把整条梯度向量的范数压到 grad_clip 以内
        #   g ← g · min(1, grad_clip / ‖g‖)
        # 防止个别 batch 梯度范数爆炸（loss landscape 陡峭区）导致参数巨幅跳动。
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_clip,
        )

        self.optimizer.step()   # Adam 参数更新
        self.scheduler.step()   # 学习率调度（每步更新）

        return {
            "loss": float(losses["total"]),
            "bce": float(losses["bce"]),
            "contrastive": float(losses["contrastive"]),
            "lr": self.scheduler.get_last_lr()[0],
        }

    # ── Epoch 循环 ────────────────────────────────────────────────

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """跑一个训练 epoch，返回精确平均的指标。"""
        self.model.train()
        self.metric_logger.reset()
        total_batches = len(self.train_loader)
        global_step = epoch * total_batches

        loader = tqdm(self.train_loader, desc=f"Epoch {epoch} [train]", leave=False)
        for step, batch in enumerate(loader):
            stats = self._train_step(batch, step)

            # 记录指标
            self.metric_logger.update(**stats)
            if (step % self.log_every == 0) or (step == total_batches - 1):
                cur = self.metric_logger.get_stats()
                loader.set_postfix(
                    loss=f"{cur['loss']:.4f}",
                    bce=f"{cur['bce']:.4f}",
                    ctr=f"{cur['contrastive']:.4f}",
                    lr=f"{cur['lr']:.2e}",
                )
                self.tb.add_scalars("train", cur, global_step + step)

        return self.metric_logger.get_precise()

    @torch.no_grad()
    def validate_epoch(self, epoch: int) -> dict[str, float]:
        """验证一个 epoch——算 frame/clip 级 AUC & AP（训练监控用）。"""
        self.model.eval()
        frame_scores: list[torch.Tensor] = []
        frame_labels: list[torch.Tensor] = []
        clip_scores: list[torch.Tensor] = []
        clip_labels: list[torch.Tensor] = []

        for batch in tqdm(self.val_loader, desc=f"Epoch {epoch} [val]", leave=False):
            batch = self._to_device(batch, self.device)
            output = self._forward_batch(batch)
            frame_scores.append(output.frame_score.cpu())
            frame_labels.append(batch["frame_label"].cpu())
            clip_scores.append(output.clip_score.cpu())
            clip_labels.append(batch["clip_label"].cpu())

        fs = torch.cat(frame_scores).view(-1).numpy()
        fl = torch.cat(frame_labels).view(-1).numpy()
        cs = torch.cat(clip_scores).numpy()
        cl = torch.cat(clip_labels).numpy()

        metrics = {}
        metrics.update(frame_level_metrics(fs, fl))
        metrics.update(clip_level_metrics(cs, cl))

        # TensorBoard 记录验证指标
        self.tb.add_scalars("val", metrics, (epoch + 1) * len(self.train_loader))
        log.info(
            "Epoch %d val | frame_auc=%.4f frame_ap=%.4f clip_auc=%.4f clip_ap=%.4f",
            epoch, metrics.get("frame_auc", 0.0), metrics.get("frame_ap", 0.0),
            metrics.get("clip_auc", 0.0), metrics.get("clip_ap", 0.0),
        )
        return metrics

    # ── 主入口 ────────────────────────────────────────────────────

    def fit(self, start_epoch: int = 1) -> dict[str, float]:
        """完整训练流程：epoch 循环 + 验证 + checkpoint。

        Args:
            start_epoch: 起始 epoch（续训时传 resumed_epoch + 1）。
        """
        log.info(
            "Training started: epochs=%d..%d, train_batches=%d, val_batches=%d, "
            "device=%s, amp=%s",
            start_epoch, self.epochs, len(self.train_loader), len(self.val_loader),
            self.device, self.amp_enabled,
        )

        for epoch in range(start_epoch, self.epochs + 1):
            train_stats = self.train_epoch(epoch)
            log.info(
                "Epoch %d train | loss=%.4f bce=%.4f contrastive=%.4f lr=%.2e",
                epoch, train_stats["loss"], train_stats["bce"],
                train_stats["contrastive"], train_stats["lr"],
            )

            val_stats = self.validate_epoch(epoch)

            # 用 clip AUC 作为选优指标（视频级最终评估在 eval.py）
            metric = val_stats.get("clip_auc", 0.0)
            is_best = metric > self.best_val_metric
            if is_best:
                self.best_val_metric = metric

            self._save_checkpoint(epoch, val_stats, is_best)

        log.info("Training finished. Best clip_auc=%.4f", self.best_val_metric)
        return {"best_clip_auc": self.best_val_metric}

    # ── Checkpoint ────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, val_stats: dict, is_best: bool) -> None:
        """保存训练状态（含 optimizer/scheduler——这样能无缝续训）。"""
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "matcher_state": self.matcher.state_dict(),
            "prompt_state": self.model.prompt_processor.state_dict(),
            "val_metrics": val_stats,
            "best_val_metric": self.best_val_metric,
            "run_name": self.run_name,
        }
        # 每 epoch 覆盖最新 + 最优两档
        save_checkpoint(state, self.checkpoint_dir / self.run_name / "last.pt")
        if is_best:
            save_checkpoint(state, self.checkpoint_dir / self.run_name / "best.pt")


def resume_from_checkpoint(
    trainer: Trainer,
    ckpt_path: str | Path,
) -> int:
    """从 checkpoint 恢复训练（模型/优化器/调度器状态全量恢复）。

    Returns:
        已训练完的 epoch 数（续训从 epoch+1 开始）。
    """
    state = load_checkpoint(ckpt_path)
    trainer.model.load_state_dict(state["model_state"])
    trainer.optimizer.load_state_dict(state["optimizer_state"])
    trainer.scheduler.load_state_dict(state["scheduler_state"])
    trainer.matcher.load_state_dict(state["matcher_state"])
    trainer.best_val_metric = float(state.get("best_val_metric", -1.0))
    log.info("Resumed from epoch %d (best_val=%.4f)",
             state["epoch"], trainer.best_val_metric)
    return int(state["epoch"])
