# 这个文件负责从融合后的帧级特征预测异常分数。
"""Anomaly detection head — frame-level prediction, clip-level aggregation.

输入 ``(B, T, D_f)``，T 维完整保留。**先得到帧级分数，再从帧级聚合出 clip 级分数**。

设计哲学：frame_score → clip_score（细粒度 → 粗粒度），永远不反向。
- frame_score 是"原生"的——Head 对每一帧独立打分
- clip_score = amax(frame_score)：只要有任一帧异常，clip 就异常
- 反过来（从 clip 复制到帧）没有信息增益，只是假装有帧级输出
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .outputs import HeadOutput


class AnomalyHead(nn.Module):
    """帧级异常检测头。

    Args:
        in_dim: 输入维度（Fusion 输出 D_f）。
        hidden_dim: 隐藏层维度，默认 256。
        dropout: Dropout 概率。

    Usage::

        head = AnomalyHead(in_dim=512, hidden_dim=256)
        out = head(fused)  # fused: (B, T, 512) → HeadOutput
        # out.frame_score  (B, T)  真正的帧级异常分数
        # out.clip_score   (B,)     amax(frame_score)
        # out.embedding    (B, 512) 全局特征
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        # 帧级预测器：每一帧独立打分
        self.frame_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, fused: Tensor) -> HeadOutput:
        """从融合特征预测异常分数。

        Args:
            fused: ``(B, T, D_f)`` — Fusion + Temporal 输出的帧级特征。

        Returns:
            HeadOutput:
            - frame_logits: ``(B, T)`` 原始 logits（供 BCEWithLogits 使用）
            - frame_score: ``(B, T)``  帧级异常分数，值域 ``[0, 1]``
            - clip_score:  ``(B,)``    clip 级 = ``amax(frame_score)``
            - embedding:   ``(B, D_f)`` clip 级全局特征，用于 per-prompt 匹配
        """
        # 帧级 logits → sigmoid → score
        frame_logits = self.frame_head(fused).squeeze(-1)  # (B, T)
        frame_score = torch.sigmoid(frame_logits)           # (B, T) in [0,1]

        # 从帧级聚合到 clip 级
        clip_score = frame_score.amax(dim=1)               # (B,)

        # 全局特征：T 维 mean pool（用于 Matcher 做 per-prompt 匹配）
        embedding = fused.mean(dim=1)                       # (B, D_f)

        return HeadOutput(
            frame_logits=frame_logits,
            frame_score=frame_score,
            clip_score=clip_score,
            embedding=embedding,
        )

    def extra_repr(self) -> str:
        return f"in_dim={self.in_dim}, hidden_dim={self.hidden_dim}"
