# 这个文件负责在融合后的特征上建模帧间依赖关系。
"""Temporal modules — frame-level sequence modeling after Fusion.

Fusion 输出 ``(B, T, D_f)`` 后，在 T 维上建模帧间依赖。

Baseline 用 ``TemporalIdentity``（零开销透传），后续可升级为：
- ``TemporalTransformer`` — self-attention over T，长程依赖
- ``TemporalGRU``        — 双向 GRU，参数少，适合短序列
- ``TemporalTCN``        — 时序卷积，局部感受野

所有实现共享相同的 forward 签名和 TemporalOutput。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .outputs import TemporalOutput


class TemporalIdentity(nn.Module):
    """不做时序处理——直接透传，baseline 默认选择。

    适用场景：
    - 快速验证整个 pipeline
    - 当 Fusion 已经充分融合帧级信息时
    - 未来 A/B test 的 baseline
    """

    def forward(self, features: Tensor) -> TemporalOutput:
        """features: (B, T, D_f) → TemporalOutput(features)，不做任何变换。"""
        return TemporalOutput(features=features)

    def extra_repr(self) -> str:
        return "passthrough"


class TemporalTransformer(nn.Module):
    """时序 Transformer — 在 T 维上做 self-attention。

    与 Spatial Transformer（在 H×W 上做 attention）不同，
    Temporal Transformer 把每一帧当作一个 token，在时间轴上做 self-attention：
    - Q/K/V 都来自帧序列
    - 每帧 attend 到所有其他帧
    - 输出位置编码增强的帧序列

    适用场景：
    - 长视频 clip（T >= 16）
    - 需要捕获长程时序依赖（如缓慢展开的异常）
    - 有充足 GPU 资源
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_length: int = 128,
    ) -> None:
        super().__init__()
        self.dim = dim

        # 可学习的位置编码（余弦位置编码的简化替代）
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_length, dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(self, features: Tensor) -> TemporalOutput:
        """features: (B, T, D_f) → TemporalOutput(features)。"""
        B, T, D = features.shape

        # 加位置编码
        x = features + self.pos_embed[:, :T, :]

        # Self-attention over T
        out = self.transformer(x)                       # (B, T, D)

        return TemporalOutput(features=out)

    def extra_repr(self) -> str:
        num_layers = self.transformer.num_layers
        return f"dim={self.dim}, num_layers={num_layers}"
