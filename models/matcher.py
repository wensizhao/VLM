# 这个文件负责计算视觉特征和文本特征之间的相似度。
"""Matcher: standalone similarity computation — decoupled from Alignment and Head.

可以在 pipeline 中多次复用：
1. **对齐阶段** — ``vis_pooled (B, D)`` × ``txt_pooled (K, D)`` → 对比学习 loss 的输入
2. **推理阶段** — ``head_embedding (B, D_f)`` × ``prompt_embs (K, D_f)`` → per-prompt 异常分数
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .outputs import MatchOutput


class Matcher(nn.Module):
    """相似度计算模块——纯函数式，不引入任何可学习参数之外的副作用。

    策略对比::

        ┌───────────┬──────────────────────┬────────────────────────────┐
        │ strategy  │ 公式                 │ 适用场景                    │
        ├───────────┼──────────────────────┼────────────────────────────┤
        │ cosine    │ cos(q, k)            │ 标准对比学习（CLIP 风格）    │
        │ dot       │ q @ kᵀ               │ 特征已 L2 归一化时等价 cosine│
        │ learnable │ q @ W @ kᵀ           │ 学习模态间的非线性映射       │
        └───────────┴──────────────────────┴────────────────────────────┘

    Temperature 控制 logits 的"锐度"：
    - 小 temperature → 更尖锐的分布（hard assignment）
    - 大 temperature → 更平滑的分布（soft assignment）
    - ``learnable_temp=True`` 时 temperature 是 nn.Parameter，训练中自动调整

    Usage::

        matcher = Matcher(strategy="cosine", temperature=0.07)

        # 对齐阶段：B 个 clip × K 个 prompt
        vis_pooled = aligned.visual_embedding.mean(dim=1)  # (B, D)
        txt_pooled = aligned.text_embedding.mean(dim=1)    # (K, D)
        match = matcher(vis_pooled, txt_pooled)            # MatchOutput

        # 推理阶段：head embedding × prompt embeddings
        anomaly = matcher(head_out.embedding, txt_pooled)  # MatchOutput
    """

    def __init__(
        self,
        strategy: str = "cosine",
        temperature: float = 0.07,
        learnable_temp: bool = True,
        in_dim: int | None = None,  # 仅 strategy="learnable" 时需要
    ) -> None:
        super().__init__()

        _STRATEGIES = ("cosine", "dot", "learnable")
        if strategy not in _STRATEGIES:
            raise ValueError(
                f"Unknown matcher strategy {strategy!r}. "
                f"Expected one of {_STRATEGIES}."
            )
        self.strategy = strategy

        # ── 可学习的模态映射矩阵（仅 learnable）─
        if strategy == "learnable":
            if in_dim is None:
                raise ValueError(
                    "in_dim is required when strategy='learnable'."
                )
            # Xavier init 保持初始输出分布稳定
            self.W = nn.Parameter(torch.empty(in_dim, in_dim))
            nn.init.xavier_uniform_(self.W)
        else:
            self.register_buffer("W", torch.empty(0))  # 占位

        # ── Temperature / logit scale ─
        if learnable_temp:
            # logit_scale = ln(1/T)，保证 exp 后为正
            self.logit_scale = nn.Parameter(
                torch.ones([]) * math.log(1.0 / temperature)
            )
        else:
            self.register_buffer(
                "logit_scale",
                torch.ones([]) * math.log(1.0 / temperature),
            )

    # ── 核心 API ─────────────────────────────────────────────────

    def forward(self, query: Tensor, key: Tensor) -> MatchOutput:
        """计算两组归一化向量之间的相似度。

        Args:
            query: ``(B, D)`` — 视觉侧池化特征。
            key:   ``(K, D)`` — 文本侧池化特征。

        Returns:
            MatchOutput:
            - ``similarity``: ``(B, K)`` 余弦相似度矩阵。
            - ``logits``:     ``(B, K)`` temperature-scaled logits。
            - ``temperature``: 当前温度值（标量 float）。
        """
        # L2 normalize 保证 cosine similarity 在 [-1, 1]
        q = F.normalize(query, dim=-1)
        k = F.normalize(key, dim=-1)

        if self.strategy == "learnable":
            # q @ W @ k^T — 学习模态映射后的相似度
            similarity = q @ self.W @ k.T                     # (B, K)
        else:
            # cosine: 等价于 q @ k^T（因为已经 normalize）
            similarity = q @ k.T                              # (B, K)

        temperature = 1.0 / self.logit_scale.exp().item()
        logits = similarity * self.logit_scale.exp()          # (B, K)

        return MatchOutput(
            similarity=similarity,
            logits=logits,
            temperature=temperature,
        )

    # ── 查询 API ─────────────────────────────────────────────────

    @property
    def temperature(self) -> float:
        """当前温度值。"""
        return 1.0 / self.logit_scale.exp().item()

    @torch.no_grad()
    def set_temperature(self, value: float) -> None:
        """手动设置温度（覆盖训练学到的值）。"""
        self.logit_scale.fill_(math.log(1.0 / value))

    def extra_repr(self) -> str:
        return (
            f"strategy={self.strategy!r}, "
            f"temperature={self.temperature:.4f}, "
            f"learnable_temp={isinstance(self.logit_scale, nn.Parameter)}"
        )
