# 这个文件负责将视觉和文本特征投影到共享语义空间——Open-Vocabulary VAD 的核心。
"""Semantic Alignment: project visual + text into a shared embedding space.

这是整个 Open-Vocabulary VAD 最核心的模块。对齐质量直接决定零样本泛化能力：
- 训练时：投影后的特征用于对比学习 loss，拉近匹配的 (video, prompt) pair
- 推理时：未见过的 prompt 能在共享空间中找到正确的视觉模式

设计原则：
1. **保留 T 和 L** — 不做池化，下游 Fusion 需要帧级 + token 级特征
2. **不计算 similarity** — similarity 是 Matcher 的职责
3. **同时输出 raw + projected** — Fusion 通过 ``fusion_input`` 自由选择
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from .outputs import AlignmentOutput


class SemanticAlignment(nn.Module):
    """视觉-语言语义对齐投影。

    把 Backbone 输出的特征投影到共享语义空间，使：
    - "有人在打架"的画面 与 "fighting" 在空间中接近
    - "正常行走"的画面 与 "fighting" 在空间中远离

    对齐质量直接影响：模型对未见异常类型的零样本检测能力。

    Usage::

        align = SemanticAlignment(in_dim=512, aligned_dim=512, proj_type="linear")
        aligned = align(vis_feat, txt_feat)
        # aligned.visual_embedding: (B, T, 512) — 投影后的视觉特征
        # aligned.visual_feature:   (B, T, 512) — 原始视觉特征
        # Fusion 可选择使用 feature / embedding / concat

    Parameters:
        in_dim: Backbone 输出维度 D。
        aligned_dim: 共享空间维度 D_a（通常等于 D）。
        proj_type: 投影类型 — "linear"（单层 + LayerNorm）或 "mlp"（双层 + GELU）。
    """

    def __init__(
        self,
        in_dim: int = 512,
        aligned_dim: int = 512,
        proj_type: str = "linear",
    ) -> None:
        super().__init__()

        _VALID_TYPES = ("linear", "mlp")
        if proj_type not in _VALID_TYPES:
            raise ValueError(
                f"proj_type must be one of {_VALID_TYPES}, got {proj_type!r}"
            )
        self.proj_type = proj_type
        self.in_dim = in_dim
        self.aligned_dim = aligned_dim

        if proj_type == "linear":
            # 单层投影 + LayerNorm — baseline，参数少，训练快
            self.vis_proj = nn.Sequential(
                nn.Linear(in_dim, aligned_dim),
                nn.LayerNorm(aligned_dim),
            )
            self.txt_proj = nn.Sequential(
                nn.Linear(in_dim, aligned_dim),
                nn.LayerNorm(aligned_dim),
            )
        elif proj_type == "mlp":
            # 双层 MLP + GELU — 更强的非线性对齐能力
            self.vis_proj = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.GELU(),
                nn.Linear(in_dim, aligned_dim),
                nn.LayerNorm(aligned_dim),
            )
            self.txt_proj = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.GELU(),
                nn.Linear(in_dim, aligned_dim),
                nn.LayerNorm(aligned_dim),
            )

    # ── 核心 API ─────────────────────────────────────────────────

    def forward(
        self,
        visual_feature: Tensor,
        text_feature: Tensor,
    ) -> AlignmentOutput:
        """将视觉和文本特征投影到共享语义空间。

        Args:
            visual_feature: ``(B, T, D)`` — Backbone 原始视觉特征，不池化。
            text_feature:   ``(K, L, D)`` — Backbone 原始文本特征，不取 EOS。

        Returns:
            AlignmentOutput:
            - visual_feature:  ``(B, T, D)``    原始视觉特征（透传）
            - visual_embedding: ``(B, T, D_a)``  投影到共享空间的视觉特征
            - text_feature:     ``(K, L, D)``    原始文本特征（透传）
            - text_embedding:   ``(K, L, D_a)``  投影到共享空间的文本特征
        """
        visual_embedding = self.vis_proj(visual_feature)  # (B, T, D) → (B, T, D_a)
        text_embedding = self.txt_proj(text_feature)       # (K, L, D) → (K, L, D_a)

        return AlignmentOutput(
            visual_feature=visual_feature,
            visual_embedding=visual_embedding,
            text_feature=text_feature,
            text_embedding=text_embedding,
        )

    # ── 查询 ─────────────────────────────────────────────────────

    def extra_repr(self) -> str:
        return (
            f"in_dim={self.in_dim}, "
            f"aligned_dim={self.aligned_dim}, "
            f"proj_type={self.proj_type!r}"
        )
