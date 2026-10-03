# 这个文件负责视觉和文本特征的融合——采用 Text Memory Bank 模式，不膨胀 batch。
"""Fusion strategies — visual-text integration via Text Memory Bank.

核心变更（v2）：不再做 Cartesian Product ``(B,K,T,D)``。

所有 K 个 prompt 的 token 序列拼接为一个 Text Memory Bank：

    text_memory (K, L, D) → flatten → (1, K*L, D) → expand → (B, K*L, D)

每个视频帧通过 Cross-Attention attend 到整个 Text Memory，产生一个 text context vector。
Fusion 输出 ``(B, T, D_f)``，**B 不膨胀**。K=1000 时显存占用与 K=1 相同。

三种 Fusion 共享统一接口，VLMModel 中直接替换：

    fusion = ConcatFusion(D, 512, fusion_input="embedding")     # baseline
    fusion = GatedFusion(D, fusion_input="embedding")            # 轻量
    fusion = CrossAttnFusion(D, 512, fusion_input="embedding")   # 最强
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .outputs import FusionOutput


# ═══════════════════════════════════════════════════════════════════
# 辅助：从 Text Memory 获取 per-frame context
# ═══════════════════════════════════════════════════════════════════

def _text_context_from_memory(
    visual: Tensor,
    text_memory: Tensor,
    attn: nn.MultiheadAttention,
) -> tuple[Tensor, Tensor]:
    """Cross-attention: 每个视频帧 attend 到整个 Text Memory。

    Args:
        visual:      ``(B, T, D)`` — per-frame visual features.
        text_memory: ``(K, L, D)`` — all prompts' token sequences.
        attn:        MultiheadAttention module with batch_first=True.

    Returns:
        text_context: ``(B, T, D)`` — per-frame text context.
        attn_weights: ``(B, T, K*L)`` — attention weights (for interpretability).
    """
    B, T, D = visual.shape
    K, L, _ = text_memory.shape

    # Flatten text memory: (K, L, D) → (B, K*L, D)
    tm = text_memory.view(1, K * L, D).expand(B, -1, -1)

    # Cross-attention: Q=visual, K/V=text_memory
    text_context, attn_weights = attn(query=visual, key=tm, value=tm)
    # text_context: (B, T, D), attn_weights: (B, T, K*L)
    return text_context, attn_weights


# ═══════════════════════════════════════════════════════════════════
# Fusion 策略
# ═══════════════════════════════════════════════════════════════════

class ConcatFusion(nn.Module):
    """逐帧拼接融合 — baseline。

    流程：
    1. 每个视频帧 attend 到整个 Text Memory → text_context (B, T, D)
    2. 逐帧 concat: fused[t] = [visual[t], text_context[t]] → (B, T, 2*D)
    3. Linear + ReLU + Dropout → (B, T, D_f)

    适用：快速验证 pipeline，轻量计算。
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.1,
        fusion_input: str = "embedding",
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.fusion_input = fusion_input
        self.in_dim = in_dim
        self.out_dim = out_dim

        # 轻量 cross-attention：获取 per-frame text context
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=in_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # 融合投影
        self.proj = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        visual: Tensor,
        text_memory: Tensor,
    ) -> FusionOutput:
        """visual: (B, T, D), text_memory: (K, L, D) → FusionOutput."""
        text_context, attn_weights = _text_context_from_memory(
            visual, text_memory, self.cross_attn,
        )
        # 逐帧 concat
        fused = torch.cat([visual, text_context], dim=-1)      # (B, T, 2*D)
        return FusionOutput(
            fused=self.proj(fused),                             # (B, T, D_f)
            attention=attn_weights,                             # (B, T, K*L)
            gate_values=None,
        )

    def extra_repr(self) -> str:
        return (
            f"in_dim={self.in_dim}, out_dim={self.out_dim}, "
            f"fusion_input={self.fusion_input!r}"
        )


class GatedFusion(nn.Module):
    """逐帧门控融合 — 轻量级。

    流程：
    1. Cross-Attention → text_context (B, T, D)
    2. gate = σ(W_v(visual) + W_t(text_context)) ∈ (B, T, D)
    3. fused = gate ⊙ visual + (1 - gate) ⊙ text_context

    每一帧的每个维度独立门控：
    - gate ≈ 1 → 该维度信任视觉（画面清晰、信息丰富）
    - gate ≈ 0 → 该维度信任文本（语义引导更重要）

    优势：参数量少（2×D×D），输出维度 = 输入维度，可解释性强。
    """

    def __init__(
        self,
        in_dim: int,
        fusion_input: str = "embedding",
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.fusion_input = fusion_input
        self.in_dim = in_dim
        # GatedFusion 的输出维度 = 输入维度（逐维门控，维度不变化）
        self.out_dim = in_dim

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=in_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        # 门控投影
        self.W_v = nn.Linear(in_dim, in_dim)
        self.W_t = nn.Linear(in_dim, in_dim)

    def forward(
        self,
        visual: Tensor,
        text_memory: Tensor,
    ) -> FusionOutput:
        """visual: (B, T, D), text_memory: (K, L, D) → FusionOutput."""
        text_context, attn_weights = _text_context_from_memory(
            visual, text_memory, self.cross_attn,
        )
        # 逐帧门控
        gate = torch.sigmoid(self.W_v(visual) + self.W_t(text_context))
        fused = gate * visual + (1.0 - gate) * text_context     # (B, T, D)
        return FusionOutput(
            fused=fused,                                         # (B, T, D)
            attention=attn_weights,                              # (B, T, K*L)
            gate_values=gate,                                    # (B, T, D)
        )

    def extra_repr(self) -> str:
        return f"in_dim={self.in_dim}, fusion_input={self.fusion_input!r}"


class CrossAttnFusion(nn.Module):
    """帧 × Token 交叉注意力融合 — 表达能力最强。

    流程：
    1. Q = visual (B, T, D), K/V = text_memory (B, K*L, D) → (B, T, D)
    2. Residual + LayerNorm
    3. Linear → (B, T, D_f)

    与 ConcatFusion / GatedFusion 的区别：
    - 前两者先 cross-attn 再独立融合（attn 只产生 context，融合在外部）
    - CrossAttnFusion 把 cross-attn 本身作为融合的核心操作
    - attention weights (B, T, K*L) 可直接用于帧到 token 的可解释性分析

    适用：有充足 GPU 资源，追求最高精度和可解释性。
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        fusion_input: str = "embedding",
    ) -> None:
        super().__init__()
        self.fusion_input = fusion_input
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.attn = nn.MultiheadAttention(
            embed_dim=in_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(
        self,
        visual: Tensor,
        text_memory: Tensor,
    ) -> FusionOutput:
        """visual: (B, T, D), text_memory: (K, L, D) → FusionOutput."""
        B, T, D = visual.shape
        K, L, _ = text_memory.shape

        # Text Memory → (B, K*L, D)
        tm = text_memory.view(1, K * L, D).expand(B, -1, -1)

        # Cross-Attention: Q=vis, K/V=text_memory
        attn_out, attn_weights = self.attn(query=visual, key=tm, value=tm)
        # attn_out: (B, T, D), attn_weights: (B, T, K*L)

        # 残差连接 + LayerNorm
        out = self.norm(visual + attn_out)                      # (B, T, D)
        return FusionOutput(
            fused=self.proj(out),                                # (B, T, D_f)
            attention=attn_weights,                              # (B, T, K*L)
            gate_values=None,
        )

    def extra_repr(self) -> str:
        return (
            f"in_dim={self.in_dim}, out_dim={self.out_dim}, "
            f"fusion_input={self.fusion_input!r}"
        )


# ═══════════════════════════════════════════════════════════════════
# Fusion 工厂
# ═══════════════════════════════════════════════════════════════════

def _select_fusion_input(
    *,
    aligned: "AlignmentOutput",  # noqa: F821  — forward ref
    fusion_input: str,
) -> tuple[Tensor, Tensor]:
    """根据 ``fusion_input`` 配置从 AlignmentOutput 中选择特征。

    这是 Fusion 模块的辅助函数，在 VLMModel.forward() 中调用。
    为什么不放在 Fusion 内部？
    — 因为 VLMModel 需要知道选择后的 in_dim 来正确构造 Fusion，
      所以选择逻辑放在 VLMModel，Fusion 只拿到选择后的结果。
    这个函数只供参考——实际选择在 VLMModel 中完成。
    """
    from .outputs import AlignmentOutput as AO  # noqa: F811

    if fusion_input == "embedding":
        return aligned.visual_embedding, aligned.text_embedding
    elif fusion_input == "raw":
        return aligned.visual_feature, aligned.text_feature
    elif fusion_input == "concat":
        vis = torch.cat([aligned.visual_feature, aligned.visual_embedding], dim=-1)
        txt = torch.cat([aligned.text_feature, aligned.text_embedding], dim=-1)
        return vis, txt
    else:
        raise ValueError(
            f"Unknown fusion_input={fusion_input!r}. "
            f"Expected 'embedding', 'raw', or 'concat'."
        )
