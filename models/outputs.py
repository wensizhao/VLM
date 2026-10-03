# 这个文件定义了所有模块的统一输出结构，确保数据流类型安全、调试友好。
"""Frozen dataclass outputs for every module in the VLM-VAD pipeline.

每个模块返回一个不可变的 dataclass，训练 loss 和 debug 工具可以直接通过
属性名访问中间层输出，而不用依赖 dict key 字符串。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor


# ═══════════════════════════════════════════════════════════════════
# 基础接口：所有输出 dataclass 的公共父类
# ═══════════════════════════════════════════════════════════════════

class _ModuleOutput:
    """所有模块输出的基类——提供 dict 转换和 compact repr。"""

    def to_dict(self) -> dict[str, Any]:
        """将 dataclass 转为 dict，方便传给 loss 函数或序列化。"""
        import dataclasses
        return dataclasses.asdict(self)

    def _compact(self, max_len: int = 80) -> str:
        """紧凑的 repr，截断过长的 shape 列表。"""
        import dataclasses
        parts: list[str] = []
        for field in dataclasses.fields(self):
            val = getattr(self, field.name)
            if isinstance(val, Tensor):
                parts.append(f"{field.name}=Tensor{tuple(val.shape)}")
            elif val is None:
                parts.append(f"{field.name}=None")
            elif isinstance(val, float):
                parts.append(f"{field.name}={val:.4f}")
            else:
                parts.append(f"{field.name}={val!r}")
        body = ", ".join(parts)
        if len(body) > max_len:
            body = body[:max_len - 3] + "..."
        return f"{type(self).__name__}({body})"

    def __repr__(self) -> str:
        return self._compact()


# ═══════════════════════════════════════════════════════════════════
# 各模块输出定义
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AlignmentOutput(_ModuleOutput):
    """SemanticAlignment 模块输出。

    同时保留 backbone 原始特征和投影到共享空间后的特征，
    使得下游 Fusion 可以通过 ``fusion_input`` 自由选择使用 raw / embedding / concat。

    Attributes:
        visual_feature: Backbone 原始视觉特征 ``(B, T, D)``。
        visual_embedding: 投影到共享语义空间后的视觉特征 ``(B, T, D_a)``。
        text_feature: Backbone 原始文本特征 ``(K, L, D)``。
        text_embedding: 投影到共享语义空间后的文本特征 ``(K, L, D_a)``。
    """

    visual_feature: Tensor       # (B, T, D)
    visual_embedding: Tensor     # (B, T, D_a)
    text_feature: Tensor         # (K, L, D)
    text_embedding: Tensor       # (K, L, D_a)


@dataclass(frozen=True)
class MatchOutput(_ModuleOutput):
    """Matcher 模块输出。

    ``similarity`` 是原始相似度（用于分析），``logits`` 是 temperature-scaled
    版本（用于 cross-entropy loss）。两者有明确分工。

    Attributes:
        similarity: 余弦相似度矩阵 ``(B, K)``，值域 ``[-1, 1]``。
        logits: Temperature-scaled logits ``(B, K)``，用于对比 loss 的输入。
        temperature: 当前温度值（标量）。
    """

    similarity: Tensor           # (B, K)
    logits: Tensor               # (B, K)
    temperature: float           # scalar


@dataclass(frozen=True)
class FusionOutput(_ModuleOutput):
    """Fusion 模块输出 — B 不膨胀，T 保留。

    可选字段 ``attention`` 和 ``gate_values`` 只在特定 Fusion 策略下非 None：
    - ``attention``: CrossAttnFusion / ConcatFusion（文本上下文注意力）
    - ``gate_values``: GatedFusion（逐帧门控系数）

    Attributes:
        fused: 融合后特征 ``(B, T, D_f)``。
        attention: 帧到 token 的交叉注意力权重 ``(B, T, K*L)``，或 None。
        gate_values: 逐帧门控系数 ``(B, T, D)``，或 None。
    """

    fused: Tensor                # (B, T, D_f)
    attention: Tensor | None     # (B, T, K*L) or None
    gate_values: Tensor | None   # (B, T, D) or None


@dataclass(frozen=True)
class TemporalOutput(_ModuleOutput):
    """Temporal 模块输出。

    Attributes:
        features: 时序建模后的帧级特征 ``(B, T, D_f)``。T 维完整保留。
    """

    features: Tensor             # (B, T, D_f)


@dataclass(frozen=True)
class HeadOutput(_ModuleOutput):
    """AnomalyHead 模块输出。

    ``frame_score`` 是真正的帧级预测（不是从 clip 复制），
    ``clip_score`` 从帧级聚合而来（``amax``）。
    这一"从细到粗"的设计确保帧级输出可用于 temporal heatmap 可视化。

    ``frame_logits`` 是 sigmoid 之前的原始 logits——训练时用它计算
    ``BCEWithLogitsLoss``。为什么？看数学：
        BCE(x, y) = -[y·ln σ(x) + (1-y)·ln(1-σ(x))]
        直接把 logits 塞进这个式子，在 |x| 很大时 σ(x) 会饱和（梯度趋近 0），
        但 PyTorch 的 BCEWithLogitsLoss 在内部用 log-sum-exp 恒等式：
            BCE(x,y) = max(x,0) - x·y + ln(1 + e^{-|x|})
        这个形式数值稳定，不会出现 log(0) 或下溢。
        所以"训练用 logits + BCEWithLogits，推理用 sigmoid"是标准做法。

    Attributes:
        frame_logits: 帧级原始 logits ``(B, T)``（未过 sigmoid）。
        frame_score: 帧级异常分数 ``(B, T)``，值域 ``[0, 1]``。
        clip_score: Clip 级异常分数 ``(B,)``，``amax(frame_score)``。
        embedding: Clip 级全局特征 ``(B, D_f)``，``mean(fused, dim=1)``。
    """

    frame_logits: Tensor         # (B, T)  — 未过 sigmoid
    frame_score: Tensor          # (B, T)
    clip_score: Tensor           # (B,)
    embedding: Tensor            # (B, D_f)


@dataclass(frozen=True)
class ModelOutput(_ModuleOutput):
    """VLMModel 顶层输出 — 包含所有中间模块输出，一站式访问。

    训练时：
    - ``alignment`` 提供 pooled 特征 → 外部 Matcher 计算对比学习 loss
    - ``head.frame_score`` 用于计算异常检测 loss（BCE）
    - ``fusion.attention`` 可用于可解释性正则化

    推理时：
    - ``frame_score`` 画 temporal heatmap
    - ``clip_score`` 做 clip 级异常判断
    - ``fusion.attention`` 做帧到 prompt 的可解释性可视化

    Attributes:
        frame_score: 最终帧级异常分数 ``(B, T)``。
        clip_score: 最终 clip 级异常分数 ``(B,)``。
        embedding: Clip 级全局特征 ``(B, D_f)``。
        alignment: 对齐模块的完整输出（含原始特征和投影特征）。
        fusion: 融合模块的完整输出（含 attention / gate_values）。
        temporal: 时序模块的完整输出。
        head: 预测头的完整输出。
    """

    frame_score: Tensor          # (B, T)
    clip_score: Tensor           # (B,)
    embedding: Tensor            # (B, D_f)
    alignment: AlignmentOutput
    fusion: FusionOutput
    temporal: TemporalOutput
    head: HeadOutput
