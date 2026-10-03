# 这个文件定义训练用的一切损失函数——异常分类 + 视觉-语言对比对齐。
"""Loss functions for VLM-VAD.

项目要求（AGENTS.md §6）至少两类 loss：

    1. ``AnomalyBCELoss``  —— 帧级异常分类（sigmoid + BCE）
    2. ``ContrastivePolarityLoss`` —— 视觉-文本对比对齐（soft-target 版 InfoNCE）

下面先讲清楚每个 loss 的数学，再看实现。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor

from models.matcher import Matcher
from models.outputs import ModelOutput
from utils.logging import get_logger

log = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 工具：prompt 极性标记（normal / abnormal）
# ═══════════════════════════════════════════════════════════════════

def build_prompt_polarity(
    prompts: list[str],
    cfg: DictConfig,
) -> Tensor:
    """根据关键词把每个 prompt 标记为 normal(-1) / abnormal(+1) / neutral(0)。

    数学/逻辑背景：
        VLM-VAD 的核心假设是"异常 = 与描述异常行为的文本语义相近"。
        对比学习需要一个正负关系，但我们没有"每个 prompt 是正还是负"的标注，
        只能从 prompt 文本本身推断极性——这是基于语义规则（rule-based）的
        伪标签（pseudo label）方法。

    规则：prompt 里出现任一 abnormal 关键词 → +1；
          出现任一 normal 关键词且不包含 abnormal 关键词 → -1；
          否则 → 0（中性，不参与对比）。

    Args:
        prompts: K 个 prompt 文本。
        cfg: 配置段，需含 ``normal_keywords`` 与 ``abnormal_keywords`` 列表。

    Returns:
        ``(K,)`` int64 张量，取值 {-1, 0, +1}。
    """
    normal_kw = list(cfg.get("normal_keywords", []))
    abnormal_kw = list(cfg.get("abnormal_keywords", []))

    polarity = torch.zeros(len(prompts), dtype=torch.int64)
    for i, p in enumerate(prompts):
        pl = p.lower()
        is_abnormal = any(kw.lower() in pl for kw in abnormal_kw)
        is_normal = any(kw.lower() in pl for kw in normal_kw)
        if is_abnormal:
            polarity[i] = 1
        elif is_normal:
            polarity[i] = -1
        # 两者都匹配或都不匹配 → 0（中性）
    return polarity


# ═══════════════════════════════════════════════════════════════════
# Loss 1：帧级异常分类
# ═══════════════════════════════════════════════════════════════════

class AnomalyBCELoss(nn.Module):
    """帧级异常分类损失——二元交叉熵（Binary Cross-Entropy）。

    数学定义：
        设 x_t 是第 t 帧的 logit，y_t ∈ {0,1} 是第 t 帧的真实标签，则
            BCE(x_t, y_t) = -[ y_t · ln σ(x_t) + (1-y_t) · ln(1-σ(x_t)) ]
        其中 σ(x) = 1 / (1 + e^{-x}) 是 sigmoid。

    从最大似然的角度看：sigmoid 输出 σ(x_t) 被解释为"第 t 帧异常的概率"，
    BCE 就是这个伯努利模型的负对数似然（NLL）。最小化 BCE ⟺ 最大化
    观测到真实标签的概率。

    实现用 ``BCEWithLogitsLoss`` 而不是手动算 sigmoid + BCE，原因（数值稳定性）：
        log(1 - σ(x)) = log(e^{-x} / (1 + e^{-x})) = -x - log(1 + e^{-x})
        当 x 很小时，直接算 log(1-σ(x)) 会 -inf；PyTorch 内部用 log-sum-exp
        恒等变形避免了这一点。

    为什么 frame 级 BCE 有意义：
        训练时我们把每个 clip 的每一帧当独立样本。异常帧被推向 score→1，
        正常帧被推向 score→0。clip_score = amax(frame_score) 继承了帧级监督。
    """

    def __init__(self) -> None:
        super().__init__()
        # pos_weight 可调类别不平衡（异常帧通常远少于正常帧）
        self._bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, output: ModelOutput, frame_label: Tensor) -> Tensor:
        """计算帧级 BCE。

        Args:
            output: 模型输出，取 ``head.frame_logits (B, T)``。
            frame_label: ``(B, T)`` int64，每帧是否异常。

        Returns:
            标量 loss。
        """
        logits = output.head.frame_logits            # (B, T)
        target = frame_label.to(logits.dtype)        # (B, T) → float
        loss = self._bce(logits, target)             # (B, T)
        return loss.mean()


# ═══════════════════════════════════════════════════════════════════
# Loss 2：视觉-文本对比对齐（Contrastive）
# ═══════════════════════════════════════════════════════════════════

class ContrastivePolarityLoss(nn.Module):
    """对比对齐损失——把视觉特征与"正确极性"的 prompt 拉近。

    核心思想（CLIP 训练目标的简化 + 极性版本）：
        CLIP 原版：给定 batch 里 B 张图和 B 句文本，第 i 张图与第 i 句文本是
        正样本对，其余是负样本对，用对称的 InfoNCE 拉近正对、推开负对。

        我们的场景：没有逐 clip 的文本标注，但每个 clip 有极性标签
        （clip 正常 → 应对齐 normal prompts；clip 异常 → 应对齐 abnormal prompts）。

        所以把任务从"检索正确的那一句"改成"检索正确极性的那一组"：
            对第 b 个 clip，它在 K 个 prompt 上的目标分布 q_b 是
            "该 clip 极性对应组内的 prompt 均匀分布"。

    数学定义（soft-target 交叉熵）：
        模型先算 clip 级视觉特征 v_b（池化），再用 Matcher 得到
            z_bk = logits(v_b, t_k)   # (B, K)，temperature-scaled
        预测分布：p_b = softmax(z_b)
        目标分布：q_b ∈ R^K，q_bk = 1/|S_b| if k∈S_b else 0
                 其中 S_b = {k : polarity_k == +1} 当 clip 异常；
                        S_b = {k : polarity_k == -1} 当 clip 正常。
        loss = - (1/B) Σ_b Σ_k q_bk · ln p_bk
              = (1/B) Σ_b KL(q_b ‖ p_b)

    直觉：
        - 拉近：正常 clip 的视觉向量与 "walking normally" 等 normal prompt 相似度↑
        - 推开：同一 clip 与 "fighting" 等 abnormal prompt 相似度↓（softmax 归一化后此消彼长）
        训练后的共享空间里，"正常画面 ↔ 正常语义"、"异常画面 ↔ 异常语义"，
        这为**零样本泛化**打下基础（没见过的异常 prompt 也能被正确匹配）。

    注意：该 loss 需要 matcher 实例（模型里没有，由 trainer 传入），
    这符合 DESIGN.md "Matcher 从模型解耦" 的约定。
    """

    def __init__(self, matcher: Matcher, temperature: float | None = None) -> None:
        super().__init__()
        self.matcher = matcher
        if temperature is not None:
            # 手动覆盖温度（默认用 matcher 学到的 logit_scale）
            self.matcher.set_temperature(temperature)

    def _pool_prompts(self, text_embedding: Tensor) -> Tensor:
        """把 token 级文本特征池化成 clip 可匹配的 prompt 级向量。

        Args:
            text_embedding: ``(K, L, D_a)``

        Returns:
            ``(K, D_a)`` —— 平均池化。
        """
        return text_embedding.mean(dim=1)

    def forward(
        self,
        output: ModelOutput,
        clip_label: Tensor,
        polarity: Tensor,
    ) -> Tensor:
        """计算对比损失。

        Args:
            output: 模型输出，取 ``alignment.visual_embedding (B,T,D_a)``
                    与 ``alignment.text_embedding (K,L,D_a)``。
            clip_label: ``(B,)`` —— 每个 clip 是否异常（0 正常 / 1 异常）。
            polarity: ``(K,)`` —— 每个 prompt 的极性 {+1 异常, -1 正常, 0 中性}。

        Returns:
            标量 loss（若没有可对比的 prompt 组则返回 0）。
        """
        B = clip_label.shape[0]
        K = polarity.shape[0]

        # ── 1. 池化 ──────────────────────────────────────────────
        vis_pooled = output.alignment.visual_embedding.mean(dim=1)  # (B, D_a)
        txt_pooled = self._pool_prompts(output.alignment.text_embedding)  # (K, D_a)

        # ── 2. 相似度矩阵 ────────────────────────────────────────
        match = self.matcher(vis_pooled, txt_pooled)   # logits (B, K)
        logits = match.logits

        # ── 3. 构造 soft target q_b ──────────────────────────────
        # 异常 prompt 的 index 与正常 prompt 的 index
        abnormal_idx = torch.nonzero(polarity == 1, as_tuple=False).squeeze(-1)
        normal_idx = torch.nonzero(polarity == -1, as_tuple=False).squeeze(-1)

        # 任一组为空 → 无法做对比，返回 0
        if abnormal_idx.numel() == 0 or normal_idx.numel() == 0:
            log.warning("Contrastive loss skipped: need both normal and abnormal prompts.")
            return logits.sum() * 0.0  # 保持计算图（可回传 0 梯度）

        # 每个 clip 的目标组：正常 clip → normal_idx；异常 clip → abnormal_idx
        target_idx = torch.where(
            clip_label > 0,                         # clip 异常？
            abnormal_idx.new_full((B,), 1),         # 是 → 用 abnormal 组
            abnormal_idx.new_full((B,), -1),        # 否 → 用 normal 组
        )
        # target_idx 现在是 (B,) 的 ±1 标记，下面按极性把每组展开为 one-hot
        target = torch.zeros(B, K, device=logits.device)
        for b in range(B):
            group = abnormal_idx if target_idx[b] == 1 else normal_idx
            target[b, group] = 1.0 / group.numel()

        # ── 4. soft-target 交叉熵 ────────────────────────────────
        # CE(q, p) = -Σ q·ln p = -Σ q·log_softmax(z)
        loss = -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        return loss.mean()


# ═══════════════════════════════════════════════════════════════════
# 组合 loss
# ═══════════════════════════════════════════════════════════════════

class VLMVADLoss(nn.Module):
    """总损失 = w_bce · BCE(帧级) + w_ctr · Contrastive(极性对齐)。

    超参数含义：
        - ``w_bce``：帧级异常分类的权重。越大 → 模型越"就事论事"地判断每帧。
        - ``w_contrastive``：语义对齐的权重。越大 → 模型越依赖 prompt 语义。

    一般经验：vlm 类的语义对齐对零样本泛化至关重要，contrastive 不宜太小；
    但 frame 监督直接决定 AUC，BCE 是主 loss。默认 w_bce=1.0, w_ctr=0.5。

    返回 dict（而不是标量）是为了：
        1. 训练时分别 logging 两个 loss 分量，看清谁在驱动收敛
        2. 方便做 loss 分量消融（ablation）
    """

    def __init__(
        self,
        matcher: Matcher,
        w_bce: float = 1.0,
        w_contrastive: float = 0.5,
        skip_bce_when_no_pos: bool = True,
    ) -> None:
        super().__init__()
        self.w_bce = w_bce
        self.w_contrastive = w_contrastive
        self.skip_bce_when_no_pos = skip_bce_when_no_pos
        self._bce = AnomalyBCELoss()
        self._contrastive = ContrastivePolarityLoss(matcher)

    def forward(
        self,
        output: ModelOutput,
        frame_label: Tensor,
        clip_label: Tensor,
        polarity: Tensor,
    ) -> dict[str, Tensor]:
        """计算组合 loss。

        Args:
            output: 模型输出。
            frame_label: ``(B, T)`` 帧级标签。
            clip_label: ``(B,)`` clip 级标签。
            polarity: ``(K,)`` prompt 极性。

        Returns:
            {"total": Tensor, "bce": Tensor, "contrastive": Tensor}。
        """
        # ── 1. 帧级 BCE ──────────────────────────────────────────
        if self.skip_bce_when_no_pos:
            # 防御：若整个 batch 没有异常帧，BCE 退化为"全推 0"，会产生
            # 假收敛。此时跳过，只保留对比信号。
            has_pos = bool((frame_label > 0).any().item())
            bce = self._bce(output, frame_label) if has_pos else output.head.frame_score.sum() * 0.0
        else:
            bce = self._bce(output, frame_label)

        # ── 2. 对比对齐 ──────────────────────────────────────────
        contrastive = self._contrastive(output, clip_label, polarity)

        # ── 3. 加权求和 ──────────────────────────────────────────
        total = self.w_bce * bce + self.w_contrastive * contrastive
        return {"total": total, "bce": bce, "contrastive": contrastive}
