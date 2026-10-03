# 这个文件是整个 VLM-VAD 模型的顶层编排——把所有模块串联起来。
"""VLM-VAD: Open-Vocabulary Video Anomaly Detection with CLIP backbone.

完整 forward 流程（七层模块）：:

    video (B,T,C,H,W)
           │
           ▼
    1. Backbone.encode_video()          → vis_feat  (B, T, D)
    2. PromptProcessor.process()        → prompts   list[K str]
    3. Backbone.encode_text(prompts)    → txt_feat  (K, L, D)
    4. Alignment(vis_feat, txt_feat)    → AlignmentOutput
    5. Matcher(vis_pooled, txt_pooled)  → MatchOutput  (contrastive loss 用)
    6. Fusion(vis, text_memory)         → FusionOutput  (B, T, D_f)
    7. Temporal(fused)                  → TemporalOutput
    8. Head(temporal_feat)              → HeadOutput
    9. Matcher(embedding, txt_pooled)   → MatchOutput  (per-prompt anomaly)
           │
           ▼
    ModelOutput

关键设计约束：
- **B 不在任何模块中膨胀** — Fusion 使用 Text Memory Bank 模式
- **T 维全链路保留** — Backbone → Alignment → Fusion → Temporal → Head
- **所有输出为 frozen dataclass** — 类型安全，可直接传给 loss 函数
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .backbone import CLIPBackbone
from .alignment import SemanticAlignment
from .fusion import ConcatFusion, CrossAttnFusion, GatedFusion
from .temporal import TemporalIdentity, TemporalTransformer
from .head import AnomalyHead
from .outputs import (
    AlignmentOutput,
    MatchOutput,
    FusionOutput,
    TemporalOutput,
    HeadOutput,
    ModelOutput,
)


# ═══════════════════════════════════════════════════════════════════
# FusionInput 选择逻辑
# ═══════════════════════════════════════════════════════════════════

def _select_fusion_features(
    aligned: AlignmentOutput,
    fusion_input: str,
) -> tuple[Tensor, Tensor]:
    """根据 ``fusion_input`` 从 AlignmentOutput 中选择视觉/文本特征。

    这是 VLMModel 内部的辅助函数，确保选择的特征维度和 Fusion 的 in_dim 一致。
    """
    if fusion_input == "embedding":
        return aligned.visual_embedding, aligned.text_embedding
    elif fusion_input == "raw":
        return aligned.visual_feature, aligned.text_feature
    elif fusion_input == "concat":
        vis = torch.cat(
            [aligned.visual_feature, aligned.visual_embedding], dim=-1
        )
        txt = torch.cat(
            [aligned.text_feature, aligned.text_embedding], dim=-1
        )
        return vis, txt
    else:
        raise ValueError(
            f"Unknown fusion_input={fusion_input!r}. "
            f"Expected 'embedding', 'raw', or 'concat'."
        )


# ═══════════════════════════════════════════════════════════════════
# VLMModel — 顶层编排
# ═══════════════════════════════════════════════════════════════════

class VLMModel(nn.Module):
    """Open-Vocabulary 视频异常检测模型。

    把所有模块串联为一个端到端的 forward 路径。

    Matcher 不放在模型内部——对齐相似度由 trainer 在外部用
    :class:`Matcher` 计算：::

        output = model(video)
        # 外部计算对比 loss
        vis_pooled = output.alignment.visual_embedding.mean(dim=1)
        txt_pooled = output.alignment.text_embedding.mean(dim=1)
        match = matcher(vis_pooled, txt_pooled)
        contrastive_loss = F.cross_entropy(match.logits, labels)

    典型用法::

        model = VLMModel(
            backbone=CLIPBackbone("ViT-B-32"),
            prompt_processor=PromptProcessor(...),
            alignment=SemanticAlignment(512, 512),
            fusion=ConcatFusion(512, 512, fusion_input="embedding"),
            temporal=TemporalIdentity(),
            head=AnomalyHead(512, 256),
        )
        output = model(video)  # video: (B, T, C, H, W)

        # 或用预提取特征
        output = model.forward_from_visual(vis_feat)  # (B, T, D)
    """

    def __init__(
        self,
        *,
        backbone: CLIPBackbone,
        prompt_processor,  # PromptProcessor (avoid hard import)
        alignment: SemanticAlignment,
        fusion: ConcatFusion | GatedFusion | CrossAttnFusion,
        temporal: TemporalIdentity | TemporalTransformer,
        head: AnomalyHead,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.prompt_processor = prompt_processor
        self.alignment = alignment
        self.fusion = fusion
        self.temporal = temporal
        self.head = head

        # 验证 fusion_input 配置
        self._validate_dims()

    # ── 核心 forward ──────────────────────────────────────────────

    def forward(self, video: Tensor) -> ModelOutput:
        """端到端 forward（原始像素 → ViT → ... → ModelOutput）。

        Args:
            video: ``(B, T, C, H, W)`` — 一批视频 clip。

        Returns:
            ModelOutput — 包含最终预测和所有中间模块输出。
        """
        vis_feat = self.backbone.encode_video(video)       # (B, T, D)
        return self._forward_from_features(vis_feat)

    def forward_from_visual(self, vis_feat: Tensor) -> ModelOutput:
        """从预提取的视觉特征开始 forward（跳过 ViT）。

        配合 :class:`FeatureDataset` 使用——backbone 冻结时，离线提取
        所有视频帧的 CLIP 特征，训练时直接传入 ``vis_feat (B, T, D)``，
        不再经过 ViT。训练速度提升 ~10×。

        Args:
            vis_feat: ``(B, T, D)`` — 预提取的 L2 归一化 CLIP 视觉特征。

        Returns:
            ModelOutput — 与 ``forward()`` 完全相同的输出结构。
        """
        return self._forward_from_features(vis_feat)

    def _forward_from_features(self, vis_feat: Tensor) -> ModelOutput:
        """共享的 downstream 路径：prompt encoding → alignment → fusion → temporal → head。"""
        # ── 1. Prompt 处理 + Text Encoding ──────────────────
        prompts: list[str] = self.prompt_processor.process()
        txt_feat = self.backbone.encode_text(prompts)      # (K, L, D)

        # ── 2. Semantic Alignment ──────────────────────────
        aligned = self.alignment(vis_feat, txt_feat)        # AlignmentOutput

        # ── 3. Fusion（Text Memory 模式，B 不膨胀）─────────
        fusion_vis, fusion_txt = _select_fusion_features(
            aligned, self.fusion.fusion_input,
        )
        fused_out = self.fusion(fusion_vis, fusion_txt)     # FusionOutput

        # ── 4. Temporal ────────────────────────────────────
        temp_out = self.temporal(fused_out.fused)            # TemporalOutput

        # ── 5. Head ────────────────────────────────────────
        head_out = self.head(temp_out.features)              # HeadOutput

        # ── 6. 组装输出 ────────────────────────────────────
        return ModelOutput(
            frame_score=head_out.frame_score,                # (B, T)
            clip_score=head_out.clip_score,                  # (B,)
            embedding=head_out.embedding,                    # (B, D_f)
            alignment=aligned,
            fusion=fused_out,
            temporal=temp_out,
            head=head_out,
        )

    # ── 维度校验 ─────────────────────────────────────────────────

    def _validate_dims(self) -> None:
        """在构造时校验所有模块的维度是否一致。

        防止隐式 bug：如 fusion_input="concat" 但 in_dim 没加上 aligned_dim。
        """
        backbone_dim = self.backbone.dim
        alignment_dim = self.alignment.aligned_dim
        fusion_input = self.fusion.fusion_input
        fusion_in_dim = self.fusion.in_dim

        if fusion_input == "embedding":
            expected = alignment_dim
        elif fusion_input == "raw":
            expected = backbone_dim
        elif fusion_input == "concat":
            expected = backbone_dim + alignment_dim
        else:
            return  # 不应该到这里

        if fusion_in_dim != expected:
            raise ValueError(
                f"Fusion in_dim mismatch: "
                f"fusion.fusion_input={fusion_input!r}, "
                f"fusion.in_dim={fusion_in_dim}, "
                f"but expected {expected} "
                f"(backbone.dim={backbone_dim}, alignment.aligned_dim={alignment_dim})"
            )

    # ── 便捷方法 ─────────────────────────────────────────────────

    def freeze_backbone(self) -> None:
        """冻结 backbone 全部参数。"""
        self.backbone.requires_grad_(False)

    def unfreeze_backbone(self) -> None:
        """解冻 backbone 全部参数。"""
        self.backbone.requires_grad_(True)

    def trainable_params(self) -> list[str]:
        """返回当前可训练的参数名列表（调试用）。"""
        return [name for name, p in self.named_parameters() if p.requires_grad]

    def __repr__(self) -> str:
        parts = [
            f"backbone={self.backbone}",
            f"prompt_processor={self.prompt_processor}",
            f"alignment={self.alignment}",
            f"fusion={self.fusion}",
            f"temporal={self.temporal}",
            f"head={self.head}",
        ]
        return f"VLMModel(\n  " + ",\n  ".join(parts) + "\n)"
