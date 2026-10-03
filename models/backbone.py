# 这个文件负责视觉和文本编码器的统一管理。
"""CLIP-based backbone: visual and text encoders that never pool."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import open_clip


class CLIPBackbone(nn.Module):
    """CLIP 视觉 + 文本编码器，严格遵守"不池化"约定。

    设计契约（§3 of DESIGN.md）：
    - ``encode_video(video)``  → ``(B, T, D)``   帧级特征，不做时间池化
    - ``encode_text(prompts)`` → ``(K, L, D)``   Token 级序列，不取 EOS
    - ``dim`` 属性返回共享嵌入维度 D

    冻结 / 微调示例::

        backbone.vision_encoder.requires_grad_(False)   # 只微调文本
        backbone.text_encoder.requires_grad_(False)     # 只微调视觉
        backbone.requires_grad_(False)                  # 全部冻结
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
    ) -> None:
        super().__init__()

        # open_clip.create_model_and_transforms 返回 (model, _, preprocess)
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained,
        )
        self._clip = model
        # Tokenizer 和 text encoder 强绑定（BPE / vocab 必须一致），放在这里
        self._tokenizer = open_clip.get_tokenizer(model_name)

        # 把模型的参数移到正确的设备上已经在 create_model_and_transforms 之后完成；
        # 这里记录 text transformer 的内部宽度以便 encode_text 做 projection 判断。
        self._text_width: int = self._clip.transformer.width

    # ── 核心 API ────────────────────────────────────────────────────

    def encode_video(self, video: Tensor) -> Tensor:
        """对视频 clip 逐帧编码，返回帧级特征，**不池化**。

        Args:
            video: ``(B, T, C, H, W)`` — 一批视频 clip，T 是时间帧数。

        Returns:
            ``(B, T, D)`` — 帧级 L2 归一化特征。每帧由 CLIP image encoder
            独立编码为一个 D 维向量，T 维完整保留。
        """
        B, T = video.shape[:2]
        # 把 T 维并入 batch 维：每帧当作独立的图像
        video_flat = video.view(B * T, *video.shape[2:])          # (B*T, C, H, W)
        features = self._clip.encode_image(video_flat, normalize=True)  # (B*T, D)
        return features.view(B, T, -1)                            # (B, T, D)

    def encode_text(self, prompts: list[str]) -> Tensor:
        """对文本 prompts 逐 token 编码，**不取 EOS 池化**。

        与 :meth:`encode_video` 对称：两者都输出完整序列特征，让 Fusion 层
        自行决定如何池化。

        Args:
            prompts: K 个文本字符串（K 可以不等于 batch size B）。

        Returns:
            ``(K, L, D)`` — Token 级 L2 归一化特征。L = context_length（通常 77）。
            包含填充 token 但已做 L2 归一化；下游 CrossAttnFusion 可通过
            ``key_padding_mask`` 屏蔽填充位置。
        """
        tokens = self._tokenizer(prompts).to(self.device)         # (K, 77)

        # 复刻 CLIP.encode_text 的前半部分，但在 text_global_pool 之前停下来
        cast_dtype = self._clip.transformer.get_cast_dtype()
        x = self._clip.token_embedding(tokens).to(cast_dtype)     # (K, 77, D_text)
        x = x + self._clip.positional_embedding.to(cast_dtype)
        x = self._clip.transformer(x, attn_mask=self._clip.attn_mask)
        x = self._clip.ln_final(x)                                # (K, 77, D_text)

        # 如果存在 text_projection，逐 token 投影到共享嵌入空间
        if self._clip.text_projection is not None:
            x = x @ self._clip.text_projection                    # (K, 77, D)

        return F.normalize(x, dim=-1)                             # token 级 L2 归一化

    def encode_text_pooled(self, prompts: list[str]) -> Tensor:
        """完整 CLIP 文本编码（EOS 池化 + projection），返回 ``(K, D)``。

        与 :meth:`encode_text` 的区别：
            encode_text        → token 级序列 ``(K, L, D)``，供 Fusion 自己做池化
            encode_text_pooled → 标准 CLIP 文本表征 ``(K, D)``，EOS 池化后归一化

        Zero-shot 直接相似度匹配应使用后者（这也是 CLIP 官方用法），
        而训练时的 Fusion 仍使用 token 序列以保留细粒度语义。
        """
        tokens = self._tokenizer(prompts).to(self.device)
        return self._clip.encode_text(tokens, normalize=True)

    # ── 查询属性 ────────────────────────────────────────────────────

    @property
    def dim(self) -> int:
        """共享嵌入维度 D（visual 和 text projection 输出维度）。"""
        return self._clip.visual.output_dim

    @property
    def device(self) -> torch.device:
        """当前模型参数所在的设备。"""
        return next(self.parameters()).device

    # ── 子模块暴露（供 freeze / unfreeze 细粒度控制）────────────────

    @property
    def vision_encoder(self) -> nn.Module:
        """视觉编码器——即 CLIP 的 Vision Transformer。

        用法::

            backbone.vision_encoder.requires_grad_(False)   # 冻结视觉
        """
        return self._clip.visual

    @property
    def text_encoder(self) -> nn.Module:
        """文本编码器的核心 Transformer 部分。

        .. note::

            完整冻结文本侧还需要冻结 ``token_embedding``、``ln_final``
            和 ``text_projection``（如果存在）。推荐使用 :meth:`freeze_text`
            一键完成，或对 ``self._clip`` 整体操作。
        """
        return self._clip.transformer

    # ── 便捷冻结方法 ────────────────────────────────────────────────

    def freeze_vision(self) -> None:
        """冻结所有视觉参数（Vision Transformer 全部权重）。"""
        self._clip.visual.requires_grad_(False)

    def freeze_text(self) -> None:
        """冻结所有文本参数（transformer + token embedding + ln_final + projection）。"""
        self._clip.transformer.requires_grad_(False)
        self._clip.token_embedding.requires_grad_(False)
        self._clip.ln_final.requires_grad_(False)
        self._clip.positional_embedding.requires_grad_(False)
        if self._clip.text_projection is not None:
            self._clip.text_projection.requires_grad_(False)

    def unfreeze_vision(self) -> None:
        """解冻所有视觉参数。"""
        self._clip.visual.requires_grad_(True)

    def unfreeze_text(self) -> None:
        """解冻所有文本参数。"""
        self._clip.transformer.requires_grad_(True)
        self._clip.token_embedding.requires_grad_(True)
        self._clip.ln_final.requires_grad_(True)
        self._clip.positional_embedding.requires_grad_(True)
        if self._clip.text_projection is not None:
            self._clip.text_projection.requires_grad_(True)

    # ── 辅助 ────────────────────────────────────────────────────────

    def tokenize(self, prompts: list[str]) -> Tensor:
        """暴露 tokenizer 供外部 debug 或 mask 构造使用。

        Returns:
            ``(K, 77)`` — token id 矩阵，0 为填充位。
        """
        return self._tokenizer(prompts).to(self.device)

    def __repr__(self) -> str:
        return (
            f"CLIPBackbone("
            f"dim={self.dim}, "
            f"text_width={self._text_width}, "
            f"device={self.device})"
        )
