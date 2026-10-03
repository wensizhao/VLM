# 这个文件负责把 OmegaConf 配置解析成可用的模型组件——配置驱动（config-driven）。
"""Model factory — build prompt processor / matcher / VLM model from OmegaConf config.

为什么要一个 factory？
-----------------------
项目要求"prompt / fusion / backbone 三向实验对比"（AGENTS.md §8）。如果模型在
脚本里硬编码，换一个配置就要改代码。Factory 把"配置 → 对象"的映射集中管理：

    model = build_model(cfg)          # 一个入口，三种 fusion / 两种 temporal 自由切换
    pp    = build_prompt_processor(cfg.prompt)
    matcher = build_matcher(cfg.model.matcher)

好处：
- 换 backbone（ViT-B-32 → ViT-B-16）、换 fusion（concat → crossattn）、
  换 prompt 类型都只需要改 yaml，不需要碰任何代码
- 维度一致性校验（VLMModel._validate_dims）在构造时就抛错，避免隐性 bug
"""

from __future__ import annotations

from omegaconf import DictConfig

from models import VLMModel
from models.alignment import SemanticAlignment
from models.backbone import CLIPBackbone
from models.fusion import ConcatFusion, CrossAttnFusion, GatedFusion
from models.head import AnomalyHead
from models.matcher import Matcher
from models.temporal import TemporalIdentity, TemporalTransformer
from prompts import PromptManager
from prompts.processor import PromptProcessor


def build_prompt_processor(prompt_cfg: DictConfig) -> PromptProcessor:
    """从配置构造 PromptProcessor。

    配置里的 ``templates`` / ``expansions`` / ``hard_negatives`` 与
    PromptProcessor 构造参数一一对应（见 configs/prompts.yaml）。

    Args:
        prompt_cfg: OmegaConf 配置段，至少包含 ``templates``。

    Returns:
        PromptProcessor 实例。
    """
    templates = dict(prompt_cfg.get("templates", {}))
    if not templates:
        # 配置为空时退回到内置默认模板（label / scene / contrast）
        templates = dict(PromptManager().to_dict())

    expansions = (
        {k: list(v) for k, v in prompt_cfg.get("expansions", {}).items()}
        if "expansions" in prompt_cfg and prompt_cfg.get("expansions")
        else None
    )
    hard_negatives = (
        list(prompt_cfg.get("hard_negatives", []))
        if "hard_negatives" in prompt_cfg
        else None
    )
    return PromptProcessor(
        templates=templates,
        expansions=expansions,
        hard_negatives=hard_negatives,
        cache_enabled=bool(prompt_cfg.get("cache_enabled", True)),
    )


def build_matcher(model_cfg: DictConfig) -> Matcher:
    """从配置构造 Matcher。

    Args:
        model_cfg: ``cfg.model`` 配置段（matcher + alignment 子段）。
            需要 alignment.aligned_dim 来计算 learnable 策略的 W 维度。

    Returns:
        Matcher 实例（注意：contrastive loss 训练用的就是这个 matcher）。
    """
    matcher_cfg = model_cfg.matcher
    strategy = str(matcher_cfg.get("strategy", "cosine"))
    in_dim = None
    if strategy == "learnable":
        # learnable 策略的 W 是 (in_dim, in_dim)，作用于池化后的共享空间向量，
        # 因此 in_dim = aligned_dim（并要求 head 输出维度与之相等才能复用）
        in_dim = int(model_cfg.alignment.get("aligned_dim", 512))
    return Matcher(
        strategy=strategy,
        temperature=float(matcher_cfg.get("temperature", 0.07)),
        learnable_temp=bool(matcher_cfg.get("learnable_temp", True)),
        in_dim=in_dim,
    )


def _build_backbone(cfg: DictConfig) -> CLIPBackbone:
    """构造 backbone 并按配置冻结 vision/text。"""
    backbone = CLIPBackbone(
        model_name=str(cfg.backbone.name),
        pretrained=str(cfg.backbone.pretrained),
    )
    # 冻结选项：设计里"backbone 冻结 + 预提取特征"是快速训练路径
    if cfg.backbone.get("freeze_vision", False):
        backbone.freeze_vision()
    if cfg.backbone.get("freeze_text", False):
        backbone.freeze_text()
    return backbone


def _build_fusion(cfg: DictConfig, in_dim: int) -> (
    ConcatFusion | GatedFusion | CrossAttnFusion
):
    """根据配置构造 Fusion（三种策略统一入口）。"""
    ftype = str(cfg.fusion.get("type", "concat"))
    fusion_input = str(cfg.fusion.get("fusion_input", "embedding"))
    out_dim = int(cfg.fusion.get("out_dim", in_dim))

    if ftype == "concat":
        return ConcatFusion(in_dim, out_dim, fusion_input=fusion_input)
    if ftype == "gated":
        return GatedFusion(in_dim, fusion_input=fusion_input)
    if ftype == "crossattn":
        return CrossAttnFusion(
            in_dim, out_dim,
            num_heads=int(cfg.fusion.get("num_heads", 4)),
            fusion_input=fusion_input,
        )
    raise ValueError(f"Unknown fusion type: {ftype!r} (expected concat/gated/crossattn)")


def _build_temporal(cfg: DictConfig, dim: int) -> TemporalIdentity | TemporalTransformer:
    """根据配置构造 Temporal 模块。"""
    ttype = str(cfg.temporal.get("type", "identity"))
    if ttype == "identity":
        return TemporalIdentity()
    if ttype == "transformer":
        return TemporalTransformer(
            dim=dim,
            num_heads=int(cfg.temporal.get("num_heads", 4)),
            num_layers=int(cfg.temporal.get("num_layers", 2)),
        )
    raise ValueError(f"Unknown temporal type: {ttype!r} (expected identity/transformer)")


def build_model(cfg: DictConfig) -> VLMModel:
    """从完整配置构造 VLMModel（backbone + alignment + fusion + temporal + head）。

    数学上需要理解的一个点：各模块的输入维度必须"对接得上"。

        fusion_input = "embedding" → Fusion 输入 = aligned_dim（投影后维度）
        fusion_input = "raw"      → Fusion 输入 = backbone.dim（原始维度）
        fusion_input = "concat"   → Fusion 输入 = backbone.dim + aligned_dim

    所以构造顺序是：先建 backbone 拿到 ``dim``，再按 ``fusion_input``
    计算 Fusion 的 ``in_dim``。VLMModel._validate_dims 会二次校验。
    """
    backbone = _build_backbone(cfg.model)
    aligned_dim = int(cfg.model.alignment.get("aligned_dim", backbone.dim))
    proj_type = str(cfg.model.alignment.get("proj_type", "linear"))

    alignment = SemanticAlignment(
        in_dim=backbone.dim,
        aligned_dim=aligned_dim,
        proj_type=proj_type,
    )

    # Fusion 的输入维度由 fusion_input 决定（见上面的注释）
    fusion_input = str(cfg.model.fusion.get("fusion_input", "embedding"))
    if fusion_input == "embedding":
        fusion_in_dim = aligned_dim
    elif fusion_input == "raw":
        fusion_in_dim = backbone.dim
    elif fusion_input == "concat":
        fusion_in_dim = backbone.dim + aligned_dim
    else:
        raise ValueError(f"Unknown fusion_input: {fusion_input!r}")

    fusion = _build_fusion(cfg.model, fusion_in_dim)
    temporal = _build_temporal(cfg.model, fusion.out_dim)
    head = AnomalyHead(
        in_dim=fusion.out_dim,
        hidden_dim=int(cfg.model.head.get("hidden_dim", 256)),
        dropout=float(cfg.model.head.get("dropout", 0.1)),
    )

    pp = build_prompt_processor(cfg.prompt)

    model = VLMModel(
        backbone=backbone,
        prompt_processor=pp,
        alignment=alignment,
        fusion=fusion,
        temporal=temporal,
        head=head,
    )

    # 维度一致性由 VLMModel._validate_dims() 保证（构造时若 mismatch 会抛 ValueError）
    return model
