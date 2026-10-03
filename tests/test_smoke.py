# 这个文件是仓库的最小冒烟测试——不下载权重、不需要数据集，保证核心链路不崩。
"""Minimal smoke tests for the VLM-VAD pipeline.

覆盖 AGENTS.md 要求的关键契约（全部可在 CPU 上秒级跑完）：

1. **配置加载** — ``configs/experiment.yaml`` 与 ``configs/prompts.yaml`` 能合并，
   且 ``frame_label_mode`` 与仓库自带数据一致。
2. **Prompt 极性** — ``build_prompt_polarity`` 能把 prompt 打成 {+1 异常, -1 正常}。
3. **评估指标** — frame/clip/video 三级 AUC & AP 数值正确，单类输入有防御。
4. **模型 forward shape** — ``build_model`` → ``forward`` 输出满足
   ``ModelOutput`` 契约 ``frame_score (B,T) / clip_score (B,) / embedding (B,D_f)``。
5. **Loss 闭环** — ``VLMVADLoss`` 返回有限标量且可以 ``backward()``。

为什么用 :class:`FakeCLIPBackbone`？
    真实 ``CLIPBackbone`` 需要下载 ~350MB 的 open_clip 权重，CI 里既慢又依赖网络。
    这里用一个**接口与形状完全一致**的替身（``dim`` / ``encode_video`` /
    ``encode_text`` / ``freeze_vision`` / ``freeze_text``），把 factory →
    alignment → fusion → temporal → head 的整条维度链路照常跑一遍。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from eval.metrics import clip_level_metrics, frame_level_metrics, video_level_metrics
from models.factory import build_matcher, build_model, build_prompt_processor
from train.losses import VLMVADLoss, build_prompt_polarity
from utils.config import load_experiment_config, save_config_snapshot

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _PROJECT_ROOT / "configs" / "experiment.yaml"

# ── forward shape 的维度常量（避免 magic number）──────────────────────
_BATCH = 2
_CLIP_LEN = 8          # 测试用短 clip（真实训练用 16）
_IMG_CHANNELS = 3
_IMG_SIZE = 32         # FakeBackbone 只做均值池化，分辨率不影响语义
_FAKE_DIM = 512        # 与 configs/experiment.yaml: model.alignment.aligned_dim 一致
_TEXT_SEQ_LEN = 77     # CLIP 文本 token 长度


@pytest.fixture(scope="session", autouse=True)
def _run_from_project_root():
    """把 cwd 切到仓库根目录——配置里的相对路径（如 ``configs/prompts.yaml``）依赖 cwd。"""
    prev = Path.cwd()
    os.chdir(_PROJECT_ROOT)
    yield
    os.chdir(prev)


@pytest.fixture(scope="module")
def cfg():
    """加载合并后的完整实验配置（experiment.yaml + prompts.yaml）。"""
    return load_experiment_config(_CONFIG_PATH)


class FakeCLIPBackbone(nn.Module):
    """``CLIPBackbone`` 的轻量替身——接口/形状一致，不下载预训练权重。

    Args:
        model_name: open_clip 模型名（仅透传，不使用）。
        pretrained: 预训练权重标签（仅透传，不使用）。
        dim: 视觉/文本特征维度（真实 CLIP ViT-B-32 为 512）。
        seq_len: 文本 token 序列长度（CLIP 为 77）。
    """

    def __init__(
        self,
        model_name: str = "fake-clip",
        pretrained: str = "none",
        dim: int = _FAKE_DIM,
        seq_len: int = _TEXT_SEQ_LEN,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self._dim = dim
        self._seq_len = seq_len
        # 帧均值 (B,T,3) → (B,T,dim)，参数化以便 backward 有真实梯度
        self.frame_proj = nn.Linear(_IMG_CHANNELS, dim)

    @property
    def dim(self) -> int:
        return self._dim

    def freeze_vision(self) -> None:  # 与 CLIPBackbone 对齐的接口
        pass

    def freeze_text(self) -> None:
        pass

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """``(B,T,C,H,W)`` → ``(B,T,D)``（对空间维做均值再投影）。"""
        return self.frame_proj(video.mean(dim=(3, 4)))

    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        """``list[str]`` → ``(K,L,D)``——确定性的非零正弦编码（形状/数值可复现）。"""
        k = len(prompts)
        d, l = self._dim, self._seq_len
        base = torch.arange(d, dtype=torch.float32)
        tokens = torch.sin(
            torch.arange(l).view(1, l, 1)
            + base.view(1, 1, d)
            + torch.arange(k, dtype=torch.float32).view(k, 1, 1)
        )
        return tokens  # (K, L, D)


def _build_test_model(cfg) -> tuple[torch.nn.Module, list[str]]:
    """用 FakeCLIPBackbone 走真实的 ``models.factory.build_model``，返回 (model, prompts)。"""
    import models.factory as factory

    original = factory.CLIPBackbone
    factory.CLIPBackbone = FakeCLIPBackbone
    try:
        model = build_model(cfg)
    finally:
        factory.CLIPBackbone = original
    return model, build_prompt_processor(cfg.prompt).process()


# ═══════════════════════════════════════════════════════════════════
# 1. 配置加载
# ═══════════════════════════════════════════════════════════════════

def test_load_experiment_config(cfg):
    assert cfg.seed == 42
    # prompts.yaml 已内联到 cfg.prompt
    assert set(cfg.prompt.templates) == {"label", "scene", "contrast"}
    assert len(cfg.prompt.normal_keywords) > 0
    assert len(cfg.prompt.abnormal_keywords) > 0
    # README 与 config 必须一致：仓库默认用运动伪标签（见 README「关于 Avenue 数据集标注」）
    assert cfg.data.frame_label_mode == "motion_diff"
    # 模型可实验维度（AGENTS.md §8）必须齐全
    assert cfg.model.fusion.type in {"concat", "gated", "crossattn"}
    assert cfg.model.temporal.type in {"identity", "transformer"}


def test_config_snapshot_roundtrip(cfg, tmp_path):
    out = save_config_snapshot(cfg, tmp_path / "config.yaml")
    assert out.is_file()
    reloaded = load_experiment_config(out)
    assert reloaded.seed == cfg.seed
    assert reloaded.data.frame_label_mode == cfg.data.frame_label_mode


# ═══════════════════════════════════════════════════════════════════
# 2. Prompt 极性
# ═══════════════════════════════════════════════════════════════════

def test_prompt_polarity(cfg):
    prompts = build_prompt_processor(cfg.prompt).process()
    assert len(prompts) > 0

    polarity = build_prompt_polarity(prompts, cfg.prompt)
    assert polarity.shape == (len(prompts),)
    assert polarity.dtype == torch.int64
    # 集合 {+1, -1} 都必须存在，否则对比 loss 会退化为 0（见 train/losses.py）
    assert bool((polarity == 1).any()), "缺少 abnormal(+1) prompt"
    assert bool((polarity == -1).any()), "缺少 normal(-1) prompt"
    # 极性取值只能是 {-1, 0, +1}
    assert set(polarity.unique().tolist()) <= {-1, 0, 1}

    # 已知词表的确定性检查
    assert polarity[prompts.index("a person fighting")] == 1
    assert polarity[prompts.index("a person walking")] == -1


def test_prompt_processor_hard_negatives(cfg):
    processor = build_prompt_processor(cfg.prompt)
    base = processor.process()
    with_neg = processor.process(hard_negatives=True)
    assert len(with_neg) >= len(base)
    assert set(base) <= set(with_neg)


# ═══════════════════════════════════════════════════════════════════
# 3. 评估指标
# ═══════════════════════════════════════════════════════════════════

def test_frame_metrics_perfect_and_inverted():
    labels = np.array([0, 0, 1, 1])
    good = np.array([0.1, 0.2, 0.8, 0.9])
    assert frame_level_metrics(good, labels)["frame_auc"] == pytest.approx(1.0)
    assert frame_level_metrics(good, labels)["frame_ap"] == pytest.approx(1.0)
    # 分数反向 → AUC = 0（不是随机 0.5）
    assert frame_level_metrics(good[::-1].copy(), labels)["frame_auc"] == pytest.approx(0.0)


def test_metrics_single_class_guard():
    """全正常帧（单类）时 AUC 未定义 → 指标必须返回 0.0 而不是抛异常。"""
    scores = np.array([0.1, 0.2, 0.3])
    labels = np.zeros(3, dtype=np.int64)
    assert frame_level_metrics(scores, labels)["frame_auc"] == 0.0
    assert clip_level_metrics(scores, labels)["clip_auc"] == 0.0
    assert video_level_metrics({"v1": 0.2}, {"v1": 0})["video_auc"] == 0.0


def test_video_metrics_dict_order_independent():
    scores = {"v1": 0.2, "v2": 0.9}
    labels = {"v2": 1, "v1": 0}  # 顺序不同也必须按 video_id 对齐
    assert video_level_metrics(scores, labels)["video_auc"] == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════
# 4. 模型 forward shape（ModelOutput 契约）
# ═══════════════════════════════════════════════════════════════════

def test_model_forward_shapes(cfg):
    model, prompts = _build_test_model(cfg)
    model.eval()

    video = torch.randn(_BATCH, _CLIP_LEN, _IMG_CHANNELS, _IMG_SIZE, _IMG_SIZE)
    with torch.no_grad():
        out = model(video)

    fusion_dim = int(cfg.model.fusion.out_dim)
    assert out.frame_score.shape == (_BATCH, _CLIP_LEN)
    assert out.clip_score.shape == (_BATCH,)
    assert out.embedding.shape == (_BATCH, fusion_dim)
    # 分数语义：frame_score ∈ [0,1]，clip_score = amax(frame_score)（见 models/head.py）
    assert torch.all(out.frame_score >= 0) and torch.all(out.frame_score <= 1)
    assert torch.allclose(out.clip_score, out.frame_score.amax(dim=1))
    # 文本分支形状（alignment 保留 K、L）
    k = len(prompts)
    assert out.alignment.text_embedding.shape[:2] == (k, _TEXT_SEQ_LEN)


def test_model_forward_from_visual_shapes(cfg):
    """预提取特征路径（FeatureDataset / 离线特征训练）必须与像素路径输出同构。"""
    model, _ = _build_test_model(cfg)
    model.eval()

    vis_feat = torch.randn(_BATCH, _CLIP_LEN, _FAKE_DIM)
    vis_feat = torch.nn.functional.normalize(vis_feat, dim=-1)
    with torch.no_grad():
        out = model.forward_from_visual(vis_feat)

    assert out.frame_score.shape == (_BATCH, _CLIP_LEN)
    assert out.clip_score.shape == (_BATCH,)
    assert out.embedding.shape == (_BATCH, int(cfg.model.fusion.out_dim))


# ═══════════════════════════════════════════════════════════════════
# 5. Loss 闭环（AGENTS.md §6）
# ═══════════════════════════════════════════════════════════════════

def test_vlmvad_loss_backward(cfg):
    model, prompts = _build_test_model(cfg)
    model.train()

    video = torch.randn(_BATCH, _CLIP_LEN, _IMG_CHANNELS, _IMG_SIZE, _IMG_SIZE)
    out = model(video)

    # 构造标签：clip 0 后半段异常，clip 1 全正常（保证两类样本都在 batch 内）
    frame_label = torch.zeros(_BATCH, _CLIP_LEN, dtype=torch.int64)
    frame_label[0, _CLIP_LEN // 2:] = 1
    clip_label = torch.tensor([1, 0], dtype=torch.int64)

    polarity = build_prompt_polarity(prompts, cfg.prompt)
    criterion = VLMVADLoss(
        matcher=build_matcher(cfg.model),
        w_bce=float(cfg.train.loss.w_bce),
        w_contrastive=float(cfg.train.loss.w_contrastive),
        skip_bce_when_no_pos=bool(cfg.train.loss.skip_bce_when_no_pos),
    )

    losses = criterion(out, frame_label, clip_label, polarity)
    assert set(losses) == {"total", "bce", "contrastive"}
    for name, value in losses.items():
        assert torch.isfinite(value), f"loss[{name}] 非有限值: {value}"

    # 反向传播必须能产生梯度（训练闭环的关键）
    losses["total"].backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
