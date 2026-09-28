# 这个文件实现视频级推理——滑窗遍历整个视频 → 逐帧异常分数 → 指标 + 可解释性输出。
"""Video-level inference: sliding-window frame scoring, metrics, and explanations.

推理时的一个关键问题：训练用 16 帧 clip，但视频有上千帧。
我们让 clip 以非重叠步长滑过整个视频（利用数据集已经切好的 clip_records），
然后把每个 clip 的逐帧分数按帧坐标累加/求平均，得到**稠密的逐帧异常分数**，
再计算真正的视频级 frame AUC/AP。

分数聚合策略：frame_score(start+j) 直接对应该视频的真实帧 start+j
（因为 clip_stride=1 时 clip 内是连续帧）。对重叠区域（include_last_clip
补的最后一个 clip）做**等权平均**，消除"同一帧被算两次"的偏差。

可解释性（AGENTS.md §5）：
    - abnormal frames ranking：frame_score 降序排列 → 最异常的帧
    - temporal anomaly heatmap：逐帧分数曲线 + GT 色带
    - template-based explanation：Matcher(embedding, prompt_embs) 相似度
      降序 → 找出"最符合的异常 prompt"→ 回答"为什么这个视频是异常的"

零样本 CLIP 保底（zero-shot baseline）：
    传入 prompt 极性 ``polarity`` 后，同一次前向会额外计算一条**完全不经训练**
    的基线分数：raw CLIP 特征 + 固定余弦相似度 + 极性做差（见
    ``_zero_shot_frame_scores`` 的数学说明）。指标以 ``zs_`` 前缀与训练指标
    并列输出——训练后的模型应打平或超过保底，否则说明训练本身出了问题。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from eval.metrics import clip_level_metrics, frame_level_metrics, video_level_metrics
from models import VLMModel
from models.matcher import Matcher
from utils.io import save_json
from utils.logging import get_logger
from utils.visualization import (
    plot_prompt_scores,
    plot_temporal_heatmap,
)

log = get_logger(__name__)


def _accumulate_scores(
    accum: np.ndarray,
    counts: np.ndarray,
    start: int,
    scores: np.ndarray,
) -> None:
    """把一段 clip 的帧分数按坐标累加。

    Args:
        accum: ``(N,)`` 累加和数组（按视频总帧数初始化）。
        counts: ``(N,)`` 每帧被覆盖次数。
        start: clip 起始帧坐标。
        scores: ``(T,)`` 该 clip 的逐帧分数。
    """
    end = min(start + len(scores), len(accum))
    n = end - start
    accum[start:end] += scores[:n]
    counts[start:end] += 1.0


def _pool_text_tokens(text_feature: Tensor, token_mask: Tensor) -> Tensor:
    """对 token 级文本特征做 masked mean pooling（排除 padding token）。

    为什么不用普通 mean：encode_text 输出固定长度 L（如 77），序列不足处
    全是 padding，直接平均会把有效语义稀释。mask 由 token id > 0 得出
    （open_clip 用 0 做填充，见 CLIPBackbone.tokenize 的文档）。

    Args:
        text_feature: ``(K, L, D)`` — 原始 CLIP token 级特征。
        token_mask: ``(K, L)`` — bool，True 为真实 token。

    Returns:
        ``(K, D)`` — 每个 prompt 一个向量。
    """
    mask = token_mask.unsqueeze(-1).to(text_feature.dtype)   # (K, L, 1)
    pooled = (text_feature * mask).sum(dim=1)                # (K, D)
    denom = mask.sum(dim=1).clamp(min=1.0)                   # (K, 1)
    return pooled / denom


def _zero_shot_frame_scores(
    visual_feature: Tensor,
    txt_pooled: Tensor,
    abnormal_mask: Tensor,
    normal_mask: Tensor,
) -> Tensor:
    """零样本 CLIP 保底分数——完全不经过任何可训练参数。

    数学定义（v_t 是第 t 帧的原始 CLIP 向量，A/N 是异常/正常 prompt 集合）：
        zs_t = (1/|A|) Σ_{k∈A} cos(v_t, p_k) − (1/|N|) Σ_{k∈N} cos(v_t, p_k)

    为什么做差而不只取异常均值：相似度整体可能随场景明暗、画面内容整体
    漂移，异常组与正常组同时被抬高或压低（共模偏移）。组间做差消掉共模
    部分，留下的才是"相对更接近异常语义"的判别信号。

    为什么不走 Matcher：Matcher 的 logit_scale（以及 learnable 策略的 W）
    本身是训练对象，训练失败时它们同样是坏的。保底必须独立于一切被训练
    的参数，因此这里直接写 F.normalize + 点积（即固定余弦相似度）。

    Args:
        visual_feature: ``(B, T, D)`` — 原始 CLIP 帧特征（未过 Alignment）。
        txt_pooled: ``(K, D)`` — masked mean 池化后的 prompt 向量。
        abnormal_mask: ``(K,)`` bool — 极性为 +1 的 prompt。
        normal_mask: ``(K,)`` bool — 极性为 -1 的 prompt。

    Returns:
        ``(B, T)`` — 逐帧零样本异常分数（值域约 [-2, 2]，AUC 只看排序）。
    """
    B, T, D = visual_feature.shape
    v = F.normalize(visual_feature.reshape(B * T, D), dim=-1)   # (B*T, D)
    p = F.normalize(txt_pooled, dim=-1)                          # (K, D)
    sim = v @ p.T                                                # (B*T, K) 余弦
    abn = sim[:, abnormal_mask].mean(dim=-1)                     # (B*T,)
    nor = sim[:, normal_mask].mean(dim=-1)                       # (B*T,)
    return (abn - nor).reshape(B, T)


def evaluate_videos(
    model: VLMModel,
    matcher: Matcher,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 4,
    top_k: int = 5,
    out_dir: str | Path | None = None,
    save_plots: bool = True,
    polarity: Tensor | None = None,
) -> dict[str, Any]:
    """对整个数据集做视频级评估，返回指标 + 解释 + 可视化文件。

    步骤：
        1. 遍历所有 clip（数据集已切好，覆盖全部视频）
        2. 每个 clip 前向一次，得到逐帧分数 (B,T)
        3. 按 start_frame 坐标累加 → 每视频稠密逐帧分数
        4. 计算 frame / clip / video 三级指标
        5. 对最异常的 clip 生成解释文本 + 图
        6. polarity 提供时，同步计算零样本 CLIP 保底指标（zs_* 前缀）

    Args:
        model: 训练好的 VLMModel。
        matcher: Matcher（用于 per-prompt 异常匹配）。
        dataset: 测试集（VideoDataset 或 FeatureDataset）。
        device: 计算设备。
        batch_size: 推理 batch 大小。
        top_k: 解释时取相似度最高的前 k 个异常 prompt。
        out_dir: 结果输出目录（None 则不落盘）。
        save_plots: 是否保存可视化图。
        polarity: ``(K,)`` prompt 极性 {+1 异常, -1 正常, 0 中性}。提供时
            额外计算零样本 CLIP 基线（不经过任何可训练参数），作为训练效果
            的保底参照；None 则跳过。

    Returns:
        dict：{"metrics": ..., "per_video": ..., "explanations": ...}。
    """
    model.eval()
    model.to(device)
    matcher.eval()
    matcher.to(device)

    prompts = model.prompt_processor.process()
    prompts_with_types = model.prompt_processor.process_with_types()

    # ── 零样本 CLIP 保底（zero-shot baseline）─────────────────────
    # 设计：完全绕开 alignment / fusion / temporal / head / matcher 这些
    # 可训练（或随机初始化）的模块，只用冻结 CLIP 的原始特征 + 固定余弦
    # 相似度 + 极性做差。训练失败时这条路径仍然有效，这就是"保底"的含义。
    zs_enabled: bool = False
    zs_txt_pooled: Tensor | None = None
    accum_zs: dict[str, np.ndarray] = {}
    zs_clip_scores: list[float] = []
    # 空张量占位：zs_enabled=False 时不会被使用，避免 Optional 判空散落各处
    zs_abn_mask = torch.empty(0, dtype=torch.bool, device=device)
    zs_nor_mask = torch.empty(0, dtype=torch.bool, device=device)
    zs_token_mask = torch.empty(0, dtype=torch.bool, device=device)
    if polarity is not None:
        pol = polarity.to(device)
        if pol.shape[0] != len(prompts):
            raise ValueError(
                f"polarity 长度 ({pol.shape[0]}) 与 prompt 数 ({len(prompts)}) "
                f"不一致，请检查 polarity 是否来自同一份 prompt 配置。"
            )
        abn, nor = pol == 1, pol == -1
        if not (bool(abn.any()) and bool(nor.any())):
            log.warning(
                "Zero-shot baseline skipped: normal 与 abnormal prompt 组需要都非空。"
            )
        else:
            zs_abn_mask, zs_nor_mask = abn, nor
            # token id > 0 表示真实 token（open_clip 用 0 做 padding）
            zs_token_mask = model.backbone.tokenize(prompts) > 0   # (K, L)
            zs_enabled = True

    # ── 每视频的累计状态 ─────────────────────────────────────────
    accum: dict[str, np.ndarray] = {}    # video_id → 分数累加
    counts: dict[str, np.ndarray] = {}   # video_id → 覆盖次数
    label_acc: dict[str, np.ndarray] = {}    # video_id → 标签累加
    label_cnt: dict[str, np.ndarray] = {}    # video_id → 标签覆盖次数
    labels: dict[str, np.ndarray] = {}       # video_id → 最终逐帧标签
    # clip 级收集
    clip_scores: list[float] = []
    clip_labels: list[int] = []
    # 每个视频选"最异常 clip"存下来做解释
    worst_clips: dict[str, dict[str, Any]] = {}

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    )
    n_pos_frames = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Video-level inference", unit="batch"):
            # 前向（自动选择像素/特征路径）
            if "vis_feat" in batch and batch["vis_feat"] is not None:
                output = model.forward_from_visual(batch["vis_feat"].to(device))
            else:
                output = model.forward(batch["video"].to(device))

            frame_scores = output.frame_score.cpu().numpy()       # (B, T)
            clip_score = output.clip_score.cpu().numpy()          # (B,)
            # 每个样本的逐 prompt 相似度（用于解释）
            txt_pooled = output.alignment.text_embedding.mean(dim=1)
            match = matcher(output.embedding, txt_pooled)
            anomaly_sims = match.similarity.cpu().numpy()         # (B, K)

            # 零样本保底：raw CLIP 特征（跳过全部可训练模块）
            # 文本特征每个 batch 都一样，只在首个 batch 池化一次并复用
            zs_fs: np.ndarray | None
            if zs_enabled:
                if zs_txt_pooled is None:
                    zs_txt_pooled = _pool_text_tokens(
                        output.alignment.text_feature, zs_token_mask,
                    )
                zs_fs = _zero_shot_frame_scores(
                    output.alignment.visual_feature, zs_txt_pooled,
                    zs_abn_mask, zs_nor_mask,
                ).cpu().numpy()                                     # (B, T)
            else:
                zs_fs = None

            for b in range(frame_scores.shape[0]):
                vid = batch["video_id"][b]
                start = int(batch["start_frame"][b])
                fs = frame_scores[b]
                lbl = batch["frame_label"][b].cpu().numpy()   # (T,)

                # 惰性初始化该视频的数组
                if vid not in accum:
                    # 从 video_records 拿真实帧数（两个 Dataset 都有该属性）
                    rec = next(r for r in dataset.video_records if r.video_id == vid)
                    n = rec.num_frames
                    accum[vid] = np.zeros(n, dtype=np.float64)
                    counts[vid] = np.zeros(n, dtype=np.float64)
                    label_acc[vid] = np.zeros(n, dtype=np.float64)
                    label_cnt[vid] = np.zeros(n, dtype=np.float64)
                    if zs_enabled:
                        accum_zs[vid] = np.zeros(n, dtype=np.float64)
                    worst_clips[vid] = {
                        "start": start, "clip_score": -1.0,
                        "anomaly_sims": None, "frame_scores": None,
                    }

                _accumulate_scores(accum[vid], counts[vid], start, fs)
                # 标签同样按坐标累加（与分数对齐，保证评估逻辑与标签来源解耦）
                _accumulate_scores(label_acc[vid], label_cnt[vid], start, lbl.astype(np.float64))

                # 零样本保底分数同样按坐标累加（共用 counts，覆盖次数一致）
                if zs_fs is not None:
                    _accumulate_scores(accum_zs[vid], counts[vid], start, zs_fs[b])
                    zs_clip_scores.append(float(zs_fs[b].max()))

                # clip 级收集
                clip_scores.append(float(clip_score[b]))
                clip_labels.append(int(batch["clip_label"][b]))

                # 记录该视频目前最异常的 clip（clip_score 最大）
                if clip_score[b] > worst_clips[vid]["clip_score"]:
                    worst_clips[vid].update(
                        start=start, clip_score=float(clip_score[b]),
                        anomaly_sims=anomaly_sims[b], frame_scores=fs,
                    )

    # ── 每视频逐帧分数 = 累加/覆盖次数（等权平均）─────────────────
    per_video: dict[str, Any] = {}
    global_frame_scores: list[np.ndarray] = []
    global_frame_labels: list[np.ndarray] = []
    video_scores: dict[str, float] = {}
    video_labels: dict[str, int] = {}

    for vid in accum:
        scores = accum[vid] / np.maximum(counts[vid], 1.0)
        # 帧标签 = 覆盖到的 clip 标签多数票（>0.5 → 异常）
        lab = (label_acc[vid] / np.maximum(label_cnt[vid], 1.0) > 0.5).astype(np.int64)
        # 该视频 frame AUC（>= 需要一个正常帧+异常帧）
        per_video[vid] = frame_level_metrics(scores, lab)
        per_video[vid]["num_frames"] = int(len(lab))
        per_video[vid]["num_anomaly_frames"] = int(lab.sum())

        global_frame_scores.append(scores)
        global_frame_labels.append(lab)
        # 视频级：用 clip 分数上确界近似视频分数（任一异常片段→视频异常）
        video_scores[vid] = float(worst_clips[vid]["clip_score"])
        video_labels[vid] = 1 if lab.sum() > 0 else 0

    gfs = np.concatenate(global_frame_scores)
    gfl = np.concatenate(global_frame_labels)

    metrics = {}
    metrics.update(frame_level_metrics(gfs, gfl))
    metrics.update(clip_level_metrics(
        np.asarray(clip_scores), np.asarray(clip_labels),
    ))
    metrics.update(video_level_metrics(video_scores, video_labels))

    log.info(
        "Eval | frame_auc=%.4f frame_ap=%.4f clip_auc=%.4f clip_ap=%.4f "
        "video_auc=%.4f",
        metrics.get("frame_auc", 0.0), metrics.get("frame_ap", 0.0),
        metrics.get("clip_auc", 0.0), metrics.get("clip_ap", 0.0),
        metrics.get("video_auc", 0.0),
    )

    # ── 零样本保底指标（zs_* 前缀，与训练指标并列）──────────────────
    if zs_enabled:
        zs_gfs = np.concatenate([
            accum_zs[vid] / np.maximum(counts[vid], 1.0) for vid in accum
        ])
        zs_frame = frame_level_metrics(zs_gfs, gfl)
        zs_clip = clip_level_metrics(
            np.asarray(zs_clip_scores), np.asarray(clip_labels),
        )
        metrics.update({f"zs_{k}": v for k, v in {**zs_frame, **zs_clip}.items()})
        log.info(
            "Zero-shot CLIP 保底 | zs_frame_auc=%.4f zs_clip_auc=%.4f | "
            "trained: frame_auc=%.4f clip_auc=%.4f —— 训练后应≥保底，"
            "若明显低于保底说明训练出了问题",
            zs_frame.get("frame_auc", 0.0), zs_clip.get("clip_auc", 0.0),
            metrics.get("frame_auc", 0.0), metrics.get("clip_auc", 0.0),
        )

    # ── 可解释性：为每个视频最异常的 clip 生成解释 ─────────────────
    explanations = _build_explanations(
        model, worst_clips, prompts_with_types,
        top_k=top_k, out_dir=out_dir, save_plots=save_plots,
    )

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_json({"metrics": metrics, "per_video": per_video}, out / "metrics.json")
        save_json(explanations, out / "explanations.json")

    return {
        "metrics": metrics,
        "per_video": per_video,
        "explanations": explanations,
    }


def _build_explanations(
    model: VLMModel,
    worst_clips: dict[str, dict[str, Any]],
    prompts_with_types: list[tuple[str, str]],
    top_k: int,
    out_dir: str | Path | None,
    save_plots: bool,
) -> dict[str, dict[str, Any]]:
    """为每个视频生成 template-based 解释（回答"为什么异常"）。

    原理：per-prompt 相似度 anomaly_sims[k] 越高 → 该 clip 画面越符合
    prompt 描述的语义。取相似度最高的**异常类** prompt 作为"异常原因"。

    Returns:
        {video_id: {"top_abnormal_prompts": [...], "explanation": str, ...}}
    """
    result: dict[str, dict[str, Any]] = {}
    for vid, clip in worst_clips.items():
        sims = clip["anomaly_sims"]
        fs = clip["frame_scores"]
        if sims is None:
            continue

        # 相似度降序 → 最匹配的 prompt 排最前
        order = np.argsort(sims)[::-1]
        top = [{
            "prompt": prompts_with_types[i][1],
            "type": prompts_with_types[i][0],
            "similarity": round(float(sims[i]), 4),
        } for i in order[:top_k]]

        # 异常的帧排名（最异常的帧在前）
        frame_rank = np.argsort(fs)[::-1].tolist()

        explanation = _template_explanation(top, clip["start"], fs)

        entry = {
            "start_frame": int(clip["start"]),
            "clip_score": round(float(clip["clip_score"]), 4),
            "top_prompts": top,
            "top_abnormal_frames": frame_rank[:5],
            "explanation": explanation,
        }
        result[vid] = entry

        # 可选：保存可视化图
        if save_plots and out_dir is not None:
            plot_dir = Path(out_dir) / "plots"
            try:
                plot_temporal_heatmap(
                    fs, title=f"{vid} clip@{clip['start']} frame anomaly",
                    save_path=plot_dir / f"{vid}_heatmap.png",
                )
                plot_prompt_scores(
                    sims, [p for _, p in prompts_with_types],
                    title=f"{vid} prompt similarity",
                    save_path=plot_dir / f"{vid}_prompts.png",
                )
            except Exception as e:  # noqa: BLE001
                log.warning("Plot failed for %s: %s", vid, e)

    return result


def _template_explanation(
    top: list[dict[str, Any]],
    start_frame: int,
    frame_scores: np.ndarray,
) -> str:
    """模板化解释——把 top-k prompt 与帧排名组织成一句话。

    回答："为什么这个视频是异常的？"
    """
    reason = " / ".join(
        f"{t['prompt']!r}(sim={t['similarity']:.2f})" for t in top[:3]
    )
    worst_frame = int(np.argmax(frame_scores))
    return (
        f"该视频段（起始帧 {start_frame}）被判定为异常，"
        f"因为画面语义与以下描述最接近：{reason}。"
        f"其中最异常的是第 {start_frame + worst_frame} 帧（分数 {float(frame_scores[worst_frame]):.2f}）。"
    )
