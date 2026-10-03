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
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from eval.metrics import clip_level_metrics, frame_level_metrics, video_level_metrics
from models import VLMModel
from models.matcher import Matcher
from train.losses import build_prompt_polarity
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


def evaluate_videos(
    model: VLMModel,
    matcher: Matcher,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 4,
    top_k: int = 5,
    out_dir: str | Path | None = None,
    save_plots: bool = True,
    cfg: Any | None = None,
) -> dict[str, Any]:
    """对整个数据集做视频级评估，返回指标 + 解释 + 可视化文件。

    步骤：
        1. 遍历所有 clip（数据集已切好，覆盖全部视频）
        2. 每个 clip 前向一次，得到逐帧分数 (B,T)
        3. 按 start_frame 坐标累加 → 每视频稠密逐帧分数
        4. 计算 frame / clip / video 三级指标
        5. 对最异常的 clip 生成解释文本 + 图

    Args:
        model: 训练好的 VLMModel。
        matcher: Matcher（用于 per-prompt 异常匹配）。
        dataset: 测试集（VideoDataset 或 FeatureDataset）。
        device: 计算设备。
        batch_size: 推理 batch 大小。
        top_k: 解释时取相似度最高的前 k 个异常 prompt。
        out_dir: 结果输出目录（None 则不落盘）。
        save_plots: 是否保存可视化图。
        cfg: 完整实验配置（可选）。传入后会计算 prompt 极性，使解释只从
            **异常极性** 的 prompt 中选 top-k，避免用正常 prompt 解释异常。

    Returns:
        dict：{"metrics", "per_video", "explanations", "frame_score", "frame_label"}。
        ``frame_score`` / ``frame_label`` 为 ``{video_id: (N,) ndarray}``，
        即整段视频的稠密逐帧分数与标签，供可视化 demo 直接复用。
    """
    model.eval()
    model.to(device)
    matcher.eval()
    matcher.to(device)

    prompts = model.prompt_processor.process()
    prompts_with_types = model.prompt_processor.process_with_types()
    polarity = (
        build_prompt_polarity(prompts, cfg.prompt) if cfg is not None else None
    )

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
                    worst_clips[vid] = {
                        "start": start, "clip_score": -1.0,
                        "anomaly_sims": None, "frame_scores": None,
                    }

                _accumulate_scores(accum[vid], counts[vid], start, fs)
                # 标签同样按坐标累加（与分数对齐，保证评估逻辑与标签来源解耦）
                _accumulate_scores(label_acc[vid], label_cnt[vid], start, lbl.astype(np.float64))

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
    dense_scores: dict[str, np.ndarray] = {}
    dense_labels: dict[str, np.ndarray] = {}
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

        dense_scores[vid] = scores
        dense_labels[vid] = lab
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

    # ── 可解释性：为每个视频最异常的 clip 生成解释 ─────────────────
    explanations = _build_explanations(
        model, worst_clips, prompts_with_types,
        top_k=top_k, out_dir=out_dir, save_plots=save_plots,
        polarity=polarity,
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
        "frame_score": dense_scores,
        "frame_label": dense_labels,
    }


def _build_explanations(
    model: VLMModel,
    worst_clips: dict[str, dict[str, Any]],
    prompts_with_types: list[tuple[str, str]],
    top_k: int,
    out_dir: str | Path | None,
    save_plots: bool,
    polarity: torch.Tensor | None = None,
) -> dict[str, dict[str, Any]]:
    """为每个视频生成 template-based 解释（回答"为什么异常"）。

    原理：per-prompt 相似度 anomaly_sims[k] 越高 → 该 clip 画面越符合
    prompt 描述的语义。解释**只从异常极性（+1）的 prompt 中选 top-k**，
    避免用 "a peaceful plaza scene" 这类正常 prompt 解释异常。

    Args:
        polarity: ``(K,)`` prompt 极性 {+1 异常, -1 正常, 0 中性}。
                  None 时退回"全部 prompt 排序"（旧行为）。

    Returns:
        {video_id: {"top_prompts": [...], "explanation": str, ...}}
    """
    pol = polarity.cpu().numpy() if polarity is not None else None
    # 允许参与解释的 prompt 下标：优先只用异常极性；没有则退回全部
    if pol is not None and (pol == 1).any():
        allowed = np.nonzero(pol == 1)[0]
    else:
        allowed = np.arange(len(prompts_with_types))

    result: dict[str, dict[str, Any]] = {}
    for vid, clip in worst_clips.items():
        sims = clip["anomaly_sims"]
        fs = clip["frame_scores"]
        if sims is None:
            continue

        # 仅在异常 prompt 内按相似度降序 → 最匹配的异常语义排最前
        order = allowed[np.argsort(sims[allowed])[::-1]]
        top = [{
            "prompt": prompts_with_types[i][1],
            "type": prompts_with_types[i][0],
            "polarity": int(pol[i]) if pol is not None else 0,
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
                    title=f"{vid} prompt similarity (abnormal-centered)",
                    save_path=plot_dir / f"{vid}_prompts.png",
                    polarity=pol,
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
