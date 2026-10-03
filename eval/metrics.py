# 这个文件定义 VAD 领域的标准评估指标——AUC 与 AP，帧级 / clip 级 / 视频级。
"""Evaluation metrics for video anomaly detection (VAD).

背景：VAD 指标为什么用 AUC / AP 而不是准确率（accuracy）？
------------------------------------------------------------------
异常检测是**极度类别不平衡**的二分类：一个视频里可能 95% 的帧都正常，
5% 的帧异常。如果只看 accuracy，模型"全都预测正常"就有 95% 的 accuracy，
指标完全失真。AUC / AP 只看**排序质量**（模型把异常排在正常前面有多准），
对不平衡鲁棒，是 VAD 的标准指标（Avenue / UCF-Crime / ShanghaiTech 都报 AUC）。

数学定义：
------------------------------------------------------------------
**ROC-AUC（Area Under the ROC Curve）**
    ROC 曲线把阈值 τ 从 +∞ 扫到 -∞，在每个阈值下计算：
        TPR(τ) = P(score > τ | y=1)  真正例率（异常被召回的比率）
        FPR(τ) = P(score > τ | y=0)  假正例率（正常被误报的比率）
    画出 TPR ~ FPR 曲线，AUC 是曲线下的面积。

    两个关键性质：
    1. AUC = P(随机抽一个异常样本分数 > 随机抽一个正常样本分数)，
       所以它度量的纯粹是**排序正确率**，与阈值无关。
    2. AUC = 0.5 表示随机猜测；AUC = 1.0 表示完美排序。
    计算：把 (score, label) 按 score 排序，用秩统计量可 O(n·log n) 算出。

**Average Precision（AP）**
    PR 曲线（Precision ~ Recall）下的面积。对类别不平衡更敏感：
        Precision = TP/(TP+FP)（预测为正中正确的比例）
        Recall    = TP/(TP+FN)（真正的正中被找回的比例）
    AP 按 recall 从高到低积分 precision，也可用均值公式：
        AP = Σ_n (R_n - R_{n-1}) · P_n
    经验上：AUC 偏高时，AP 能区分"好"和"很好"的排序（更严格）。

**帧级 vs clip 级 vs 视频级**
    - frame-level：每一帧一个 (score, label) 对 → 帧定位精度
    - clip-level：  每个 clip 一个 (score, label) 对 → 片段检测精度
    - video-level： 每个视频一个 (score, label) 对 → 视频整体判断
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _guard_sklearn(fn: Any, scores: np.ndarray, labels: np.ndarray) -> float:
    """安全地调用 sklearn 指标。

    sklearn 的 AUC 要求两类都至少有一个样本（否则无法定义 TPR/FPR）。
    异常检测验证集可能出现"全是正常帧"的极端情况，此时返回 0.0。
    """
    if len(scores) == 0 or len(labels) == 0:
        return 0.0
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.0  # 单类 → 指标未定义
    try:
        return float(fn(labels, scores))
    except ValueError:
        return 0.0


def frame_level_metrics(
    frame_scores: np.ndarray,
    frame_labels: np.ndarray,
) -> dict[str, float]:
    """帧级指标——衡量模型能否精确定位异常发生的帧。

    Args:
        frame_scores: ``(N,)`` 每帧的异常分数。
        frame_labels: ``(N,)`` 每帧的 0/1 标签。

    Returns:
        {"frame_auc": ..., "frame_ap": ...}。
    """
    frame_scores = np.asarray(frame_scores, dtype=np.float64).reshape(-1)
    frame_labels = np.asarray(frame_labels, dtype=np.int64).reshape(-1)
    return {
        "frame_auc": _guard_sklearn(roc_auc_score, frame_scores, frame_labels),
        "frame_ap": _guard_sklearn(average_precision_score, frame_scores, frame_labels),
    }


def clip_level_metrics(
    clip_scores: np.ndarray,
    clip_labels: np.ndarray,
) -> dict[str, float]:
    """clip 级指标——衡量模型能否把异常片段与正常片段分开。

    Args:
        clip_scores: ``(B,)`` 每个 clip 的异常分数（clip_score = amax(frame)）。
        clip_labels: ``(B,)`` 每个 clip 的 0/1 标签。

    Returns:
        {"clip_auc": ..., "clip_ap": ...}。
    """
    clip_scores = np.asarray(clip_scores, dtype=np.float64).reshape(-1)
    clip_labels = np.asarray(clip_labels, dtype=np.int64).reshape(-1)
    return {
        "clip_auc": _guard_sklearn(roc_auc_score, clip_scores, clip_labels),
        "clip_ap": _guard_sklearn(average_precision_score, clip_scores, clip_labels),
    }


def video_level_metrics(
    video_scores: dict[str, float],
    video_labels: dict[str, int],
) -> dict[str, float]:
    """视频级指标——每个视频一个异常分数。

    Args:
        video_scores: {video_id: score}。
        video_labels: {video_id: 0/1}。

    Returns:
        {"video_auc": ..., "video_ap": ...}。
    """
    # 按 video_id 对齐（两个 dict 可能顺序不同）
    common = [k for k in video_scores if k in video_labels]
    scores = np.asarray([video_scores[k] for k in common], dtype=np.float64)
    labels = np.asarray([video_labels[k] for k in common], dtype=np.int64)
    return {
        "video_auc": _guard_sklearn(roc_auc_score, scores, labels),
        "video_ap": _guard_sklearn(average_precision_score, scores, labels),
    }
