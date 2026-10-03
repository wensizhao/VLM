# 这个文件负责可视化——把模型输出画成图，让异常"看得见"。
"""Visualization utilities — temporal heatmaps, attention maps, per-prompt scores.

支持三种可视化（对应 AGENTS.md §5 的可解释性要求）：
- ``plot_temporal_heatmap`` —— 帧级异常分数随时间的曲线 + 高亮热区（回答"哪几帧异常"）
- ``plot_frame_attention``  —— CrossAttnFusion 的 attention (T, K*L) → 每帧对每个 prompt 的关注度
- ``plot_prompt_scores``    —— 一个 clip 与 K 个 prompt 的相似度条形图（回答"最匹配哪个语义"）

设计约定：所有函数 ``save_path=None`` 时直接显示；否则保存 PNG。
在无显示器环境（服务器/Docker）必须先 ``matplotlib.use("Agg")``，否则 import 会崩。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无窗口后端，保证在服务器/容器里也能出图

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

# 中文字体不是必须的；避免缺字体导致的 warn 噪音
plt.rcParams["font.size"] = 11


def _save_or_show(fig, save_path: str | Path | None) -> None:
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_temporal_heatmap(
    frame_scores: Tensor | np.ndarray,
    frame_labels: Tensor | np.ndarray | None = None,
    title: str = "Frame-level anomaly scores",
    save_path: str | Path | None = None,
    threshold: float | None = None,
) -> None:
    """绘制帧级异常分数曲线（可叠加上 GT 标签作为背景色带）。

    用途：回答"这个视频里哪些帧是异常的？"——异常帧分数显著抬升。
    传入整段视频的 ``(T,)`` 分数即可得到完整时间轴（不再是单个 clip）。

    Args:
        frame_scores: ``(T,)`` 帧级异常分数 [0, 1]。
        frame_labels: ``(T,)`` 可选——GT 标签，用于把异常区间涂成红色背景。
        title: 图标题。
        save_path: 保存路径；None 则显示。
        threshold: 可选——判定异常的水平线（虚线）。
    """
    scores = (
        frame_scores.cpu().numpy() if isinstance(frame_scores, Tensor)
        else np.asarray(frame_scores)
    )

    fig, ax = plt.subplots(figsize=(12, 3.5))

    # GT 异常区间背景：红色半透明色带，一眼看出模型有没有在 GT 区间抬分
    if frame_labels is not None:
        labels = (
            frame_labels.cpu().numpy() if isinstance(frame_labels, Tensor)
            else np.asarray(frame_labels)
        )
        ax.imshow(
            labels[None, :], aspect="auto", cmap="Reds", alpha=0.25,
            extent=(0, len(scores) - 1, 0, 1), interpolation="nearest",
        )

    ax.plot(np.arange(len(scores)), scores, color="tab:blue", linewidth=1.2)
    if threshold is not None:
        ax.axhline(
            float(threshold), color="tab:red", linestyle="--", linewidth=1.0,
            label=f"threshold={threshold:.2f}",
        )
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Anomaly score")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)

    _save_or_show(fig, save_path)


def plot_prompt_scores(
    similarity: Tensor | np.ndarray,
    prompts: list[str],
    title: str = "Clip-to-prompt similarity",
    save_path: str | Path | None = None,
    polarity: np.ndarray | None = None,
    center: bool = True,
) -> list[int]:
    """绘制一个 clip 与 K 个 prompt 的相似度横向条形图。

    用途：回答"这个视频为什么异常？"——相似度最高的 prompt 就是这个 clip
    最符合的语义描述（异常类型定位）。

    为什么需要 center：CLIP 余弦相似度普遍被压缩在 0 附近，直接画条
    所有条看起来一样长。默认画 ``sim - mean(sim)``（相对平均的偏差），
    差异立刻可见；正值表示"比平均更像该 prompt"。

    Args:
        similarity: ``(K,)`` 该 clip 与每个 prompt 的相似度。
        prompts: K 个 prompt 文本。
        title: 图标题。
        save_path: 保存路径。
        polarity: 可选 ``(K,)`` —— {+1 异常, -1 正常, 0 中性}，
            用于把异常 prompt 标红、正常标绿，一眼看出极性。
        center: 是否按均值中心化（默认 True）。

    Returns:
        按相似度从高到低排序的 prompt 下标列表（供解释文本复用）。
    """
    sim = (
        similarity.cpu().numpy() if isinstance(similarity, Tensor)
        else np.asarray(similarity)
    )
    order = np.argsort(sim)[::-1]  # 降序：最匹配的在前
    values = sim - sim.mean() if center else sim

    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.35 * len(prompts))))
    if polarity is not None:
        # 极性着色：异常=红，正常=绿，中性=灰
        color_map = {1: "tab:red", -1: "tab:green", 0: "tab:gray"}
        colors = [color_map.get(int(polarity[i]), "tab:gray") for i in order]
    else:
        colors = plt.cm.viridis(
            (sim[order] - sim[order].min()) /
            (sim[order].max() - sim[order].min() + 1e-8)
        )
    ax.barh(range(len(order)), values[order], color=colors)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([prompts[i] for i in order], fontsize=8)
    ax.invert_yaxis()  # 最高的显示在最上面
    ax.set_xlabel("Similarity (centered)" if center else "Similarity")
    ax.set_title(title)

    _save_or_show(fig, save_path)
    return [int(i) for i in order]


def plot_frame_attention(
    attention: Tensor | np.ndarray,
    prompts: list[str],
    max_tokens_per_prompt: int = 77,
    title: str = "Frame-to-prompt cross-attention",
    save_path: str | Path | None = None,
) -> None:
    """把 CrossAttnFusion 的 attention ``(T, K*L)`` 压缩成帧×prompt 矩阵。

    数学背景：
        CrossAttnFusion 里每帧 attend 到 Text Memory Bank（K 个 prompt 的
        token 序列拼接，共 K*L 个 token），得到权重 ``(T, K*L)``。
        这里把属于同一个 prompt 的 L 个 token 的注意力求和，得到
        "第 t 帧对第 k 个 prompt 的总关注度" —— 回答"模型在看哪个语义"。

    Args:
        attention: ``(T, K*L)`` 交叉注意力权重（softmax 归一化过）。
        prompts: K 个 prompt 文本。
        max_tokens_per_prompt: 每个 prompt 的 token 数 L（CLIP 默认 77）。
        title: 图标题。
        save_path: 保存路径。
    """
    attn = (
        attention.cpu().numpy() if isinstance(attention, Tensor)
        else np.asarray(attention)
    )
    T, K = attn.shape[0], len(prompts)
    # (T, K*L) -> (T, K, L) -> sum over L
    per_prompt = attn.reshape(T, K, -1).sum(axis=-1)  # (T, K)
    per_prompt = per_prompt / (per_prompt.sum(axis=-1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(10, max(3, 0.4 * K)))
    im = ax.imshow(per_prompt.T, aspect="auto", cmap="magma")
    ax.set_yticks(range(K))
    ax.set_yticklabels(prompts, fontsize=8)
    ax.set_xlabel("Frame index")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="normalized attention")

    _save_or_show(fig, save_path)
