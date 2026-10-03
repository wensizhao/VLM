# 这个文件实现真正的 zero-shot 视频异常检测——冻结 CLIP 直接相似度匹配，无需训练。
"""Zero-shot video anomaly detection with a frozen CLIP backbone.

与 eval/inference.py 的区别：
    eval/inference.py       → 用训练好的 VLMModel（alignment/fusion/head）推理
    eval/zero_shot.py       → 完全跳过 VLMModel，直接用 CLIP 的视觉/文本相似度

为什么需要独立的 zero-shot 路径（研究动机）：
    VLM-VAD 的卖点之一是"无需训练即可检测异常"。训练前必须有一个可信的
    zero-shot baseline：给定异常 prompt 集合 A 与正常 prompt 集合 N，

        s_t = mean_k∈A cos(v_t, t_k) − mean_k∈N cos(v_t, t_k)

    即"这一帧更像异常描述还是正常描述"。分数越高越异常。这是一种
    open-vocabulary 的异常评分，不需要任何异常样本训练。

数据流：
    video (B,T,C,H,W) --CLIP encode_video--> vis_feat (B,T,D)
    prompts (K,)      --CLIP encode_text-----> txt_feat (K,L,D) --mean pool--> (K,D)
    sims = vis_feat @ txt_feat.T               (B,T,K)   余弦相似度
    anomaly = mean(sims[:,:,A]) - mean(sims[:,:,N])   (B,T)

评估：
    与 eval/inference.py 相同的滑窗逐帧聚合 + frame/clip/video 三级 AUC & AP，
    保证 zero-shot 与 fine-tuned 结果可直接对比（AGENTS.md §8）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from eval.metrics import clip_level_metrics, frame_level_metrics, video_level_metrics
from models.backbone import CLIPBackbone
from prompts.processor import PromptProcessor
from train.losses import build_prompt_polarity
from utils.io import save_json
from utils.logging import get_logger
from utils.visualization import plot_prompt_scores, plot_temporal_heatmap

log = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 特征工具
# ═══════════════════════════════════════════════════════════════════

def _pool_text(token_emb: torch.Tensor) -> torch.Tensor:
    """把 token 级文本特征池化成 prompt 级向量并 L2 归一化。

    仅作为退化路径使用（当 backbone 不提供 ``encode_text_pooled`` 时）。
    Zero-shot 默认走 CLIP 官方的 EOS 池化（见 ``ZeroShotDetector._prepare_text``）。

    Args:
        token_emb: ``(K, L, D)`` CLIP 文本编码器输出（token 级归一化）。

    Returns:
        ``(K, D)`` —— 平均池化 + L2 归一化，便于与归一化的视觉特征做余弦相似度。
    """
    return F.normalize(token_emb.mean(dim=1), dim=-1)


def _accumulate(
    accum: np.ndarray,
    counts: np.ndarray,
    start: int,
    values: np.ndarray,
) -> None:
    """把一段 clip 的值按帧坐标累加（重叠区域等权平均用）。"""
    end = min(start + len(values), len(accum))
    n = end - start
    accum[start:end] += values[:n]
    counts[start:end] += 1.0


# ═══════════════════════════════════════════════════════════════════
# ZeroShotDetector
# ═══════════════════════════════════════════════════════════════════

class ZeroShotDetector:
    """冻结 CLIP 的 zero-shot 异常评分器。

    视觉特征与文本特征都已 L2 归一化，因此点积即为余弦相似度。
    异常分数定义为"异常 prompt 组平均相似度 − 正常 prompt 组平均相似度"。
    """

    def __init__(
        self,
        backbone: CLIPBackbone,
        processor: PromptProcessor,
        polarity: torch.Tensor,
        device: torch.device,
        prompt_types: list[str] | None = None,
        text_pool: str = "mean",
    ) -> None:
        self.backbone = backbone.to(device).eval()
        self.processor = processor
        self.device = device
        self.prompts = processor.process(types=prompt_types)
        self.polarity = polarity.to(device)
        if text_pool not in {"mean", "eos"}:
            raise ValueError(f"text_pool must be 'mean' or 'eos', got {text_pool!r}")
        self.text_pool = text_pool

        ab = int((self.polarity == 1).sum())
        no = int((self.polarity == -1).sum())
        if ab == 0 or no == 0:
            raise ValueError(
                "Zero-shot scoring needs both abnormal (+1) and normal (-1) "
                f"prompts, got {ab} abnormal / {no} normal. "
                "Check prompt templates/expansions and polarity keywords."
            )
        self._abn_idx = (self.polarity == 1)
        self._nor_idx = (self.polarity == -1)
        self._text_emb: torch.Tensor | None = None

    def _prepare_text(self) -> torch.Tensor:
        """编码一次 prompt 文本并缓存（K 个 prompt 只需编码一次）。

        text_pool="mean"：token 平均池化——与训练时 ContrastivePolarityLoss
            使用的文本表征一致（losses.py: text_embedding.mean(dim=1)）。
        text_pool="eos" ：CLIP 官方 EOS 池化表征（encode_text_pooled）。
        """
        if self._text_emb is None:
            if self.text_pool == "eos" and hasattr(self.backbone, "encode_text_pooled"):
                self._text_emb = self.backbone.encode_text_pooled(self.prompts)
            else:
                token_emb = self.backbone.encode_text(self.prompts)   # (K, L, D)
                self._text_emb = _pool_text(token_emb)                # (K, D)
        return self._text_emb

    @torch.no_grad()
    def score(self, vis_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """对一批帧级视觉特征打分。

        Args:
            vis_feat: ``(B, T, D)`` L2 归一化的 CLIP 视觉特征。

        Returns:
            (anomaly, sims):
                anomaly — ``(B, T)`` 异常分数（异常相似度 − 正常相似度）。
                sims    — ``(B, T, K)`` 逐帧对每个 prompt 的余弦相似度。
        """
        text_emb = self._prepare_text()                      # (K, D)
        sims = torch.einsum("btd,kd->btk", vis_feat, text_emb)   # (B,T,K)
        anomaly = (
            sims[..., self._abn_idx].mean(dim=-1)
            - sims[..., self._nor_idx].mean(dim=-1)
        )                                                    # (B,T)
        return anomaly, sims


# ═══════════════════════════════════════════════════════════════════
# 主评估函数
# ═══════════════════════════════════════════════════════════════════

def evaluate_zero_shot(
    backbone: CLIPBackbone,
    processor: PromptProcessor,
    cfg: Any,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 4,
    top_k: int = 5,
    out_dir: str | Path | None = None,
    save_plots: bool = True,
    prompt_types: list[str] | None = None,
    text_pool: str | None = None,
) -> dict[str, Any]:
    """对测试集做 zero-shot 视频级评估，返回真实指标 + 解释。

    输出遵守 AGENTS.md §4 的契约：
        anomaly_score : dict[video_id → 视频级异常分数]
        frame_score   : dict[video_id → (T,) 稠密逐帧异常分数]
        embedding     : dict[video_id → (D,) 该视频最异常 clip 的视觉特征]
        explanation   : dict[video_id → {top_prompts, reason, ...}]
    """
    if text_pool is None:
        text_pool = str(cfg.eval.get("text_pool", "mean"))
    prompts = processor.process(types=prompt_types)
    polarity = build_prompt_polarity(prompts, cfg.prompt)
    detector = ZeroShotDetector(
        backbone, processor, polarity, device, prompt_types, text_pool=text_pool,
    )
    prompts = detector.prompts

    accum: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    label_acc: dict[str, np.ndarray] = {}
    label_cnt: dict[str, np.ndarray] = {}
    clip_scores: list[float] = []
    clip_labels: list[int] = []
    worst: dict[str, dict[str, Any]] = {}

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    for batch in tqdm(loader, desc="Zero-shot inference", unit="batch"):
        with torch.no_grad():
            if batch.get("vis_feat", None) is not None:
                vis_feat = batch["vis_feat"].to(device)             # (B,T,D)
            else:
                vis_feat = backbone.encode_video(batch["video"].to(device))
            anomaly, sims = detector.score(vis_feat)                # (B,T), (B,T,K)
        anomaly = anomaly.cpu().numpy()
        sims = sims.cpu().numpy()

        for b in range(anomaly.shape[0]):
            vid = batch["video_id"][b]
            start = int(batch["start_frame"][b])
            fs = anomaly[b]
            lbl = batch["frame_label"][b].cpu().numpy().astype(np.float64)

            if vid not in accum:
                rec = next(r for r in dataset.video_records if r.video_id == vid)
                n = rec.num_frames
                accum[vid] = np.zeros(n, dtype=np.float64)
                counts[vid] = np.zeros(n, dtype=np.float64)
                label_acc[vid] = np.zeros(n, dtype=np.float64)
                label_cnt[vid] = np.zeros(n, dtype=np.float64)
                worst[vid] = {"start": start, "clip_score": -np.inf,
                              "sims": None, "frame_scores": None,
                              "embedding": None}

            _accumulate(accum[vid], counts[vid], start, fs)
            _accumulate(label_acc[vid], label_cnt[vid], start, lbl)

            clip_score = float(fs.max())
            clip_scores.append(clip_score)
            clip_labels.append(int(batch["clip_label"][b]))

            if clip_score > worst[vid]["clip_score"]:
                worst[vid].update(
                    start=start, clip_score=clip_score,
                    sims=sims[b].mean(axis=0),            # (K,) 该 clip 平均相似度
                    frame_scores=fs,
                    embedding=vis_feat[b].mean(dim=0).detach().cpu().numpy(),
                )

    per_video: dict[str, Any] = {}
    frame_score: dict[str, np.ndarray] = {}
    anomaly_score: dict[str, float] = {}
    embedding: dict[str, np.ndarray] = {}
    video_labels: dict[str, int] = {}
    global_scores: list[np.ndarray] = []
    global_labels: list[np.ndarray] = []

    for vid in accum:
        scores = accum[vid] / np.maximum(counts[vid], 1.0)
        lab = (label_acc[vid] / np.maximum(label_cnt[vid], 1.0) > 0.5).astype(np.int64)

        per_video[vid] = frame_level_metrics(scores, lab)
        per_video[vid]["num_frames"] = int(len(lab))
        per_video[vid]["num_anomaly_frames"] = int(lab.sum())

        frame_score[vid] = scores
        anomaly_score[vid] = float(worst[vid]["clip_score"])
        embedding[vid] = worst[vid]["embedding"]
        video_labels[vid] = 1 if lab.sum() > 0 else 0
        global_scores.append(scores)
        global_labels.append(lab)

    metrics: dict[str, float] = {}
    if global_scores:
        metrics.update(frame_level_metrics(
            np.concatenate(global_scores), np.concatenate(global_labels)))
    metrics.update(clip_level_metrics(
        np.asarray(clip_scores), np.asarray(clip_labels)))
    metrics.update(video_level_metrics(anomaly_score, video_labels))

    log.info(
        "Zero-shot | prompts(K=%d, +1=%d, -1=%d) | frame_auc=%.4f frame_ap=%.4f "
        "video_auc=%.4f video_ap=%.4f",
        len(prompts), int((polarity == 1).sum()), int((polarity == -1).sum()),
        metrics.get("frame_auc", 0.0), metrics.get("frame_ap", 0.0),
        metrics.get("video_auc", 0.0), metrics.get("video_ap", 0.0),
    )

    explanations = _build_explanations(
        worst, prompts, detector.polarity, top_k, out_dir, save_plots,
        frame_score=frame_score,
    )

    results: dict[str, Any] = {
        "metrics": metrics,
        "per_video": per_video,
        "anomaly_score": anomaly_score,
        "frame_score": frame_score,
        "embedding": embedding,
        "explanation": explanations,
        "num_prompts": len(prompts),
        "prompts": prompts,
        "polarity": detector.polarity.cpu().numpy().tolist(),
    }

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_json(
            {"metrics": metrics, "per_video": per_video,
             "anomaly_score": anomaly_score},
            out / "zero_shot_metrics.json",
        )
        save_json(explanations, out / "zero_shot_explanations.json")

    return results


def _build_explanations(
    worst: dict[str, dict[str, Any]],
    prompts: list[str],
    polarity: torch.Tensor,
    top_k: int,
    out_dir: str | Path | None,
    save_plots: bool,
    frame_score: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    """生成 template-based 解释——回答"为什么这个视频是异常的"。"""
    pol = polarity.cpu().numpy()
    # 解释只从异常极性 prompt 中选，避免用正常 prompt 解释异常
    allowed = np.nonzero(pol == 1)[0] if (pol == 1).any() else np.arange(len(prompts))
    result: dict[str, dict[str, Any]] = {}

    for vid, clip in worst.items():
        sims = clip["sims"]
        if sims is None:
            continue

        order = allowed[np.argsort(sims[allowed])[::-1]]
        top = [{
            "prompt": prompts[i],
            "polarity": int(pol[i]),
            "similarity": round(float(sims[i]), 4),
        } for i in order[:top_k]]

        fs = clip["frame_scores"]
        reason = " / ".join(
            f"{t['prompt']!r}(sim={t['similarity']:.2f})" for t in top[:3]
        )
        worst_frame = int(np.argmax(fs))
        explanation = (
            f"该视频最异常的片段起始于第 {clip['start']} 帧，"
            f"与以下 prompt 语义最接近：{reason}。"
            f"其中第 {clip['start'] + worst_frame} 帧异常分数最高"
            f"（{float(fs[worst_frame]):.3f}）。"
        )

        result[vid] = {
            "start_frame": int(clip["start"]),
            "clip_score": round(float(clip["clip_score"]), 4),
            "top_prompts": top,
            "top_abnormal_frames": np.argsort(fs)[::-1][:5].tolist(),
            "explanation": explanation,
        }

        if save_plots and out_dir is not None:
            plot_dir = Path(out_dir) / "zero_shot_plots"
            try:
                plot_temporal_heatmap(
                    frame_score[vid],
                    title=f"{vid} zero-shot frame anomaly",
                    save_path=plot_dir / f"{vid}_heatmap.png",
                )
                plot_prompt_scores(
                    sims, prompts,
                    title=f"{vid} zero-shot prompt similarity (abnormal-centered)",
                    save_path=plot_dir / f"{vid}_prompts.png",
                    polarity=pol,
                )
            except Exception as e:  # noqa: BLE001
                log.warning("Plot failed for %s: %s", vid, e)

    return result
