#!/usr/bin/env python3
# 这个脚本把训练好的 VLM-VAD 检测效果渲染成可直接观看的 MP4/GIF——让异常"看得见"。
"""Render an intuitive anomaly-detection demo (MP4 / GIF / montage) from a checkpoint.

用法::

    python scripts/visualize_demo.py --config configs/experiment.yaml \\
        --ckpt checkpoints/exp01/best.pt --videos 01 06 18 --out results/demo

产物（每个视频）：
    {out}/{vid}_demo.mp4        真实画面 + 异常高亮 + 分数条 + 完整时间轴
    {out}/{vid}_demo.gif        最异常片段短动图（可直接贴 PPT）
    {out}/{vid}_top_frames.png  最异常 K 帧缩略图网格
    {out}/timeline_{vid}.png    整段视频逐帧分数 + GT 色带 + 阈值线
另输出 {out}/index.md —— 所有视频一览（含中文解释与文件链接）。

设计说明：
    - 逐帧分数直接复用 ``eval.inference.evaluate_videos``（不重复实现推理），
      它返回的 ``frame_score`` / ``frame_label`` 是整段视频的稠密 ``(N,)`` 数组。
    - 视频上文字用英文（容器内无中文字体），中文解释写入 index.md。
    - MP4 通过 ffmpeg(libx264/yuv420p) 编码，保证浏览器/VSCode 可直接播放。
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from datasets.builders import build_split_dataloader  # noqa: E402
from datasets.video_dataset import (  # noqa: E402
    _normalize_split,
    _resolve_avenue_root,
)
from eval.inference import evaluate_videos  # noqa: E402
from models.factory import build_matcher, build_model  # noqa: E402
from utils.config import load_experiment_config, save_config_snapshot  # noqa: E402
from utils.io import ensure_dirs, load_checkpoint  # noqa: E402
from utils.logging import get_logger  # noqa: E402
from utils.reproducibility import set_seed  # noqa: E402
from utils.visualization import plot_temporal_heatmap  # noqa: E402

log = get_logger(__name__)

# ── 画布布局（1280x720）────────────────────────────────────────────
CANVAS_W, CANVAS_H = 1280, 720
FRAME_W, FRAME_H = 960, 540            # 视频画面区域（640x360 × 1.5）
PANEL_X = FRAME_W                      # 右侧信息面板起点
PANEL_BG = (28, 28, 28)
TIMELINE_BG = (18, 18, 18)
COLOR_ABNORMAL = (0, 0, 255)           # BGR 红
COLOR_NORMAL = (150, 150, 150)         # 灰
COLOR_GT = (70, 20, 20)                # 暗红（GT 异常带）
COLOR_CURVE = (230, 160, 40)           # 蓝绿曲线
COLOR_THRESHOLD = (0, 0, 255)
COLOR_TEXT = (235, 235, 235)

# 时间轴绘图区（在 PANEL 之外的底部面板内）
TL_Y0, TL_Y1 = 560, 700
TL_X0, TL_X1 = 70, CANVAS_W - 30


# ═══════════════════════════════════════════════════════════════════
# 基础绘制工具
# ═══════════════════════════════════════════════════════════════════

def _put(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 0.5,
    color: tuple[int, int, int] = COLOR_TEXT,
    thickness: int = 1,
) -> None:
    cv2.putText(
        img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness,
        cv2.LINE_AA,
    )


def _map_x(idx: int, n: int) -> int:
    if n <= 1:
        return TL_X0
    return int(TL_X0 + idx / (n - 1) * (TL_X1 - TL_X0))


def _build_timeline_background(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    lo: float,
    hi: float,
) -> np.ndarray:
    """预渲染时间轴静态层（GT 色带 + 阈值 + 完整曲线 + 坐标轴）。"""
    tl = np.full((CANVAS_H - 540, CANVAS_W, 3), TIMELINE_BG, dtype=np.uint8)
    n = len(scores)

    def y_of(s: float) -> int:
        return int(TL_Y1 - (s - lo) / max(hi - lo, 1e-6) * (TL_Y1 - TL_Y0))

    # GT 异常带（暗红竖线）
    for i in np.nonzero(labels > 0)[0]:
        x = _map_x(int(i), n)
        cv2.line(tl, (x, TL_Y0), (x, TL_Y1), COLOR_GT, 1)

    # 阈值虚线
    y_thr = y_of(float(threshold))
    for x in range(TL_X0, TL_X1, 16):
        cv2.line(tl, (x, y_thr), (min(x + 8, TL_X1), y_thr), COLOR_THRESHOLD, 1)
    _put(tl, f"thr {threshold:.2f}", (TL_X1 - 90, y_thr - 6), 0.42, COLOR_THRESHOLD)

    # 完整分数曲线
    pts = np.array(
        [[_map_x(i, n), y_of(float(scores[i]))] for i in range(n)],
        dtype=np.int32,
    )
    cv2.polylines(tl, [pts], isClosed=False, color=COLOR_CURVE, thickness=2)

    # 坐标轴与刻度
    cv2.line(tl, (TL_X0, TL_Y0), (TL_X0, TL_Y1), (90, 90, 90), 1)
    cv2.line(tl, (TL_X0, TL_Y1), (TL_X1, TL_Y1), (90, 90, 90), 1)
    _put(tl, f"{hi:.2f}", (TL_X0 - 60, TL_Y0 + 4), 0.42)
    _put(tl, f"{lo:.2f}", (TL_X0 - 60, TL_Y1 + 4), 0.42)
    _put(tl, "score", (TL_X0 - 62, (TL_Y0 + TL_Y1) // 2), 0.42)
    _put(tl, f"frame 0", (TL_X0, TL_Y1 + 18), 0.42)
    _put(tl, f"frame {n - 1}", (TL_X1 - 90, TL_Y1 + 18), 0.42)
    _put(tl, "GT abnormal (red band) | anomaly score curve", (TL_X0, 552), 0.46)
    return tl


# ═══════════════════════════════════════════════════════════════════
# MP4 写入（ffmpeg libx264，浏览器可播）
# ═══════════════════════════════════════════════════════════════════

class _Mp4Writer:
    """写 H.264/yuv420p MP4。

    优先用 PyAV（其内置 FFmpeg 带 libx264，浏览器/VSCode 可直接播放）；
    不可用时回退到 cv2 的 mp4v 编码器。
    """

    def __init__(self, path: Path, fps: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._av = None
        self._container = None
        self._stream = None
        self._cv: cv2.VideoWriter | None = None
        try:
            import av
            from fractions import Fraction

            self._av = av
            self._container = av.open(str(path), mode="w")
            self._stream = self._container.add_stream(
                "libx264", rate=Fraction(int(round(fps * 1000)), 1000),
            )
            self._stream.width = CANVAS_W
            self._stream.height = CANVAS_H
            self._stream.pix_fmt = "yuv420p"
            self._stream.options = {"crf": "23", "preset": "veryfast"}
        except Exception as e:  # noqa: BLE001
            log.warning("PyAV H.264 unavailable (%s); falling back to cv2 mp4v", e)
            self._av = None
            self._cv = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                (CANVAS_W, CANVAS_H),
            )

    def write(self, frame: np.ndarray) -> None:
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if self._container is not None:
            vf = self._av.VideoFrame.from_ndarray(frame, format="bgr24")
            for packet in self._stream.encode(vf):
                self._container.mux(packet)
        elif self._cv is not None:
            self._cv.write(frame)

    def close(self) -> None:
        if self._container is not None:
            for packet in self._stream.encode():
                self._container.mux(packet)
            self._container.close()
        if self._cv is not None:
            self._cv.release()


# ═══════════════════════════════════════════════════════════════════
# 单个视频渲染
# ═══════════════════════════════════════════════════════════════════

def render_video(
    video_path: Path,
    vid: str,
    scores: np.ndarray,
    labels: np.ndarray,
    explanation: dict,
    out_dir: Path,
    fps: float,
    threshold: float | None,
    top_k: int,
) -> dict:
    """渲染一个视频的 MP4 + GIF，返回给 index.md 用的摘要。"""
    n = len(scores)
    if threshold is None:
        # 展示用自适应阈值：取该视频 frame score 的 90 分位（top 10%）。
        # 说明：模型分数未经校准（不同视频分布不同），固定 0.5 会漏掉
        # 低分域视频；这里用相对阈值保证每个视频都能看出"最异常"的帧。
        threshold = float(np.quantile(scores, 0.90))
    lo = float(max(0.0, scores.min() - 0.05))
    hi = float(min(1.0, scores.max() + 0.05))
    if hi - lo < 0.2:
        mid = (hi + lo) / 2
        lo, hi = max(0.0, mid - 0.1), min(1.0, mid + 0.1)

    tl_bg = _build_timeline_background(scores, labels, threshold, lo, hi)
    base = np.full((CANVAS_H, CANVAS_W, 3), PANEL_BG, dtype=np.uint8)
    base[540:720, :] = TIMELINE_BG

    top_prompts = explanation.get("top_prompts", [])[:top_k]

    # 最异常片段 → GIF 窗口
    worst_start = int(explanation.get("start_frame", int(np.argmax(scores))))
    gif_lo = max(0, worst_start - 15)
    gif_hi = min(n, worst_start + 45)

    mp4_path = out_dir / f"{vid}_demo.mp4"
    gif_path = out_dir / f"{vid}_demo.gif"
    writer = _Mp4Writer(mp4_path, fps)
    gif_frames: list[np.ndarray] = []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    t = 0
    while t < n:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        canvas = base.copy()
        canvas[0:FRAME_H, 0:FRAME_W] = frame

        score = float(scores[t])
        is_abn = score >= threshold
        color = COLOR_ABNORMAL if is_abn else COLOR_NORMAL

        # 画面边框 + 角标
        cv2.rectangle(canvas, (2, 2), (FRAME_W - 3, FRAME_H - 3), color, 6)
        badge = f"{'ABNORMAL' if is_abn else 'NORMAL'}  {score:.2f}"
        (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(canvas, (16, 16), (16 + tw + 24, 16 + th + 22), color, -1)
        _put(canvas, badge, (28, 16 + th + 6), 1.0, (255, 255, 255), 2)

        # 右侧信息面板
        x_text = PANEL_X + 16
        y = 40
        _put(canvas, f"VIDEO {vid}", (x_text, y), 0.7)
        y += 34
        _put(canvas, f"frame {t} / {n - 1}", (x_text, y), 0.5)
        y += 26
        _put(canvas, f"time  {t / fps:.1f} s", (x_text, y), 0.5)
        y += 26
        gt = "ABNORMAL" if labels[t] > 0 else "normal"
        _put(canvas, f"GT    {gt}", (x_text, y), 0.5,
             COLOR_ABNORMAL if labels[t] > 0 else COLOR_NORMAL)
        y += 26
        _put(canvas, f"score {score:.3f}", (x_text, y), 0.5, color)
        y += 40

        _put(canvas, "TOP ABNORMAL PROMPTS", (x_text, y), 0.48)
        y += 26
        if top_prompts:
            for rank, p in enumerate(top_prompts, start=1):
                snippet = str(p.get("prompt", ""))[:30]
                sim = float(p.get("similarity", 0.0))
                _put(canvas, f"{rank}. {snippet}", (x_text, y), 0.42)
                y += 20
                _put(canvas, f"   sim={sim:.3f}", (x_text, y), 0.4, (170, 170, 170))
                y += 24
        else:
            _put(canvas, "(none)", (x_text, y), 0.42)

        # 底部时间轴：叠加已播放高亮 + 播放头
        x_t = _map_x(t, n)
        if x_t > TL_X0:
            sub = canvas[TL_Y0:TL_Y1, TL_X0:x_t]
            highlight = np.full_like(sub, (55, 55, 55))
            canvas[TL_Y0:TL_Y1, TL_X0:x_t] = cv2.addWeighted(
                sub, 0.78, highlight, 0.22, 0,
            )
        cv2.line(canvas, (x_t, TL_Y0 - 4), (x_t, TL_Y1 + 4),
                 COLOR_ABNORMAL, 2)

        writer.write(canvas)
        if gif_lo <= t < gif_hi and (t - gif_lo) % 2 == 0:
            gif_frames.append(cv2.resize(canvas, (640, 360)))
        t += 1

    cap.release()
    writer.close()

    # GIF（PIL）
    if gif_frames:
        from PIL import Image
        imgs = [
            Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            for f in gif_frames
        ]
        imgs[0].save(
            gif_path, save_all=True, append_images=imgs[1:],
            duration=int(1000 / fps * 2), loop=0, optimize=True,
        )

    # 最异常帧拼接图
    top_idx = np.argsort(scores)[::-1][:top_k]
    _save_top_frames(video_path, vid, scores, labels, top_idx,
                     out_dir / f"{vid}_top_frames.png")

    # 整段视频时间轴
    plot_temporal_heatmap(
        scores, labels,
        title=f"{vid} full-video anomaly timeline (threshold={threshold:.2f})",
        save_path=out_dir / f"timeline_{vid}.png",
        threshold=threshold,
    )

    return {
        "vid": vid,
        "num_frames": n,
        "num_anomaly": int((labels > 0).sum()),
        "threshold": threshold,
        "peak_score": float(scores.max()),
        "peak_gt": bool(labels[int(np.argmax(scores))] > 0),
        "top_prompt": (
            str(top_prompts[0]["prompt"]) if top_prompts else "n/a"
        ),
    }


def _save_top_frames(
    video_path: Path,
    vid: str,
    scores: np.ndarray,
    labels: np.ndarray,
    top_idx: np.ndarray,
    save_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = 4
    rows = int(np.ceil(len(top_idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.4))
    axes = np.atleast_1d(axes).ravel()

    cap = cv2.VideoCapture(str(video_path))
    for ax, idx in zip(axes, top_idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        ax.axis("off")
        if not ok or frame is None:
            continue
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        gt = "ABNORMAL" if labels[idx] > 0 else "normal"
        ax.set_title(
            f"frame {int(idx)} | score {float(scores[idx]):.3f} | GT {gt}",
            fontsize=9,
        )
    cap.release()
    for ax in axes[len(top_idx):]:
        ax.axis("off")
    fig.suptitle(f"{vid} top-{len(top_idx)} most anomalous frames", fontsize=12)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=140, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# index.md
# ═══════════════════════════════════════════════════════════════════

def _write_index(
    out_dir: Path,
    summaries: list[dict],
    explanations: dict,
    per_video: dict,
    ckpt: str,
    cfg_path: str,
    started: str,
) -> None:
    lines = [
        "# VLM-VAD 可视化演示",
        "",
        f"- 生成时间：{started}",
        f"- checkpoint：`{ckpt}`",
        f"- 配置：`{cfg_path}`",
        "- 逐帧标签来源：`motion_diff` 运动伪标签（**非官方 mask，仅供框架验证**）",
        "",
        "| Video | Frames | GT abnormal | Frame AUC | Peak score | Top abnormal prompt | MP4 | GIF | Top frames |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        vid = s["vid"]
        auc = per_video.get(vid, {}).get("frame_auc", float("nan"))
        lines.append(
            f"| {vid} | {s['num_frames']} | {s['num_anomaly']} | {auc:.3f} | "
            f"{s['peak_score']:.3f} | `{s['top_prompt']}` | "
            f"[mp4]({vid}_demo.mp4) | [gif]({vid}_demo.gif) | "
            f"[png]({vid}_top_frames.png) |"
        )

    lines += ["", "## 逐视频解释（template-based）", ""]
    for s in summaries:
        vid = s["vid"]
        exp = explanations.get(vid, {})
        lines.append(f"### Video {vid}")
        lines.append("")
        lines.append(f"- Frame AUC：{per_video.get(vid, {}).get('frame_auc', float('nan')):.4f}")
        lines.append(f"- 阈值：{s['threshold']:.3f}")
        lines.append(f"- 时间轴：[timeline_{vid}.png](timeline_{vid}.png)")
        lines.append("")
        lines.append(f"> {exp.get('explanation', '（无）')}")
        lines.append("")

    lines += [
        "## 说明与局限",
        "",
        "- 视频上文字为英文（容器内无中文字体），中文解释见上方。",
        "- 红框阈值默认取该视频 frame score 的 90 分位（展示用相对阈值，",
        "  非模型校准后的决策边界）；可用 `--threshold` 覆盖。",
        "- 模型只输出**帧级**分数，红框表示该帧分数超过阈值，不能定位画面中具体区域。",
        "- 当前标签为 `motion_diff` 伪标签，**不能作为学术结论**；获取官方二值 mask 后",
        "  切换 `frame_label_mode: pixel` 重新训练/评估。",
        "",
    ]
    (out_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Render VLM-VAD visual demo")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint .pt")
    parser.add_argument("--videos", nargs="+", default=["01", "06", "18"],
                        help="Video ids without extension, e.g. 01 06 18")
    parser.add_argument("--out", default="results/demo", help="Output directory")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Anomaly threshold for highlighting; default auto "
                             "(per-video 90th percentile)")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cfg = load_experiment_config(args.config)
    set_seed(int(cfg.seed))
    device = torch.device(str(cfg.device))
    out_dir = Path(args.out)
    ensure_dirs({"out": out_dir})
    save_config_snapshot(cfg, out_dir / "config.yaml")

    # ── 模型 + 数据 ──────────────────────────────────────────────
    model = build_model(cfg)
    matcher = build_matcher(cfg.model)
    state = load_checkpoint(args.ckpt)
    model.load_state_dict(state["model_state"])
    if "matcher_state" in state:
        matcher.load_state_dict(state["matcher_state"])
    log.info("Checkpoint loaded (epoch=%s, best_val=%s)",
             state.get("epoch", "?"), state.get("best_val_metric", "?"))

    eval_split = str(cfg.data.eval_split)
    loader, src = build_split_dataloader(
        cfg.data, split=eval_split, shuffle=False, seed=int(cfg.seed),
    )
    log.info("Eval data ready | split=%s source=%s batches=%d",
             eval_split, src, len(loader))

    # ── 推理（复用 eval.inference，拿到稠密逐帧分数）─────────────
    results = evaluate_videos(
        model=model,
        matcher=matcher,
        dataset=loader.dataset,
        device=device,
        batch_size=int(cfg.eval.batch_size),
        top_k=int(cfg.eval.top_k_prompts),
        out_dir=None,
        save_plots=False,
        cfg=cfg,
    )
    frame_scores = results["frame_score"]
    frame_labels = results["frame_label"]
    explanations = results["explanations"]
    per_video = results["per_video"]

    # ── 视频路径 ─────────────────────────────────────────────────
    root = _resolve_avenue_root(cfg.data.root)
    split_name = _normalize_split(eval_split)
    video_dir = root / f"{split_name}_videos"

    summaries: list[dict] = []
    for raw in args.videos:
        vid = str(raw).zfill(2)
        if vid not in frame_scores:
            log.error("Video %s not found in eval results; skipping", vid)
            continue
        video_path = video_dir / f"{vid}.avi"
        if not video_path.is_file():
            log.error("Video file missing: %s; skipping", video_path)
            continue
        log.info("Rendering %s (%d frames)...", vid, len(frame_scores[vid]))
        summary = render_video(
            video_path=video_path,
            vid=vid,
            scores=frame_scores[vid],
            labels=frame_labels[vid],
            explanation=explanations.get(vid, {}),
            out_dir=out_dir,
            fps=float(args.fps),
            threshold=args.threshold,
            top_k=int(args.top_k),
        )
        summaries.append(summary)
        log.info("  done: %s_demo.mp4 | peak=%.3f | GT=%d frames",
                 vid, summary["peak_score"], summary["num_anomaly"])

    if summaries:
        _write_index(
            out_dir, summaries, explanations, per_video,
            ckpt=str(args.ckpt), cfg_path=str(args.config), started=started,
        )
    log.info("Demo written to %s (%d videos)", out_dir, len(summaries))
    log.info("Open %s/index.md for the summary", out_dir)


if __name__ == "__main__":
    main()
