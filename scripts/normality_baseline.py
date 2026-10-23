#!/usr/bin/env python3
"""Fit a normal reference bank on training videos and evaluate official frames."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from omegaconf import OmegaConf
import torch

from datasets.builders import build_split_dataloader
from datasets.labels import load_frame_labels
from datasets.normal_features import load_normal_training_features
from eval.metrics import binary_ranking_metrics, reported_video_metrics
from eval.protocol import evaluation_provenance
from models.normality import NormalityMemory
from utils.config import load_experiment_config, save_config_snapshot
from utils.io import save_checkpoint, save_json
from utils.provenance import runtime_info
from utils.reproducibility import set_seed
from utils.visualization import plot_temporal_heatmap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/official_eval.yaml")
    parser.add_argument("--normality-config", default="configs/normality.yaml")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists() or args.checkpoint.exists():
        raise FileExistsError("Use new output/checkpoint paths to preserve prior experiments")
    cfg = load_experiment_config(args.config)
    params = OmegaConf.load(args.normality_config)
    q = float(params.calibration_quantile)
    if not 0 < q < 1 or cfg.data.frame_label_mode != "official":
        raise ValueError("Require official testing annotations and calibration quantile in (0,1)")
    set_seed(int(cfg.seed))
    device = torch.device(cfg.device)
    training, calibration, protocol = load_normal_training_features(cfg.data, cfg.model.backbone, int(cfg.seed))
    model, bank_indices = NormalityMemory.fit(torch.cat(list(training.values())),
        int(params.max_bank_frames), int(cfg.seed), k=int(params.k),
        query_chunk_size=int(params.query_chunk_size))
    model = model.to(device).eval()
    calibration_scores = {vid: model(feat.unsqueeze(0).to(device))["frame_score"][0].cpu().numpy()
                          for vid, feat in calibration.items()}
    threshold = float(np.quantile(np.concatenate(list(calibration_scores.values())), q))
    args.out_dir.mkdir(parents=True)
    save_config_snapshot(cfg, args.out_dir / "config.yaml")
    OmegaConf.save(params, args.out_dir / "normality_config.yaml")
    save_json(protocol, args.out_dir / "split_manifest.json")
    save_json(runtime_info(), args.out_dir / "environment.json")
    np.savez_compressed(args.out_dir / "calibration_scores.npz", **calibration_scores)
    save_checkpoint({"format_version": 1, "method": "normal_feature_memory", "model_state": model.state_dict(),
                     "parameters": OmegaConf.to_container(params), "bank_indices": bank_indices,
                     "training_protocol": protocol, "calibration_threshold": threshold,
                     "backbone": OmegaConf.to_container(cfg.model.backbone)}, args.checkpoint)

    # Test data is accessed only after fitting and normal-training calibration.
    loader, _ = build_split_dataloader(cfg.data, "testing", False, int(cfg.seed), cfg.model.backbone)
    dataset = loader.dataset
    if not hasattr(dataset, "_resolve_feature"):
        raise ValueError("Normality baseline requires the verified feature cache")
    scores, labels, per_video, explanations = {}, {}, {}, {}
    video_scores, video_labels = {}, {}
    for record in dataset.video_records:
        vid = record.video_id
        feat = dataset._resolve_feature(vid)
        score = model(feat.unsqueeze(0).to(device))["frame_score"][0].cpu().numpy()
        label = load_frame_labels(dataset.label_dir / f"{vid}.pt", record.num_frames).numpy()
        scores[vid], labels[vid] = score, label
        video_scores[vid], video_labels[vid] = float(score.max()), int(label.any())
        per_video[vid] = {**binary_ranking_metrics(score, label, "frame"),
                          "num_frames": len(label), "num_anomaly_frames": int(label.sum())}
        top = np.argsort(score)[::-1][:5].tolist()
        explanations[vid] = {"top_abnormal_frames": top, "peak_score": float(score.max()),
                             "score_threshold": threshold,
                             "explanation": f"第 {top[0]} 帧与正常训练特征库的最近邻距离最大。"
                                            "该分数表示视觉特征偏离，不确认具体异常行为。"}
        plot_temporal_heatmap(score, label, title=f"{vid}: normal-reference cosine distance / official GT",
                              save_path=args.out_dir / "plots" / f"{vid}_heatmap.png", threshold=threshold)
    all_scores, all_labels = np.concatenate(list(scores.values())), np.concatenate(list(labels.values()))
    metrics = {**binary_ranking_metrics(all_scores, all_labels, "frame"),
               **reported_video_metrics(video_scores, video_labels)}
    prediction = all_scores > threshold
    threshold_counts = {"tp": int((prediction & (all_labels == 1)).sum()),
                        "fp": int((prediction & (all_labels == 0)).sum()),
                        "tn": int((~prediction & (all_labels == 0)).sum()),
                        "fn": int((~prediction & (all_labels == 1)).sum())}
    save_json({"metrics": metrics, "per_video": per_video, "parameters": OmegaConf.to_container(params),
               "calibration": {"source": "held_out_training_videos_assumed_normal", "quantile": q,
                               "threshold": threshold, "test_confusion_counts": threshold_counts},
               **evaluation_provenance(dataset, protocol)}, args.out_dir / "metrics.json")
    save_json(explanations, args.out_dir / "explanations.json")
    np.savez_compressed(args.out_dir / "frame_scores.npz", **scores)
    np.savez_compressed(args.out_dir / "frame_labels.npz", **labels)
    print("Normality baseline", metrics, "training-calibrated threshold", threshold)


if __name__ == "__main__":
    main()
