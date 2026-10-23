"""Attach ground-truth and training provenance to exported evaluations."""
from typing import Any


def evaluation_provenance(dataset: Any, training: dict | None = None) -> dict:
    mode = getattr(dataset, "frame_label_mode", "unknown")
    official = mode == "official" and hasattr(dataset, "label_provenance")
    return {
        "label_mode": mode,
        "benchmark_valid": official,
        "metric_protocol": "pooled_raw_frame_scores_any_mask_pixel_no_per_video_normalization",
        "annotation_scope": "frame detection; spatial localization is not evaluated",
        "undefined_metric_policy": "ROC-AUC=null for single class; AP=null without positives",
        "label_provenance": getattr(dataset, "label_provenance", None),
        "training_protocol": training or {"mode": "zero_shot", "test_labels_used_for_training": False},
    }
