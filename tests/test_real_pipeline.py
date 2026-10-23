"""Opt-in GPU tests with real cached CLIP weights and Avenue videos.

VLM_REAL_INTEGRATION=1 HF_HUB_OFFLINE=1 python -m pytest -q tests/test_real_pipeline.py
CI skips these checks; ordinary regressions never download model weights.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Subset

from datasets.builders import build_split_dataloader
from models.backbone import CLIPBackbone
from models.factory import build_matcher, build_model
from train.trainer import Trainer, resume_from_checkpoint
from utils.config import load_experiment_config
from utils.reproducibility import set_seed

pytestmark = pytest.mark.skipif(os.environ.get("VLM_REAL_INTEGRATION") != "1",
                                reason="Requires real Avenue caches and CLIP weights")


def test_real_pixel_features_equal_cache():
    set_seed(42)
    cfg = load_experiment_config("configs/experiment.yaml")
    cfg.data.num_workers = 0
    cached, _ = build_split_dataloader(cfg.data, "training", False, 42, cfg.model.backbone)
    cfg.data.use_feature = False
    cfg.data.video_backend = "opencv"
    pixels, _ = build_split_dataloader(cfg.data, "training", False, 42, cfg.model.backbone)
    sample = pixels.dataset[0]
    backbone = CLIPBackbone().cuda().eval()
    with torch.no_grad():
        encoded = backbone.encode_video(sample["video"].unsqueeze(0).cuda()).cpu()[0]
    torch.testing.assert_close(encoded, cached.dataset[0]["vis_feat"], atol=1e-5, rtol=1e-4)
    assert torch.equal(sample["frame_label"], cached.dataset[0]["frame_label"])


def test_real_official_annotation_alignment():
    from zipfile import ZipFile
    import numpy as np
    from datasets.annotations import frame_labels_from_mat
    from datasets.labels import load_frame_labels

    cfg = load_experiment_config("configs/official_eval.yaml")
    loader, _ = build_split_dataloader(cfg.data, "testing", False, 42, cfg.model.backbone)
    dataset = loader.dataset
    assert dataset.label_provenance["num_frames"] == 15324
    assert dataset.label_provenance["num_anomaly_frames"] == 3712
    cfg.data.use_feature = False
    cfg.data.video_backend = "opencv"
    pixel, _ = build_split_dataloader(cfg.data, "testing", False, 42, cfg.model.backbone)
    for index in (0, len(dataset) - 1):
        assert torch.equal(dataset[index]["frame_label"], pixel.dataset[index]["frame_label"])
    with ZipFile("data/ground_truth_demo.zip") as package:
        for record in dataset.video_records:
            if record.video_id not in {"01", "08", "18"}:
                continue
            content = package.read(f"ground_truth_demo/testing_label_mask/{int(record.video_id)}_label.mat")
            expected, _ = frame_labels_from_mat(content, record.num_frames)
            actual = load_frame_labels(dataset.label_dir / f"{record.video_id}.pt", record.num_frames)
            np.testing.assert_array_equal(actual.numpy(), expected)


def test_real_amp_training_validation_and_resume(tmp_path):
    cfg = load_experiment_config("configs/experiment.yaml")
    cfg.data.num_workers = 0
    cfg.train.epochs = 2
    cfg.train.amp = True
    cfg.train.warmup_steps = 0
    cfg.prompt.types = ["scene"]
    cfg.paths.result_dir = str(tmp_path / "results")
    source, _ = build_split_dataloader(cfg.data, "training", False, 42, cfg.model.backbone)
    dataset = Subset(source.dataset, list(range(8)))

    def make(name):
        model = build_model(cfg)
        loader = DataLoader(dataset, batch_size=4, shuffle=True,
                            generator=torch.Generator().manual_seed(42))
        return Trainer(model, build_matcher(cfg.model), loader, DataLoader(dataset, batch_size=4),
                       cfg, torch.device("cuda"), tmp_path / "logs", tmp_path / "checkpoints", name)

    set_seed(42)
    first = make("first")
    stats = first.train_epoch(1)
    assert stats["bce"] > 0 and stats["contrastive"] > 0
    validation = first.validate_epoch(1)  # autocast text cache must not leak fp16 into fp32 eval
    first._save_checkpoint(1, validation, True)
    first.train_epoch(2)
    expected = {k: v.detach().cpu().clone() for k, v in first.model.state_dict().items()
                if not k.startswith("backbone")}
    restored = make("restored")
    assert resume_from_checkpoint(restored, tmp_path / "checkpoints/first/last.pt") == 1
    restored.train_epoch(2)
    for key, value in restored.model.state_dict().items():
        if key in expected:
            torch.testing.assert_close(value.cpu(), expected[key], atol=0, rtol=0)
    first.tb.close()
    restored.tb.close()
