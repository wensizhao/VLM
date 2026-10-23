"""Studio uploads, caches and saved runs must follow the upstream dataset API."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from omegaconf import OmegaConf

from datasets.builders import _resolve_split_paths, build_split_dataloader
from test_smoke import FakeCLIPBackbone
from utils.config import load_experiment_config
from webapp import config, data_catalog, prepare_dataset, training


@pytest.fixture
def studio_upload(tmp_path, monkeypatch):
    paths = {
        "PROJECT_ROOT": tmp_path,
        "DATA_ROOT": tmp_path / "data",
        "UPLOAD_ROOT": tmp_path / "data/uploads",
        "RESULTS_ROOT": tmp_path / "results",
        "CHECKPOINT_ROOT": tmp_path / "checkpoints",
        "LOG_ROOT": tmp_path / "logs",
        "WEB_JOB_LOG_ROOT": tmp_path / "logs/web_jobs",
        "GENERATED_CONFIG_ROOT": tmp_path / "configs/generated",
        "OFFICIAL_LABEL_ROOT": tmp_path / "data/labels_official",
    }
    for module in (config, data_catalog, training):
        for name, path in paths.items():
            if hasattr(module, name):
                monkeypatch.setattr(module, name, path)
    monkeypatch.setattr(data_catalog, "_dataset_cache", None)
    root = paths["UPLOAD_ROOT"] / "selected"
    for split, ids in (("training", ("01", "02")), ("testing", ("03",))):
        directory = root / (split + "_videos")
        directory.mkdir(parents=True)
        for video_id in ids:
            writer = cv2.VideoWriter(str(directory / (video_id + ".avi")),
                                     cv2.VideoWriter_fourcc(*"MJPG"), 10, (16, 16))
            assert writer.isOpened()
            for frame in range(16):
                writer.write(np.full((16, 16, 3), frame * 10, dtype=np.uint8))
            writer.release()
    import tools.extract_video_features as extraction
    monkeypatch.setattr(extraction, "CLIPBackbone", FakeCLIPBackbone)
    monkeypatch.setattr(sys, "argv", ["prepare_dataset", "--root", str(root),
                                     "--features", "--device", "cpu"])
    prepare_dataset.main()
    return root


def _training_config():
    generated = training.build_training_config({
        "dataset_id": "uploads/selected", "run_name": "selected_upload",
        "label_mode": "motion_diff", "use_feature": "true", "device": "cpu",
    })
    return OmegaConf.load(generated["config_path"])


def test_uploaded_dataset_preparation_and_training_use_selected_paths(studio_upload, monkeypatch):
    cfg = _training_config()
    assert "root" not in cfg.data
    assert cfg.data.train.video_path == "data/uploads/selected/training_videos"
    assert cfg.data.test.video_path == "data/uploads/selected/testing_videos"
    assert cfg.data.train.mask_path is None
    assert (studio_upload / "features_v2/training/manifest.json").is_file()
    assert (studio_upload / "features_v2/testing/manifest.json").is_file()
    monkeypatch.chdir(config.PROJECT_ROOT)
    cached, source = build_split_dataloader(cfg.data, "training", False, 42, cfg.model.backbone)
    assert source == "feature"
    assert cached.dataset.video_dir == studio_upload / "training_videos"
    assert next(iter(cached))["vis_feat"].shape == (2, 16, 512)
    cfg.data.use_feature = False
    pixels, source = build_split_dataloader(cfg.data, "testing", False, 42, cfg.model.backbone)
    assert source == "video"
    assert pixels.dataset.video_dir == studio_upload / "testing_videos"
    assert pixels.dataset[0]["video"].shape == (16, 3, 16, 16)


def test_old_studio_run_is_migrated_without_rewriting_snapshot(studio_upload):
    cfg = _training_config()
    del cfg.data["train"]
    del cfg.data["test"]
    cfg.data.root = "data/uploads/selected"
    run_dir = config.RESULTS_ROOT / "old_run"
    run_dir.mkdir(parents=True)
    snapshot = run_dir / "config.yaml"
    OmegaConf.save(cfg, snapshot)
    original = snapshot.read_bytes()
    checkpoint = config.CHECKPOINT_ROOT / "old_run/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()  # Config generation does not load or execute the checkpoint.
    generated = training.build_eval_config("old_run", "motion")
    migrated = OmegaConf.load(generated["config_path"])
    assert "root" not in migrated.data
    assert migrated.data.test.video_path == "data/uploads/selected/testing_videos"
    assert snapshot.read_bytes() == original


@pytest.mark.parametrize("name", ["experiment", "official_eval", "live_smoke_official", "live_full_official"])
def test_checked_in_configs_resolve_explicit_split_paths(name):
    cfg = load_experiment_config(Path(__file__).resolve().parents[1] / "configs" / (name + ".yaml"))
    for split in ("training", "testing"):
        normalized, video_path, _ = _resolve_split_paths(cfg.data, split)
        assert normalized == split
        assert video_path.endswith(split + "_videos")


def test_studio_routes_and_video_preview_after_upstream_merge(studio_upload, monkeypatch):
    app_module = importlib.import_module("webapp.app")
    monkeypatch.setitem(app_module.app.config, "TESTING", True)
    # These modules can already be imported by another test; keep all IO isolated.
    for module in (app_module, importlib.import_module("webapp.runs")):
        for name in ("PROJECT_ROOT", "RESULTS_ROOT", "CHECKPOINT_ROOT", "OFFICIAL_LABEL_ROOT"):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, getattr(config, name))
    client = app_module.app.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/api/options").get_json()["prompt_types"] == ["label", "scene", "contrast"]
    datasets = client.get("/api/datasets").get_json()["datasets"]
    assert [item["id"] for item in datasets] == ["uploads/selected"]
    assert client.get("/api/datasets/detail", query_string={"id": "uploads/selected"}).status_code == 200
    preview = client.get("/api/datasets/frame", query_string={
        "id": "uploads/selected", "split": "training", "video": "01", "index": 0,
    })
    assert preview.status_code == 200
    assert preview.mimetype == "image/jpeg"
    assert len(preview.data) > 100
