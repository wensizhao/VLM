"""Regression tests for experimental integrity and actual training state changes."""
from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from scipy.io import savemat
from torch.utils.data import DataLoader

from datasets.cache import FEATURE_VERSION, validate_feature_cache, validate_motion_cache
from datasets.feature_dataset import FeatureDataset
from datasets.splits import split_training_videos
from datasets.video_dataset import VideoDataset, _load_mask_volume
from eval.aggregation import accumulate_frames, dense_average
from eval.inference import evaluate_videos
from models.factory import build_matcher, build_model, build_prompt_processor
from train.experiments import experiment_matrix, run_matrix
from train.losses import VLMVADLoss, build_prompt_polarity
from train.trainer import Trainer, resume_from_checkpoint
from utils.config import load_experiment_config, save_config_snapshot
from utils.io import save_json
from utils.reproducibility import set_seed
from test_smoke import FakeCLIPBackbone


@pytest.fixture
def config(tmp_path):
    cfg = load_experiment_config(Path(__file__).resolve().parents[1] / "configs/experiment.yaml")
    cfg.device = "cpu"
    cfg.model.alignment.aligned_dim = 32
    cfg.model.fusion.out_dim = 24  # intentionally different from matcher space
    cfg.model.head.hidden_dim = 16
    cfg.train.epochs = 2
    cfg.train.warmup_steps = 0
    cfg.train.log_every = 2
    cfg.paths.result_dir = str(tmp_path / "results")
    return cfg


@pytest.fixture
def model_factory(monkeypatch):
    import models.factory as factory
    monkeypatch.setattr(factory, "CLIPBackbone",
                        lambda **kwargs: FakeCLIPBackbone(**kwargs, dim=32, seq_len=8))
    return build_model


@pytest.fixture
def avenue(tmp_path):
    root = tmp_path / "Avenue_Dataset"
    features = tmp_path / "features"
    labels = tmp_path / "labels"
    (root / "training_videos").mkdir(parents=True)
    (root / "training_vol").mkdir()
    features.mkdir()
    labels.mkdir()
    n = 12
    for vid in ("01", "02", "03", "04"):
        writer = cv2.VideoWriter(str(root / "training_videos" / f"{vid}.avi"),
                                 cv2.VideoWriter_fourcc(*"MJPG"), 10, (16, 16))
        for i in range(n):
            writer.write(np.full((16, 16, 3), (i // 3) * 20, dtype=np.uint8))
        writer.release()
        savemat(root / "training_vol" / f"vol{vid}.mat", {"vol": np.zeros((4, 4, n), np.uint8)})
        torch.save(torch.randn(n, 32), features / f"{vid}.pt")
        torch.save(torch.tensor([i % 2 for i in range(n)]), labels / f"{vid}.pt")
    return root, features, labels


def feature_data(avenue, **kwargs):
    root, features, labels = avenue
    return FeatureDataset(features, root / "training_videos", clip_length=4, clip_step=4,
                          label_dir=labels, frame_label_mode="motion_diff", **kwargs)


def test_pixel_mode_never_uses_pseudo_labels(avenue):
    root, features, labels = avenue
    ds = FeatureDataset(features, root / "training_videos", clip_length=4, label_dir=labels, frame_label_mode="pixel")
    assert ds[0]["frame_label"].sum() == 0
    assert feature_data(avenue)[0]["frame_label"].sum() == 2


def test_label_count_and_values_checked(avenue):
    _, _, labels = avenue
    ds = feature_data(avenue)
    torch.save(torch.ones(3), labels / "01.pt")
    with pytest.raises(ValueError, match="shape"):
        ds[0]
    torch.save(torch.full((12,), 0.5), labels / "01.pt")
    with pytest.raises(ValueError, match="binary"):
        ds[0]


def test_reject_grayscale_as_pixel_masks(tmp_path):
    path = tmp_path / "gray.mat"
    frames = (np.arange(16 * 64) % 200 + 1).reshape(16, 8, 8).astype(np.uint8)
    cells = np.empty((1, 16), dtype=object)
    for i in range(16):
        cells[0, i] = frames[i]
    savemat(path, {"volLabel": cells})
    with pytest.raises(ValueError, match="grayscale"):
        _load_mask_volume(str(path))


def test_video_disjoint_seeded_partition(avenue):
    ds = feature_data(avenue)
    tr, val = split_training_videos(ds, 0.25, 42)
    tr2, val2 = split_training_videos(ds, 0.25, 42)
    assert set(r.video_id for r in tr.video_records).isdisjoint(r.video_id for r in val.video_records)
    assert tr.indices == tr2.indices and val.indices == val2.indices
    assert len(tr) + len(val) == len(ds)


def test_stride_and_repeated_indices_accumulate_correctly():
    accum, count = np.zeros(5), np.zeros(5)
    accumulate_frames(accum, count, np.array([0, 2, 4, 4]), np.array([1., 2., 3., 5.]))
    assert accum.tolist() == [1., 0., 2., 0., 8.]
    assert count.tolist() == [1., 0., 1., 0., 2.]
    with pytest.raises(ValueError, match="uncovered"):
        dense_average(accum, count)
    accumulate_frames(accum, count, np.array([1, 3]), np.array([6., 7.]))
    assert dense_average(accum, count).tolist() == [1., 6., 2., 7., 4.]


def test_demo_timeline_has_curve_in_visible_bottom_panel():
    from scripts.visualize_demo import _build_timeline_background, CANVAS_H, CANVAS_W, COLOR_CURVE
    image = _build_timeline_background(np.linspace(0.1, 0.9, 20), np.ones(20), 0.5, 0., 1., "PSEUDO")
    assert image.shape == (CANVAS_H, CANVAS_W, 3)
    assert np.all(image[540:] == np.array(COLOR_CURVE), axis=-1).sum() > 100


def test_motion_labels_independent_of_clip_boundaries(avenue):
    root, _, _ = avenue
    ds = VideoDataset(root / "training_videos", clip_length=4, clip_step=3, frame_label_mode="motion_diff",
                      image_size=None, motion_threshold=3.0, video_backend="opencv")
    a, b = ds[0], ds[1]
    assert a["frame_indices"][-1] == b["frame_indices"][0]
    assert a["frame_label"][-1] == b["frame_label"][0] == 1


def test_motion_extraction_chunk_boundary(monkeypatch, tmp_path):
    import tools.extract_motion_labels as module
    vr = SimpleNamespace(video_id="01", num_frames=257, video_path=Path("unused"))
    monkeypatch.setattr(module, "VideoDataset", lambda **kwargs: SimpleNamespace(video_records=[vr]))
    monkeypatch.setattr(module, "_read_frames_opencv",
                        lambda path, indices: np.where(indices[:, None, None, None] >= 256, 100, 0)
                        .astype(np.uint8) * np.ones((len(indices), 2, 2, 3), np.uint8))
    module.extract("unused", "training", 3.0, tmp_path)
    labels = torch.load(tmp_path / "training/01.pt", weights_only=True)
    assert labels.shape == (257,) and labels[-1] == 1 and labels[:-1].sum() == 0
    validate_motion_cache(tmp_path / "training", 3.0)


def test_cache_identity_rejects_wrong_or_unknown_encoder(config, tmp_path):
    with pytest.raises(ValueError, match="legacy"):
        validate_feature_cache(tmp_path, config.model.backbone)
    meta = {"model_name": "ViT-B-32", "pretrained": config.model.backbone.pretrained,
            "preprocessing": FEATURE_VERSION}
    save_json(meta, tmp_path / "manifest.json")
    validate_feature_cache(tmp_path, config.model.backbone)
    config.model.backbone.name = "ViT-B-16"
    with pytest.raises(ValueError, match="mismatch"):
        validate_feature_cache(tmp_path, config.model.backbone)


def test_preprocessing_normalizes_pixels(monkeypatch):
    import models.backbone as module
    dummy = torch.nn.Module()
    dummy.transformer = SimpleNamespace(width=32)
    monkeypatch.setattr(module.open_clip, "create_model_and_transforms", lambda *a, **k: (dummy, None, None))
    monkeypatch.setattr(module.open_clip, "get_tokenizer", lambda name: None)
    monkeypatch.setattr(module.open_clip, "get_model_preprocess_cfg", lambda model:
                        {"size": 16, "mean": (0.2, 0.3, 0.4), "std": (0.5, 0.5, 0.5)})
    backbone = module.CLIPBackbone()
    output = backbone.preprocess_images(torch.ones(2, 3, 20, 40))
    assert output.shape == (2, 3, 16, 16)
    torch.testing.assert_close(output[:, :, 8, 8], torch.tensor([[1.6, 1.4, 1.2]]).expand(2, -1))


@pytest.mark.parametrize("prompt_type", ["label", "scene", "contrast"])
def test_train_prompt_switch_and_output_contract(config, model_factory, prompt_type):
    config.prompt.prompt_type = prompt_type
    model = model_factory(config)
    assert {t for t, _ in model.prompt_processor.process_with_types()} == {prompt_type}
    output = model.forward_from_visual(torch.randn(2, 4, 32))
    mapping = output.to_dict()
    assert {"anomaly_score", "frame_score", "embedding", "explanation"} <= mapping.keys()
    assert mapping["anomaly_score"].shape == (2,)
    mapping["frame_score"].sum().backward()  # conversion preserves autograd tensors


def make_trainer(config, model_factory, ds, tmp_path):
    model = model_factory(config)
    loader = DataLoader(ds, batch_size=2, shuffle=True,
                        generator=torch.Generator().manual_seed(42))
    return Trainer(model, build_matcher(config.model), loader, DataLoader(ds, batch_size=2),
                   config, torch.device("cpu"), tmp_path / "logs", tmp_path / "checkpoints", "regression")


def test_matcher_parameters_update_and_normal_bce_trains(config, model_factory, avenue, tmp_path):
    config.model.matcher.strategy = "learnable"
    trainer = make_trainer(config, model_factory, feature_data(avenue), tmp_path)
    before = {k: v.detach().clone() for k, v in trainer.matcher.named_parameters()}
    batch = next(iter(trainer.train_loader))
    batch["frame_label"].zero_()
    batch["clip_label"].zero_()
    stats = trainer._train_step(batch, 0)
    assert stats["bce"] > 0
    for key, value in trainer.matcher.named_parameters():
        assert not torch.equal(before[key], value), key
        assert value.grad is not None and torch.isfinite(value.grad).all()
    trainer.tb.close()


def test_resume_matches_uninterrupted_training(config, model_factory, avenue, tmp_path):
    ds = feature_data(avenue)
    set_seed(42)
    first = make_trainer(config, model_factory, ds, tmp_path / "first")
    first.train_epoch(1)
    first._save_checkpoint(1, {"clip_auc": 0.5}, True)
    expected_rng = (random.random(), float(np.random.rand()), float(torch.rand(())))
    first.train_epoch(2)
    expected = {k: v.clone() for k, v in first.model.state_dict().items()}
    restored = make_trainer(config, model_factory, ds, tmp_path / "restored")
    assert resume_from_checkpoint(restored, tmp_path / "first/checkpoints/regression/last.pt") == 1
    actual_rng = (random.random(), float(np.random.rand()), float(torch.rand(())))
    assert expected_rng == actual_rng
    restored.train_epoch(2)
    for key, value in restored.model.state_dict().items():
        assert torch.equal(value, expected[key]), key
    first.tb.close()
    restored.tb.close()


def test_evaluation_uses_aligned_space_and_global_frame_ranking(config, model_factory, avenue, tmp_path):
    result = evaluate_videos(model_factory(config), build_matcher(config.model), feature_data(avenue),
                             torch.device("cpu"), out_dir=tmp_path, cfg=config, save_plots=True)
    assert all(v.shape == (12,) for v in result["frame_score"].values())
    assert (tmp_path / "plots/01_heatmap.png").is_file()
    assert (tmp_path / "frame_scores.npz").is_file()
    for entry in result["explanations"].values():
        assert all(i >= entry["start_frame"] for i in entry["top_abnormal_frames"])


def test_zero_shot_baseline_does_not_change_trained_frame_scores(config, model_factory, avenue):
    model = model_factory(config)
    matcher = build_matcher(config.model)
    dataset = feature_data(avenue)
    with_baseline = evaluate_videos(model, matcher, dataset, torch.device("cpu"),
                                    cfg=config, save_plots=False)
    without_baseline = evaluate_videos(model, matcher, dataset, torch.device("cpu"),
                                       cfg=None, save_plots=False)
    assert "zs_frame_auc" in with_baseline["metrics"]
    assert "zs_frame_auc" not in without_baseline["metrics"]
    for video_id, scores in with_baseline["frame_score"].items():
        np.testing.assert_allclose(scores, without_baseline["frame_score"][video_id])


def test_config_snapshot_is_independent_and_overrides_are_respected(config, tmp_path):
    config.prompt.templates.scene = "custom {scene}"
    path = save_config_snapshot(config, tmp_path / "config.yaml")
    reloaded = load_experiment_config(path)
    assert reloaded.prompt.templates.scene == "custom {scene}"
    assert "config_path" not in reloaded.prompt


def test_registered_encoder_pair_is_swappable(config, monkeypatch):
    import models.factory as factory
    from models.registry import BackboneRegistry
    registry = BackboneRegistry()
    registry.register("test_encoder_pair", lambda cfg: FakeCLIPBackbone(dim=32, seq_len=8))
    monkeypatch.setattr(factory, "backbones", registry)
    config.model.backbone.provider = "test_encoder_pair"
    output = factory.build_model(config)(torch.randn(2, 4, 3, 16, 16))
    assert output.anomaly_score.shape == (2,) and output.embedding.shape == (2, 24)
    with pytest.raises(ValueError, match="Unknown"):
        registry.build("missing", config.model.backbone)


def test_checkpoint_rejects_changed_prompt_or_model(config, model_factory):
    from utils.provenance import validate_checkpoint
    prompts = model_factory(config).prompt_processor.process()
    state = {"format_version": 2, "prompts": prompts,
             "config": OmegaConf.to_container(config, resolve=True)}
    validate_checkpoint(state, config, prompts)
    with pytest.raises(ValueError, match="prompt"):
        validate_checkpoint(state, config, prompts[:-1])
    config.model.fusion.type = "crossattn"
    with pytest.raises(ValueError, match="model"):
        validate_checkpoint(state, config, prompts)


def test_sweep_dimensions_and_failure_status(config, tmp_path, monkeypatch):
    sweep = OmegaConf.create({"name": "test", "epochs": 1, "prompt_types": ["label", "scene", "contrast"],
                              "fusions": ["concat", "crossattn"], "seeds": [42, 43],
                              "backbones": [{"id": "b32", "name": "ViT-B-32", "pretrained": "none",
                                             "feature_dir": "features_b32"},
                                            {"id": "b16", "name": "ViT-B-16", "pretrained": "none",
                                             "feature_dir": "features_b16"}]})
    plan = experiment_matrix(config, sweep)
    assert len(plan) == 24
    assert config.prompt.get("types") is None
    assert len({n for n, _ in plan}) == 24
    assert {c.data.feature_dir for _, c in plan} == {"features_b32", "features_b16"}
    sweep.prompt_types = ["label"]
    sweep.fusions = ["concat"]
    sweep.seeds = [42]
    sweep.backbones = [sweep.backbones[0]]
    monkeypatch.setattr("train.experiments.subprocess.run", lambda *a, **k: SimpleNamespace(returncode=3))
    result = run_matrix(config, sweep, tmp_path / "sweep")
    assert result[0]["status"] == "failed" and result[0]["returncode"] == 3
