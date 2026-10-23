"""Normality score geometry, deterministic fitting, and label-free partitions."""
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from datasets.cache import FEATURE_VERSION
from datasets.normal_features import load_normal_training_features
from models.normality import NormalityMemory


def test_normal_distance_output_and_reload(tmp_path):
    model = NormalityMemory(torch.tensor([[1., 0.], [1., 0.]]), k=2, query_chunk_size=1)
    features = torch.tensor([[[1., 0.], [0., 1.], [-1., 0.]]])
    output = model(features)
    torch.testing.assert_close(output["frame_score"], torch.tensor([[0., 1., 2.]]))
    assert output["anomaly_score"].shape == (1,)
    assert output["embedding"].shape == (1, 2)
    assert output["explanation"] is None
    path = tmp_path / "memory.pt"
    torch.save(model.state_dict(), path)
    state = torch.load(path, weights_only=True)
    restored = NormalityMemory(state["bank"], k=2)
    restored.load_state_dict(state)
    torch.testing.assert_close(restored(features)["frame_score"], output["frame_score"])


def test_seeded_bank_and_chunk_independence():
    vectors = torch.randn(100, 8, generator=torch.Generator().manual_seed(23))
    a, ia = NormalityMemory.fit(vectors, 20, 42, k=3, query_chunk_size=1)
    b, ib = NormalityMemory.fit(vectors, 20, 42, k=3, query_chunk_size=17)
    _, ic = NormalityMemory.fit(vectors, 20, 43, k=3)
    assert torch.equal(ia, ib) and not torch.equal(ia, ic)
    torch.testing.assert_close(a(vectors[:9].unsqueeze(0))["frame_score"],
                               b(vectors[:9].unsqueeze(0))["frame_score"], atol=2e-7, rtol=1e-5)


def test_invalid_bank_and_query():
    with pytest.raises(ValueError, match="nonzero"):
        NormalityMemory(torch.zeros(2, 4))
    model = NormalityMemory(torch.ones(5, 4))
    with pytest.raises(ValueError, match="matching"):
        model(torch.ones(1, 3, 6))
    with pytest.raises(ValueError, match="nonzero"):
        model(torch.zeros(1, 3, 4))
    with pytest.raises(ValueError, match="k must"):
        NormalityMemory(torch.ones(3, 4), k=5)


def test_training_only_features_no_annotation_dependency(tmp_path):
    root = tmp_path / "Avenue_Dataset"
    (root / "training_videos").mkdir(parents=True)
    cache = tmp_path / "features/training"
    cache.mkdir(parents=True)
    (cache / "manifest.json").write_text(json.dumps({"model_name": "test", "pretrained": "test",
                                                     "preprocessing": FEATURE_VERSION}))
    for vid in range(1, 5):
        name = f"{vid:02d}"
        writer = cv2.VideoWriter(str(root / "training_videos" / f"{name}.avi"),
                                cv2.VideoWriter_fourcc(*"MJPG"), 10, (16, 16))
        for _ in range(4):
            writer.write(np.zeros((16, 16, 3), np.uint8))
        writer.release()
        torch.save(torch.full((4, 8), float(vid)), cache / f"{name}.pt")
    cfg = SimpleNamespace(root=root, feature_dir=tmp_path / "features", validation_fraction=.25)
    backbone = {"name": "test", "pretrained": "test", "freeze_vision": True}
    fit, calibration, protocol = load_normal_training_features(cfg, SimpleNamespaceDict(backbone), 42)
    assert len(fit) == 3 and len(calibration) == 1
    assert set(fit).isdisjoint(calibration)
    assert not protocol["motion_labels_used"] and not protocol["test_labels_used_for_training"]
    assert protocol["source_split"] == "training"
    torch.save(torch.ones(3, 8), cache / "01.pt")
    with pytest.raises(ValueError, match="Invalid"):
        load_normal_training_features(cfg, SimpleNamespaceDict(backbone), 42)


class SimpleNamespaceDict(dict):
    __getattr__ = dict.__getitem__
