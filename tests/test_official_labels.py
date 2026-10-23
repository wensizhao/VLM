"""Official MAT parsing, dataset alignment, provenance, and undefined metrics."""
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import pytest
from scipy.io import savemat
import torch

from datasets.annotations import frame_labels_from_mat
from datasets.cache import validate_official_cache
from datasets.feature_dataset import FeatureDataset
from datasets.video_dataset import VideoDataset
from eval.metrics import binary_ranking_metrics, reported_video_metrics
from eval.protocol import evaluation_provenance
from tools.prepare_official_labels import prepare_labels


def mat_bytes(variable, value):
    stream = BytesIO()
    savemat(stream, {variable: value})
    return stream.getvalue()


def cell_masks(labels, shape=(16, 16)):
    cells = np.empty((1, len(labels)), dtype=object)
    for i, label in enumerate(labels):
        mask = np.zeros(shape, dtype=np.uint8)
        mask[3, 4] = label
        cells[0, i] = mask
    return cells


@pytest.mark.parametrize("transpose", [False, True])
def test_matlab_cell_frame_order(transpose):
    cells = cell_masks([1, 0, 0, 1])
    labels, meta = frame_labels_from_mat(mat_bytes("volLabel", cells.T if transpose else cells), 4)
    assert labels.tolist() == [1, 0, 0, 1]
    assert meta["mask_shape"] == [16, 16]


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_binary_volume_time_axis(axis):
    masks = np.zeros((4, 5, 6), np.uint8)
    masks[2, 1, 1] = 255
    labels, meta = frame_labels_from_mat(mat_bytes("vol", np.moveaxis(masks, 0, axis)), 4)
    assert labels.tolist() == [0, 0, 1, 0]
    assert meta["time_axis"] == axis


def test_reject_gray_wrong_count_and_ambiguous_time():
    with pytest.raises(ValueError, match="grayscale"):
        frame_labels_from_mat(mat_bytes("volLabel", cell_masks([0, 23, 0, 1])), 4)
    with pytest.raises(ValueError, match="count"):
        frame_labels_from_mat(mat_bytes("volLabel", cell_masks([0, 1])), 4)
    with pytest.raises(ValueError, match="Ambiguous"):
        frame_labels_from_mat(mat_bytes("vol", np.zeros((4, 4, 6))), 4)


@pytest.fixture
def official_data(tmp_path):
    root = tmp_path / "Avenue_Dataset"
    (root / "training_videos").mkdir(parents=True)
    (root / "testing_videos").mkdir()
    features = tmp_path / "features"
    features.mkdir()
    labels = [0, 0, 1, 1, 0, 1, 0, 0]
    writer = cv2.VideoWriter(str(root / "testing_videos/01.avi"),
                            cv2.VideoWriter_fourcc(*"MJPG"), 10, (16, 16))
    for _ in labels:
        writer.write(np.zeros((16, 16, 3), np.uint8))
    writer.release()
    torch.save(torch.randn(len(labels), 32), features / "01.pt")
    archive = tmp_path / "ground_truth.zip"
    with ZipFile(archive, "w") as package:
        package.writestr("demo/testing_label_mask/1_label.mat", mat_bytes("volLabel", cell_masks(labels)))
    return root, features, archive, tmp_path / "labels", labels


def test_prepare_and_both_input_paths_no_gray_vol_dependency(official_data):
    root, features, archive, out, expected = official_data
    meta = prepare_labels(archive, root, out)
    assert meta["num_frames"] == 8
    assert meta["num_anomaly_frames"] == 3
    kwargs = dict(video_dir=root / "testing_videos", split="testing", clip_length=4, clip_step=4,
                  frame_label_mode="official", label_dir=out / "testing")
    pixel = VideoDataset(**kwargs, video_backend="opencv", image_size=(8, 8), resize_mask_to_video=True)
    cached = FeatureDataset(features, **kwargs)
    for i in range(len(cached)):
        a, b = cached[i], pixel[i]
        assert torch.equal(a["frame_indices"], b["frame_indices"])
        assert torch.equal(a["frame_label"], b["frame_label"])
    assert torch.cat([cached[i]["frame_label"] for i in range(len(cached))]).tolist() == expected
    assert evaluation_provenance(cached)["benchmark_valid"]
    with pytest.raises(ValueError, match="testing split"):
        FeatureDataset(features, **{**kwargs, "split": "training"})
    with pytest.raises(FileExistsError):
        prepare_labels(archive, root, out)


def test_source_and_checksum_guard(official_data):
    root, features, archive, out, _ = official_data
    prepare_labels(archive, root, out)
    path = out / "testing"
    torch.save(torch.zeros(8, dtype=torch.long), path / "01.pt")
    with pytest.raises(ValueError, match="checksum"):
        FeatureDataset(features, root / "testing_videos", split="testing", frame_label_mode="official", label_dir=path)
    meta = json.loads((path / "manifest.json").read_text())
    meta["label_mode"] = "motion_diff"
    (path / "manifest.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="provenance"):
        validate_official_cache(path)


@pytest.mark.parametrize("failure", ["missing", "duplicate", "count", "resolution"])
def test_invalid_archive_does_not_publish(official_data, failure):
    root, _, archive, out, labels = official_data
    with ZipFile(archive, "w") as package:
        if failure != "missing":
            cells = cell_masks(labels[:-1] if failure == "count" else labels,
                               shape=(18, 16) if failure == "resolution" else (16, 16))
            package.writestr("demo/testing_label_mask/1_label.mat", mat_bytes("volLabel", cells))
            if failure == "duplicate":
                package.writestr("other/testing_label_mask/01_label.mat", mat_bytes("volLabel", cells))
    with pytest.raises(ValueError):
        prepare_labels(archive, root, out)
    assert not (out / "testing").exists()


def test_export_single_class_metrics_and_invalid_scores():
    assert binary_ranking_metrics([.1, .2], [1, 1], "frame") == {"frame_auc": None, "frame_ap": 1.0}
    assert binary_ranking_metrics([.1, .2], [0, 0], "frame") == {"frame_auc": None, "frame_ap": None}
    assert reported_video_metrics({"01": .2}, {"01": 1})["video_auc"] is None
    with pytest.raises(ValueError, match="finite"):
        binary_ranking_metrics([np.nan], [1], "frame")


def test_timeline_preserves_raw_negative_score_range(monkeypatch):
    from utils import visualization
    captured = {}
    def capture(fig, path):
        captured["limits"] = fig.axes[0].get_ylim()
        visualization.plt.close(fig)
    monkeypatch.setattr(visualization, "_save_or_show", capture)
    visualization.plot_temporal_heatmap(np.array([-.2, -.15, -.1]), np.array([0, 1, 0]))
    low, high = captured["limits"]
    assert low < -.2 and high > -.1 and high < 0
