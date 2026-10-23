"""CPU/ARM64 can import datasets and decode real frames without Decord installed."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.mark.parametrize("module_name,class_name", [
    ("datasets.video_dataset", "VideoDataset"),
    ("data", "AvenueDataset"),
])
def test_decode_without_decord(tmp_path, module_name, class_name):
    video = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (16, 16))
    assert writer.isOpened()
    for value in (20, 80, 140):
        writer.write(np.full((16, 16, 3), value, dtype=np.uint8))
    writer.release()

    # A fresh interpreter also catches accidental imports of Decord at module scope.
    code = '''
import importlib, sys
from pathlib import Path
import numpy as np
sys.modules["decord"] = None
module = importlib.import_module(sys.argv[1])
dataset = object.__new__(getattr(module, sys.argv[2]))
dataset.video_backend = "auto"
frames = dataset._read_video_frames(Path(sys.argv[3]), np.array([0, 2, 2]))
assert frames.shape == (3, 16, 16, 3)
assert np.allclose(frames.mean(axis=(1, 2, 3)), [20, 140, 140], atol=3)
dataset.video_backend = "opencv"
assert np.array_equal(dataset._read_video_frames(Path(sys.argv[3]), np.array([0, 2, 2])), frames)
dataset.video_backend = "decord"
try:
    dataset._read_video_frames(Path(sys.argv[3]), np.array([0]))
except ModuleNotFoundError:
    pass
else:
    raise AssertionError("Explicit decord must fail when unavailable")
'''
    result = subprocess.run(
        [sys.executable, "-c", code, module_name, class_name, str(video)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
