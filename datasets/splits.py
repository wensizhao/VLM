"""Video-disjoint validation, preserving video metadata for dense inference."""
from __future__ import annotations

import math
import random
from typing import Any

from torch.utils.data import Dataset


class VideoSubset(Dataset):
    """Select whole videos, never split overlapping clips across partitions."""

    def __init__(self, dataset: Dataset, video_ids: list[str]) -> None:
        self.dataset = dataset
        selected = set(video_ids)
        self.video_records = [r for r in dataset.video_records if r.video_id in selected]
        self.indices = [i for i, c in enumerate(dataset.clip_records)
                        if dataset.video_records[c.video_index].video_id in selected]
        self.split = dataset.split
        self.frame_label_mode = dataset.frame_label_mode
        if not self.indices:
            raise ValueError("Video partition has no clips")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.dataset[self.indices[index]]


def split_training_videos(dataset: Dataset, fraction: float, seed: int) -> tuple[VideoSubset, VideoSubset]:
    """Hold out a seeded fraction of training videos; test data is never read."""
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    ids = [r.video_id for r in dataset.video_records]
    if len(ids) < 2:
        raise ValueError("At least two training videos are required for validation")
    random.Random(seed).shuffle(ids)
    count = min(len(ids) - 1, max(1, math.ceil(len(ids) * fraction)))
    return VideoSubset(dataset, sorted(ids[count:])), VideoSubset(dataset, sorted(ids[:count]))
