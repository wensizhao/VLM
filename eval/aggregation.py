"""Shared scoring aggregation for supervised and zero-shot evaluation."""
import numpy as np


def accumulate_frames(accum: np.ndarray, counts: np.ndarray,
                      indices: np.ndarray, scores: np.ndarray) -> None:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.shape != scores.shape or (indices < 0).any() or (indices >= len(accum)).any():
        raise ValueError("Invalid frame indices or score shape")
    # add.at counts repeated indices correctly for padded short clips.
    np.add.at(accum, indices, scores)
    np.add.at(counts, indices, 1.0)


def dense_average(accum: np.ndarray, counts: np.ndarray) -> np.ndarray:
    if (counts == 0).any():
        raise ValueError(f"Evaluation left {int((counts == 0).sum())} frames uncovered; "
                         "use clip_stride=1 and clip_step <= clip_length")
    return accum / counts
