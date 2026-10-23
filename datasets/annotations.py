"""Parse explicit binary MAT annotations without inferring labels from video pixels."""
from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
from scipy.io import loadmat


def _binary_mask(array: np.ndarray) -> np.ndarray:
    if not np.issubdtype(array.dtype, np.number) and array.dtype != np.bool_:
        raise ValueError("Annotation mask must be numeric or boolean")
    if not np.isfinite(array).all():
        raise ValueError("Annotation mask contains non-finite values")
    values = set(np.unique(array).tolist())
    if not values <= {0, 1} and not values <= {0, 255}:
        raise ValueError(f"Expected binary annotation, found grayscale/multivalued data: {sorted(values)[:8]}")
    return array != 0


def frame_labels_from_mat(content: bytes, num_frames: int) -> tuple[np.ndarray, dict[str, Any]]:
    """MATLAB cell (1,N)/(N,1) or numeric mask volume → labels (N,).

    Frames remain in MATLAB cell order. A frame is abnormal iff any annotated pixel
    is nonzero. Numeric volumes require one unambiguous axis matching num_frames.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    variables = loadmat(BytesIO(content))
    keys = [key for key in ("volLabel", "vol") if key in variables]
    if len(keys) != 1:
        raise ValueError("Expected exactly one annotation variable: volLabel or vol")
    key = keys[0]
    volume = np.asarray(variables[key])
    if volume.dtype == object:
        if volume.ndim != 2 or 1 not in volume.shape or volume.size != num_frames:
            raise ValueError(f"Cell annotation frame count/shape mismatch: {volume.shape}, expected {num_frames}")
        frames = volume.reshape(-1, order="F")
        labels = []
        mask_shape = None
        for frame in frames:
            frame = np.asarray(frame)
            if frame.ndim != 2 or frame.size == 0:
                raise ValueError("Each annotation cell must be a nonempty 2D binary mask")
            if mask_shape is not None and frame.shape != mask_shape:
                raise ValueError("Annotation mask resolution changes between frames")
            mask_shape = frame.shape
            labels.append(int(_binary_mask(frame).any()))
        metadata = {"variable": key, "format": "matlab_cell", "mask_shape": list(mask_shape)}
    else:
        if volume.ndim != 3:
            raise ValueError(f"Annotation volume must have 3 axes, got {volume.shape}")
        axes = [axis for axis, size in enumerate(volume.shape) if size == num_frames]
        if len(axes) != 1:
            raise ValueError(f"Ambiguous or mismatched annotation time axis: {volume.shape}, frames={num_frames}")
        masks = np.moveaxis(_binary_mask(volume), axes[0], 0)
        labels = masks.reshape(num_frames, -1).any(axis=1).astype(np.int64)
        metadata = {"variable": key, "format": "numeric_volume", "time_axis": axes[0],
                    "mask_shape": list(masks.shape[1:])}
    return np.asarray(labels, dtype=np.int64), metadata
