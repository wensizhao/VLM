"""Runtime and checkpoint contracts for reproducible evaluation."""
from __future__ import annotations

import importlib.metadata
import platform
from typing import Any

import torch
from omegaconf import OmegaConf


def runtime_info() -> dict[str, Any]:
    packages = {}
    for name in ("torch", "torchvision", "open_clip_torch", "numpy", "scipy", "scikit-learn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "packages": packages,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "deterministic": torch.are_deterministic_algorithms_enabled()}


def validate_checkpoint(state: dict, cfg: Any, prompts: list[str]) -> None:
    if state.get("format_version") != 2:
        raise ValueError("Legacy checkpoint uses a different preprocessing/validation protocol; "
                         "use a v2 checkpoint for the current pipeline")
    if state["prompts"] != prompts:
        raise ValueError("Checkpoint prompt set differs from evaluation config")
    current = OmegaConf.to_container(cfg.model, resolve=True)
    if state["config"]["model"] != current:
        raise ValueError("Checkpoint model config differs from evaluation config")
