"""Check the selected image, video dependencies, device access, and bind permissions.

No model weights or dataset downloads; a CUDA check executes a small GPU operation.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import platform
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--allow-missing-gpu", action="store_true",
                        help="Only for image-build CI runners without NVIDIA hardware")
    args = parser.parse_args()

    import torch

    modules = ("torchvision", "cv2", "av", "open_clip", "omegaconf",
               "datasets.video_dataset", "data")
    for name in modules:
        importlib.import_module(name)

    available = torch.cuda.is_available()
    errors: list[str] = []
    report = {
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "uid": os.getuid(),
        "gid": os.getgid(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": available,
        "decord_installed": importlib.util.find_spec("decord") is not None,
        "imports": list(modules),
    }
    if args.expect == "cpu" and torch.version.cuda is not None:
        errors.append("Expected CPU-only PyTorch, but this image includes CUDA.")
    if args.expect == "cuda":
        if torch.version.cuda != "12.1":
            errors.append("Expected the pinned CUDA 12.1 PyTorch runtime.")
        if not available and not args.allow_missing_gpu:
            errors.append("CUDA is unavailable: check the host driver, NVIDIA Container Toolkit, and GPU Compose override.")

    device = "cuda" if args.expect == "cuda" and available else "cpu"
    tensor = torch.ones((2, 2), device=device)
    if not torch.equal(tensor @ tensor, torch.full_like(tensor, 2)):
        errors.append(f"Tensor operation failed on {device}.")
    report["tensor_device"] = device
    if available:
        report["gpu"] = torch.cuda.get_device_name(0)

    workspace = Path(os.environ.get("WORKSPACE_DIR", "/workspace/vlm_ws"))
    directories = [Path.home(), *(workspace / name for name in ("results", "logs", "checkpoints"))]
    for directory in directories:
        try:
            with tempfile.TemporaryFile(dir=directory):
                pass
        except OSError as exc:
            errors.append(f"Cannot write {directory}: {exc}")
    report["errors"] = errors
    print(json.dumps(report, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
