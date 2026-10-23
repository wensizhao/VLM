"""Shared paths and validation helpers for VLM-VAD Studio."""
from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
UPLOAD_ROOT = DATA_ROOT / "uploads"
RESULTS_ROOT = PROJECT_ROOT / "results"
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"
LOG_ROOT = PROJECT_ROOT / "logs"
WEB_JOB_LOG_ROOT = LOG_ROOT / "web_jobs"
CONFIG_ROOT = PROJECT_ROOT / "configs"
GENERATED_CONFIG_ROOT = CONFIG_ROOT / "generated"
OFFICIAL_LABEL_ROOT = DATA_ROOT / "labels_official"
OFFICIAL_EVAL_CONFIG = CONFIG_ROOT / "official_eval.yaml"
BASE_TRAIN_CONFIG = CONFIG_ROOT / "experiment.yaml"

RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
DATASET_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def ensure_runtime_dirs() -> None:
    """Create writable runtime directories if they do not exist."""
    for path in (UPLOAD_ROOT, WEB_JOB_LOG_ROOT, GENERATED_CONFIG_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def sanitize_run_name(name: str) -> str:
    value = (name or "").strip()
    if not RUN_NAME_RE.fullmatch(value):
        raise ValueError(
            "Run name must be 1-64 chars using letters, digits, dot, dash or underscore"
        )
    return value


def sanitize_dataset_name(name: str) -> str:
    value = (name or "").strip().replace(" ", "_")
    if not DATASET_NAME_RE.fullmatch(value):
        raise ValueError(
            "Dataset name must be 1-64 chars using letters, digits, dot, dash or underscore"
        )
    return value


def rel_to_project(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Path is outside the project workspace: {resolved}") from exc


def is_within(path: str | Path, root: str | Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def resolve_under(path: str | Path, root: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()
    if not is_within(candidate, root):
        raise ValueError(f"Path escapes the allowed root: {candidate}")
    return candidate
