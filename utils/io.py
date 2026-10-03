# 这个文件负责 checkpoint 保存/加载、JSON/YAML 读写等磁盘 IO 工具。
"""IO utilities — checkpointing, JSON/YAML serialization, directory setup.

checkpoint 里保存什么？
-----------------------
训练到一半，我们需要把"恢复训练所需的最小状态集"存下来：
- ``model``        —— 网络权重（必须）
- ``optimizer``    —— 优化器动量/Adam 矩估计（要接着训就必须存）
- ``scheduler``    —— 学习率调度器的步数状态（要接着训就必须存）
- ``epoch``        —— 当前 epoch（用于恢复日志步数）
- ``metrics``      —— 到目前为止的最佳指标（用于"超越才保存"判断）
- ``prompt_state`` —— PromptProcessor 配置（prompt 变了，旧模型语义就变了）

保存优化器状态的意义：Adam 维护每个参数的 ``m``（一阶矩，近似梯度均值）和
``v``（二阶矩，近似梯度平方均值）。如果不存，恢复后 Adam 从零起步，
等效于"学习率瞬间跳变"，会破坏训练曲线。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .logging import get_logger

log = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 目录
# ═══════════════════════════════════════════════════════════════════

def ensure_dirs(paths: dict[str, str | Path]) -> dict[str, Path]:
    """确保所有输出目录存在，返回解析后的 Path 字典。"""
    out: dict[str, Path] = {}
    for key, p in paths.items():
        path = Path(p)
        path.mkdir(parents=True, exist_ok=True)
        out[key] = path
    return out


# ═══════════════════════════════════════════════════════════════════
# JSON / YAML
# ═══════════════════════════════════════════════════════════════════

def save_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    """把任意 JSON 可序列化对象写到磁盘。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    """从磁盘读回 JSON。"""
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════
# Checkpoint
# ═══════════════════════════════════════════════════════════════════

def save_checkpoint(
    state: dict[str, Any],
    path: str | Path,
) -> None:
    """保存 checkpoint。

    Args:
        state: 要保存的状态字典（model / optimizer / epoch / metrics ...）。
        path: 保存路径（后缀自动补 .pt）。
    """
    path = Path(path)
    if path.suffix != ".pt":
        path = path.with_suffix(path.suffix + ".pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))
    log.info("Checkpoint saved to %s", path)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """加载 checkpoint 到 CPU 内存（方便跨设备迁移）。

    Args:
        path: checkpoint 文件路径。

    Returns:
        状态字典。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    log.info("Checkpoint loaded from %s", path)
    return state
