# 这个文件负责日志（Python logging）和训练指标的平滑统计。
"""Logging utilities — Python logger + in-memory metric tracking for training loops.

两个互补的日志工具：
- ``get_logger``：标准库 ``logging`` 的封装，把训练过程打到终端和文件。
- ``MetricLogger``：在内存里跟踪一组数值指标的滑动平均，供进度条/终端/TensorBoard 使用。

数学背景：滑动平均（moving average）
-------------------------------------
训练时 loss 波动大，直接打裸 loss 看不出趋势。滑动平均使用指数加权：

    ema  ←  beta * ema  +  (1 - beta) * value

- beta 越大 → 对历史越"念旧"，曲线越平滑，但对变化反应越慢
- beta = 0   → 退化成直接输出当前值（不平滑）

它的本质是给历史样本一个指数衰减的权重（越久远的样本权重越低），
是深度学习里监控训练曲线最常用的手段（TensorBoard 的 smoothing 就是它）。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterable

from torch import Tensor

# 模块级 logger 名字空间，避免不同文件重复注册 handler
_LOGGERS: set[str] = set()


def get_logger(
    name: str = "vlm-vad",
    log_file: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """获取（或创建）一个配置好的 logger。

    Args:
        name: logger 名称（同名 logger 复用，避免重复 handler）。
        log_file: 可选——同时把日志写到文件。
        level: 日志级别。

    Returns:
        配置完成的 logger。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # 不向 root logger 传播，避免重复打印

    if name in _LOGGERS:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 终端 handler
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    # 文件 handler（可选）
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_file))
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    _LOGGERS.add(name)
    return logger


def _as_float(value: float | Tensor) -> float:
    """把标量张量转成 Python float（脱离子图，避免无意中保存计算图）。"""
    if isinstance(value, Tensor):
        return float(value.detach().cpu().item())
    return float(value)


class MetricLogger:
    """滑动平均指标跟踪器——一个对象同时跟踪多个指标。

    Usage::

        ml = MetricLogger(beta=0.98)
        for batch in loader:
            ml.update(loss=loss.item(), lr=lr)
        stats = ml.get_stats()   # {"loss": 0.31, "lr": 1e-3}

    数学背景：EMA（指数移动平均）
    -------------------------------
        ema_k = beta * ema_{k-1} + (1 - beta) * value_k

    等价于给第 i 个样本权重 (1-beta)*beta^{n-i}，越久远的样本指数级衰减。
    ``beta=0.98`` 时，半衰期约为 ``-1 / ln(beta) ≈ 50`` 个样本。
    """

    def __init__(self, beta: float = 0.98) -> None:
        self.beta = beta
        self._ema: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._sums: dict[str, float] = {}  # 原始累加和（用于精确平均）

    def update(self, **values: float | Tensor) -> None:
        """更新一批指标。

        Args:
            **values: 指标名 → 数值。例如 ``update(loss=0.3, acc=0.9)``。
        """
        for key, val in values.items():
            v = _as_float(val)
            if key not in self._ema:
                self._ema[key] = v
            else:
                # 指数加权：当前值只贡献 (1-beta) 的权重
                self._ema[key] = self.beta * self._ema[key] + (1.0 - self.beta) * v
            self._sums[key] = self._sums.get(key, 0.0) + v
            self._counts[key] = self._counts.get(key, 0) + 1

    def get_stats(self) -> dict[str, float]:
        """返回当前所有指标的滑动平均。"""
        return dict(self._ema)

    def get_precise(self) -> dict[str, float]:
        """返回精确平均（除以真实样本数）——epoch 结束时用来汇报最终值。"""
        return {
            k: self._sums[k] / self._counts[k]
            for k in self._sums
        }

    def reset(self) -> None:
        """清空所有统计状态。"""
        self._ema.clear()
        self._counts.clear()
        self._sums.clear()


# TensorBoard 支持的懒加载封装（避免无 tensorboard 环境时 import 失败）
class TBLogger:
    """TensorBoard 写日志的薄封装。

    - 每个 epoch 记录 loss / lr / 验证指标（scalar）
    - 提供 ``add_histogram`` / ``add_image`` 的透传
    """

    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(str(self.log_dir))
        except Exception as e:  # noqa: BLE001
            logging.getLogger(__name__).warning(
                f"TensorBoard unavailable ({e}); metrics will only go to console."
            )

    def add_scalars(self, tag: str, values: dict[str, float], step: int) -> None:
        if self._writer is None:
            return
        for k, v in values.items():
            self._writer.add_scalar(f"{tag}/{k}", v, step)

    def add_histogram(self, tag: str, values: Tensor, step: int) -> None:
        if self._writer is None:
            return
        self._writer.add_histogram(tag, values.cpu().detach().numpy(), step)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
