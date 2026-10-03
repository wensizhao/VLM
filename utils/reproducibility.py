# 这个文件负责实验的可复现性（Reproducibility）——固定所有随机源。
"""Reproducibility utilities — seed every random source so experiments are repeatable.

为什么需要固定随机种子？
========================
深度学习训练中有多个随机源（random source）：

    1. Python 的 ``random``        —— 某些数据增强 / 采样逻辑
    2. NumPy 的 ``np.random``      —— 数据处理中的随机操作
    3. PyTorch 的 ``torch.manual_seed`` —— 权重初始化、dropout、数据 shuffle
    4. CUDA 的 cuDNN               —— 卷积的实现选择本身是"随机"的

如果不固定这些种子，同样的配置每次运行结果都不同，无法对比两个实验的优劣——
你分不清指标差异来自模型改进还是来自随机波动（random fluctuation）。
固定种子后，同一个 config 在任何机器上跑出相同结果 → 可复现。

数学背景：为什么 cuDNN 是"随机"的？
-----------------------------------
cuDNN 对同一个卷积有多种底层实现（implicit GEMM、winograd、FFT...），
每种在数值精度和速度上不同。``torch.backends.cudnn.benchmark = False``
告诉 PyTorch 每次用同一套确定性算法，否则它会根据输入尺寸动态选最快的实现，
导致结果在不同次运行间有微小但可观测的差异。
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """固定所有随机源，保证实验可复现。

    Args:
        seed: 全局随机种子（通常是任意整数，如 42）。
    """
    # 1) Python 内置 random —— 影响 random.shuffle / random.choice 等
    random.seed(seed)

    # 2) NumPy —— 影响 np.random 系列
    np.random.seed(seed)

    # 3) PyTorch 主随机数生成器（CPU 权重初始化、dropout、sampler 等）
    torch.manual_seed(seed)

    # 4) CUDA 随机源（如果可用）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 5) 让 GPU 上的内核选择变成确定性的
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 6) cuBLAS 的 workspace 配置必须在第一次 CUDA 矩阵乘之前固定，
    #    否则 use_deterministic_algorithms(True) 会在 CUDA >= 10.2 上直接报错：
    #    "Deterministic behavior was enabled ... but this operation uses CuBLAS".
    #    使用 setdefault 避免覆盖用户显式设置的环境变量。
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # 7) 有些算子（如某些 reduce 操作）在非确定性模式下更快，
    #    关闭它以换取可复现性。注意：某些算子（如 ATen 的部分稀疏 op）
    #    不支持 deterministic 模式，如果遇到报错可以关闭这一行。
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:  # noqa: BLE001
        # 个别环境/算子不支持 deterministic，降级为普通模式并告警
        import warnings
        warnings.warn(f"Deterministic algorithms unavailable ({e}); "
                      "falling back to non-deterministic mode.")
        torch.use_deterministic_algorithms(False, warn_only=True)


def set_env_reproducible(seed: int) -> None:
    """在固定随机源之外，同时固定一些环境变量（容器/分布式场景下更严格）。

    ``PYTHONHASHSEED`` 影响 dict / set 的迭代顺序。Python 默认对字符串
    hash 加盐（salted），跨进程不可复现。在容器启动脚本里固定它可以避免
    一个极难排查的复现性 bug。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # 让 cuBLAS 有界工作区，保证确定性
    set_seed(seed)
