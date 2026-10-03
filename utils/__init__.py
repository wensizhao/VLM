# 这个文件是 utils 包的统一导出入口。
"""Logging, visualization, IO, and reproducibility utilities.

按功能分成四个子模块：
- ``reproducibility`` —— set_seed 等复现性工具
- ``logging``         —— get_logger / MetricLogger / TBLogger
- ``io``              —— save_checkpoint / load_checkpoint / save_json ...
- ``visualization``   —— 时序热力图 / attention / prompt 相似度绘图
"""

from .reproducibility import set_env_reproducible, set_seed
from .logging import MetricLogger, TBLogger, get_logger
from .io import (
    ensure_dirs,
    load_checkpoint,
    load_json,
    save_checkpoint,
    save_json,
)
from .visualization import (
    plot_frame_attention,
    plot_prompt_scores,
    plot_temporal_heatmap,
)
from .config import load_experiment_config

__all__ = [
    # reproducibility
    "set_seed",
    "set_env_reproducible",
    # logging
    "get_logger",
    "MetricLogger",
    "TBLogger",
    # io
    "ensure_dirs",
    "save_checkpoint",
    "load_checkpoint",
    "save_json",
    "load_json",
    # visualization
    "plot_temporal_heatmap",
    "plot_prompt_scores",
    "plot_frame_attention",
    # config
    "load_experiment_config",
]
