# 这个文件负责配置加载——把 experiment.yaml 与 prompts.yaml 合并成一份完整配置。
"""Config loading — merge experiment config with the prompt config.

为什么拆成两个 yaml？
---------------------
- ``experiment.yaml``：训练/模型/数据/路径（跟具体 prompt 无关）
- ``prompts.yaml``：   prompt 模板与极性关键词（VAD 语义词汇表）

两者合并后，``cfg.prompt`` 段里既有模板又有极性关键词，模型构造和
loss 构造都能从同一份配置取数——保证"模型看到什么 prompt、loss 就对什么
prompt 打极性"完全一致，前后闭环。
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_experiment_config(path: str | Path) -> DictConfig:
    """加载实验配置并内联 prompt 配置。

    Args:
        path: experiment.yaml 路径。

    Returns:
        合并后的 DictConfig（``cfg.prompt`` 已包含 prompts.yaml 的内容）。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")

    cfg = OmegaConf.load(str(path))

    # 内联 prompt 配置（若配置了 config_path）
    prompt_cfg_path = cfg.prompt.get("config_path", None)
    if prompt_cfg_path:
        p = Path(prompt_cfg_path)
        # 解析顺序：1) 直接按给定路径（相对 cwd）2) 相对 config 文件所在目录
        if not p.is_file() and not p.is_absolute():
            alt = path.parent / p
            if alt.is_file():
                p = alt
        if not p.is_file():
            raise FileNotFoundError(f"Prompt config not found: {p}")
        prompt_cfg = OmegaConf.load(str(p))
        # 合并：prompts.yaml 的内容覆盖 experiment.yaml 里 prompt 段的同名键
        cfg.prompt = OmegaConf.merge(cfg.prompt, prompt_cfg)

    # 解析设备
    device_str = str(cfg.get("device", "auto")).lower()
    if device_str == "auto":
        import torch
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


def save_config_snapshot(cfg: DictConfig, path: str | Path) -> Path:
    """把合并后的完整配置落盘到 results/，保证实验可复现（AGENTS.md §7）。

    Args:
        cfg: load_experiment_config() 返回的完整配置。
        path: 输出路径（通常 ``{result_dir}/{run_name}/config.yaml``）。

    Returns:
        实际写入的 Path。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, str(path))
    return path
