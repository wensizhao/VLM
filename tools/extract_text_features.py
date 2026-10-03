#!/usr/bin/env python3
# 这个脚本离线提取所有 prompt 的 CLIP 文本特征——一次 encode_text，后续训练复用。
"""Extract CLIP text features from PromptProcessor config.

文本 prompt 在整个训练中通常不变，一次计算永久复用::

    python tools/extract_text_features.py \\
        --config ./configs/prompts.yaml \\
        --model ViT-B-32 \\
        --out ./data/features/text/prompts.pt

Output: ``prompts.pt`` — a dict ``{"features": (K, L, D), "prompts": [str, ...]}``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.backbone import CLIPBackbone
from prompts.processor import PromptProcessor


# ═════════════════════════════════════════════════════════════════
# YAML config support (optional — if PyYAML is available)
# ═════════════════════════════════════════════════════════════════

def _load_prompt_config(config_path: str | Path | None) -> dict:
    """从 YAML/JSON 文件或默认值加载 PromptProcessor 配置。"""
    if config_path is None:
        return _default_config()

    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Prompt config not found: {config_path}")

    if config_path.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML config. pip install pyyaml")
        with open(config_path) as f:
            return yaml.safe_load(f)
    elif config_path.suffix == ".json":
        import json
        with open(config_path) as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")


def _default_config() -> dict:
    return {
        "templates": {
            "label": "a person running",
            "scene": "a surveillance scene with abnormal behavior",
            "contrast": "normal vs abnormal behavior comparison",
        },
    }


# ═════════════════════════════════════════════════════════════════
# Main extraction
# ═════════════════════════════════════════════════════════════════

def extract(
    config_path: str | Path | None = None,
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    out_path: str | Path = "./data/features/text/prompts.pt",
    device: str = "cuda",
    expand: bool = True,
    hard_negatives: bool = False,
) -> None:
    """一次 encode_text，存为单个 .pt 文件。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. 加载 backbone
    print(f"[1/3] Loading backbone: {model_name}")
    backbone = CLIPBackbone(model_name=model_name, pretrained=pretrained)
    backbone.eval()
    backbone.to(device)

    # 2. 构建 prompts
    config = _load_prompt_config(config_path)
    pp = PromptProcessor(
        templates=config.get("templates"),
        expansions=config.get("expansions"),
        hard_negatives=config.get("hard_negatives"),
    )
    prompts = pp.process(expand=expand, hard_negatives=hard_negatives)
    print(f"[2/3] Processing {len(prompts)} prompts "
          f"(expand={expand}, hard_negatives={hard_negatives})")

    # 3. Encode text
    print(f"[3/3] Encoding text features...")
    with torch.no_grad():
        features = backbone.encode_text(prompts)  # (K, L, D)
        features = features.cpu()

    # 保存
    data = {
        "features": features,
        "prompts": prompts,
        "model_name": model_name,
        "shape": tuple(features.shape),
    }
    torch.save(data, out_path)
    print(f"Done! Saved {tuple(features.shape)} to {out_path}")
    print(f"  K={features.shape[0]} prompts, L={features.shape[1]} tokens, "
          f"D={features.shape[2]}")


# ═════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract CLIP text features for all prompts",
    )
    parser.add_argument("--config", default=None, help="Path to prompts YAML/JSON config")
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--out", default="./data/features/text/prompts.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-expand", action="store_true",
                        help="Disable placeholder expansion")
    parser.add_argument("--hard-negatives", action="store_true",
                        help="Include hard negative prompts")
    args = parser.parse_args()

    extract(
        config_path=args.config,
        model_name=args.model,
        pretrained=args.pretrained,
        out_path=args.out,
        device=args.device,
        expand=not args.no_expand,
        hard_negatives=args.hard_negatives,
    )


if __name__ == "__main__":
    main()
