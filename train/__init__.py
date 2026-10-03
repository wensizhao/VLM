# 这个文件是 train 包的统一导出入口。
"""Training: losses and the full trainer loop.

- ``losses.py``  —— VLMVADLoss（AnomalyBCE + ContrastivePolarity）
- ``trainer.py`` —— Trainer（optimizer / scheduler / AMP / checkpoint）
"""

from .losses import (
    AnomalyBCELoss,
    ContrastivePolarityLoss,
    VLMVADLoss,
    build_prompt_polarity,
)
from .trainer import Trainer, build_optimizer, build_scheduler, resume_from_checkpoint

__all__ = [
    # losses
    "AnomalyBCELoss",
    "ContrastivePolarityLoss",
    "VLMVADLoss",
    "build_prompt_polarity",
    # trainer
    "Trainer",
    "build_optimizer",
    "build_scheduler",
    "resume_from_checkpoint",
]
