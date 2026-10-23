"""A fixed normal-feature memory baseline with the project's output contract."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class NormalityMemory(nn.Module):
    """Mean k-nearest cosine distance to a training-only memory bank.

    Input: visual features (B,T,D); bank: (M,D); scores: (B,T).
    Larger distance means less similar to normal reference frames, not a calibrated
    anomaly probability. It is a nonparametric baseline; no classifier is trained.
    """

    def __init__(self, bank: torch.Tensor, k: int = 5, query_chunk_size: int = 256):
        super().__init__()
        if bank.ndim != 2 or not len(bank) or not torch.isfinite(bank).all() or (bank.norm(dim=-1) <= 0).any():
            raise ValueError("Memory bank must be finite nonzero (M,D) features")
        if not 1 <= k <= len(bank) or query_chunk_size <= 0:
            raise ValueError("k must fit the bank and query_chunk_size must be positive")
        self.register_buffer("bank", F.normalize(bank.float(), dim=-1))
        self.k = k
        self.query_chunk_size = query_chunk_size

    @classmethod
    def fit(cls, features: torch.Tensor, max_bank_frames: int, seed: int, **kwargs):
        if max_bank_frames <= 0:
            raise ValueError("max_bank_frames must be positive")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        indices = torch.randperm(len(features), generator=generator)[:max_bank_frames]
        return cls(features[indices.to(features.device)].clone(), **kwargs), indices

    @torch.no_grad()
    def forward(self, visual: torch.Tensor) -> dict:
        if visual.ndim != 3 or not visual.shape[0] or not visual.shape[1] or visual.shape[-1] != self.bank.shape[-1]:
            raise ValueError("Expected visual (B,T,D) matching memory dimension")
        if not torch.isfinite(visual).all() or (visual.norm(dim=-1) <= 0).any():
            raise ValueError("Visual features must be finite and nonzero")
        query = F.normalize(visual.float(), dim=-1).reshape(-1, visual.shape[-1])
        distances = []
        for chunk in query.split(self.query_chunk_size):
            similarities = chunk @ self.bank.T
            nearest = similarities.topk(self.k, dim=-1).values
            distances.append((1 - nearest).clamp_min(0).mean(dim=-1))
        scores = torch.cat(distances).reshape(visual.shape[:2])
        return {"anomaly_score": scores.amax(dim=1), "frame_score": scores,
                "embedding": F.normalize(visual.mean(dim=1), dim=-1), "explanation": None}
