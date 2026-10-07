"""
Combines title search and photo search for one text query.

Each side's similarities are z-normalized per query (subtract the mean, divide
by the standard deviation across the gallery) so the two models' score scales
are comparable, then blended. A product that one side matches confidently
stands far above that side's average, so confidence carries into the blend.
Rank-based fusion (RRF) throws that away; see IMPROVEMENTS.md.

PHOTO_WEIGHT was tuned with tune_fusion.py on held-out products.
"""
from __future__ import annotations

import torch

PHOTO_WEIGHT = 0.7


def zscore(s: torch.Tensor) -> torch.Tensor:
    return (s - s.mean(-1, keepdim=True)) / s.std(-1, keepdim=True)


def blend(title_scores: torch.Tensor, photo_scores: torch.Tensor, w: float = PHOTO_WEIGHT) -> torch.Tensor:
    """Blended score for every gallery product; works on one query or a batch."""
    return w * zscore(photo_scores) + (1 - w) * zscore(title_scores)
