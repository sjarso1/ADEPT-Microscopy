"""Attention-based multiple-instance learning for malignancy-suspicion triage (S3-ID-2).

Requires torch. Slide-level labels, tile-level evidence:

    tile embeddings  e_i ∈ R^d   (from the S2 backbone or a pathology foundation encoder)
    attention        a_i = softmax_i( wᵀ tanh(V e_i) ⊙ sigmoid(U e_i) )
                     (gated attention, Ilse et al. 2018)
    slide embedding  z   = Σ_i a_i e_i
    slide score      P(malignant) = σ(W z)

The attention weights ``a_i`` *are* the per-tile suspicion ranking that feeds
:func:`adept.models.roi.rank_rois` — and the quantity whose concordance with
pathologist-marked regions (S3-ID-3) is measured. ``TopKMIL`` is the
aggregation-experiment alternative.
"""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as e:  # pragma: no cover
    raise ImportError("adept.models.mil needs the [torch] extra") from e


class AttentionMIL(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int = 256,
        n_classes: int = 1,
        gated: bool = True,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.gated = gated
        self.V = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Dropout(dropout))
        self.U = (
            nn.Sequential(nn.Linear(in_dim, hidden), nn.Sigmoid(), nn.Dropout(dropout))
            if gated
            else None
        )
        self.w = nn.Linear(hidden, 1)
        self.classifier = nn.Linear(in_dim, n_classes)

    def forward(self, tiles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """tiles: (N, in_dim) for one slide → (logits (n_classes,), attention (N,))"""
        h = self.V(tiles)
        if self.gated:
            h = h * self.U(tiles)
        a = torch.softmax(self.w(h).squeeze(-1), dim=0)
        z = (a.unsqueeze(-1) * tiles).sum(0)
        return self.classifier(z), a


class TopKMIL(nn.Module):
    """Mean of the top-k tile logits: the aggregation-experiment baseline."""

    def __init__(self, in_dim: int, k: int = 8, n_classes: int = 1):
        super().__init__()
        self.k = k
        self.tile_classifier = nn.Linear(in_dim, n_classes)

    def forward(self, tiles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.tile_classifier(tiles)  # (N, C)
        k = min(self.k, logits.shape[0])
        top, idx = torch.topk(logits[:, 0], k)
        attn = torch.zeros(logits.shape[0], device=tiles.device)
        attn[idx] = 1.0 / k
        return logits[idx].mean(0), attn
