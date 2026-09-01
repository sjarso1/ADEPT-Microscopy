"""CNN adequacy classifier arms (S1-ID-3 architecture experiment). Requires torch.

Two arms share one interface so the experiment is a config switch:

* ``pretrained=True``  — EfficientNet-B0 (timm, ImageNet weights) fine-tuned
* ``pretrained=False`` — same topology trained from scratch

Input: canonical RGB tiles (3, H, W) in [0, 1]. Output: logit of P(inadequate).
The five criterion scores may be concatenated to the pooled features
(``use_criteria=True``) so the CNN can be ablated against the transparent
combiner on equal footing.
"""

from __future__ import annotations

try:
    import timm
    import torch
    from torch import nn
except ImportError as e:  # pragma: no cover
    raise ImportError("adept.models.adequacy_cnn needs the [torch] extra") from e


class AdequacyCNN(nn.Module):
    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        use_criteria: bool = False,
        n_criteria: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feat = self.backbone.num_features
        self.use_criteria = use_criteria
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(feat + (n_criteria if use_criteria else 0), 1)
        )

    def forward(self, x: torch.Tensor, criteria: torch.Tensor | None = None) -> torch.Tensor:
        f = self.backbone(x)
        if self.use_criteria:
            if criteria is None:
                raise ValueError("model was built with use_criteria=True")
            f = torch.cat([f, criteria], dim=1)
        return self.head(f).squeeze(1)


def sensitivity_weighted_bce(pos_weight: float = 3.0):
    """BCE with a positive-class (inadequate) weight — the loss-side expression
    of 'sensitivity dominates'."""
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
