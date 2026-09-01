"""Four-class artifact patch classifier with mask output (Criterion 5, trained arm).

Requires torch. Implements the :class:`adept.criteria.artifact.ArtifactDetector`
protocol: ``predict_mask(rgb) -> (H, W) int`` with classes
0 clean · 1 fold · 2 bubble · 3 crush · 4 desiccation.

The network is a small encoder over ``patch_px`` patches with stride
``stride_px``; per-patch class logits are painted back into a mask. This keeps
the labelling cost at patch level (what the annotation workstream produces)
while still giving a spatial mask for ROI planning.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as e:  # pragma: no cover
    raise ImportError("adept.models.artifact_cnn needs the [torch] extra") from e

from adept.calibration.od import as_float_rgb

N_CLASSES = 5


class ArtifactPatchNet(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES, width: int = 32):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2)
            )

        self.net = nn.Sequential(
            block(3, width),
            block(width, width * 2),
            block(width * 2, width * 4),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width * 4, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class TorchArtifactDetector:
    def __init__(
        self, model: ArtifactPatchNet, patch_px: int = 32, stride_px: int = 16, device="cpu"
    ):
        self.model = model.to(device).eval()
        self.patch_px, self.stride_px, self.device = patch_px, stride_px, device

    @torch.no_grad()
    def predict_mask(self, rgb: np.ndarray) -> np.ndarray:
        rgb = as_float_rgb(rgb)
        H, W = rgb.shape[:2]
        p, s = self.patch_px, self.stride_px
        votes = np.zeros((N_CLASSES, H, W), np.float32)
        coords, patches = [], []
        for y in range(0, max(H - p, 0) + 1, s):
            for x in range(0, max(W - p, 0) + 1, s):
                patches.append(rgb[y : y + p, x : x + p])
                coords.append((y, x))
        if not patches:
            return np.zeros((H, W), np.int8)
        xb = torch.from_numpy(np.stack(patches)).permute(0, 3, 1, 2).to(self.device)
        probs = torch.softmax(self.model(xb), dim=1).cpu().numpy()
        for (y, x), pr in zip(coords, probs, strict=True):
            votes[:, y : y + p, x : x + p] += pr[:, None, None]
        return votes.argmax(axis=0).astype(np.int8)
