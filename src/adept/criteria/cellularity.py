"""Criterion 2 — Cellularity: nucleus density per unit tissue area.

The default detector is a classical, label-free nucleus finder on the canonical
haematoxylin channel (Gaussian smoothing → local maxima above an adaptive
threshold). It is deliberately simple and instrument-invariant by
construction, because it consumes canonical concentrations only. A pretrained
deep nucleus model (e.g. StarDist / HoVer-Net) can be plugged in through the
``detector`` argument; S1-ID-2 (Maximum endpoint) compares the two.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

from adept.schema import CriterionResult

NucleusDetector = Callable[[np.ndarray], np.ndarray]
"""Callable taking (H, W, 2) canonical concentrations and returning an (N, 2)
array of nucleus centroids (row, col)."""


def detect_nuclei_classical(
    concentrations: np.ndarray,
    sigma: float = 1.5,
    min_distance_px: int = 4,
    h_threshold: float = 0.35,
) -> np.ndarray:
    """Local-maxima nucleus detector on the haematoxylin channel."""
    h = ndi.gaussian_filter(concentrations[..., 0], sigma)
    if h.max() <= 0:
        return np.zeros((0, 2), dtype=int)
    thr = max(h_threshold, float(np.percentile(h[h > 0], 60))) if np.any(h > 0) else h_threshold
    peaks = peak_local_max(h, min_distance=min_distance_px, threshold_abs=thr, exclude_border=False)
    return peaks.astype(int)


def nuclei_per_mm2(n_nuclei: int, tissue_mask: np.ndarray, microns_per_pixel: float) -> float:
    area_mm2 = tissue_mask.sum() * (microns_per_pixel / 1000.0) ** 2
    return float(n_nuclei / area_mm2) if area_mm2 > 0 else 0.0


def assess_cellularity(
    concentrations: np.ndarray,
    tissue_mask: np.ndarray,
    cfg,
    detector: NucleusDetector | None = None,
) -> CriterionResult:
    detector = detector or detect_nuclei_classical
    pts = detector(concentrations)
    if len(pts):
        inside = tissue_mask[pts[:, 0], pts[:, 1]]
        pts = pts[inside]
    density = nuclei_per_mm2(len(pts), tissue_mask, cfg.microns_per_pixel)
    score = float(np.clip(density / max(cfg.cellularity_min_nuclei_per_mm2, 1e-6), 0, 1))
    return CriterionResult(
        name="cellularity",
        value=density,
        unit="nuclei_per_mm2",
        score=score,
        passed=bool(density >= cfg.cellularity_min_nuclei_per_mm2),
        threshold=cfg.cellularity_min_nuclei_per_mm2,
        details={"n_nuclei": int(len(pts)), "detector": getattr(detector, "__name__", "custom")},
    )
