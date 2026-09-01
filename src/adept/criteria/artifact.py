"""Criterion 5 — Freedom from artifact: folds, air bubbles, crush, desiccation.

Two implementations share one interface:

* :class:`HeuristicArtifactDetector` — label-free rules (very dark, saturated
  ridges → folds; bright, circular, sharp-edged voids inside tissue → bubbles).
  Good enough for the Minimum endpoint's single-type detector and as a
  baseline for the trained model.
* A trained patch classifier (``adept.models.artifact``, torch) exposing the
  same ``predict_mask``. S1-ID-2 Target endpoint trains it on the labelled
  fold / bubble / crush / desiccation patch set.

Mask classes: 0 clean · 1 fold · 2 bubble · 3 crush · 4 desiccation.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label, regionprops

from adept.calibration.od import as_float_rgb
from adept.schema import CriterionResult, ReasonCode

ARTIFACT_CLASSES = {0: "clean", 1: "fold", 2: "bubble", 3: "crush", 4: "desiccation"}
_REASON_FOR_CLASS = {
    1: ReasonCode.ARTIFACT_FOLD,
    2: ReasonCode.ARTIFACT_BUBBLE,
    3: ReasonCode.ARTIFACT_CRUSH,
    4: ReasonCode.ARTIFACT_DESICCATION,
}


class ArtifactDetector(Protocol):
    def predict_mask(self, rgb: np.ndarray) -> np.ndarray:  # (H, W) int, classes above
        ...


class HeuristicArtifactDetector:
    """Rule-based fold and bubble finder (classes 1 and 2 only)."""

    def __init__(self, fold_darkness: float = 0.25, bubble_brightness: float = 0.92):
        self.fold_darkness = fold_darkness
        self.bubble_brightness = bubble_brightness

    def predict_mask(self, rgb: np.ndarray) -> np.ndarray:
        rgb = as_float_rgb(rgb)
        lum = rgb.mean(axis=-1)
        out = np.zeros(lum.shape, np.int8)
        # folds: very dark, elongated components
        dark = ndi.binary_opening(lum < self.fold_darkness, structure=np.ones((3, 3)))
        lab = label(dark)
        for rp in regionprops(lab):
            if rp.area >= 40 and rp.eccentricity > 0.9:
                out[lab == rp.label] = 1
        # bubbles: bright, roundish holes with a sharp dark rim
        bright = lum > self.bubble_brightness
        rim = ndi.binary_dilation(bright, iterations=2) & (lum < 0.5)
        lab = label(bright)
        for rp in regionprops(lab):
            if rp.area < 30 or rp.eccentricity > 0.8:
                continue
            ring = ndi.binary_dilation(lab == rp.label, iterations=2) & ~(lab == rp.label)
            if ring.any() and (rim[ring].mean() > 0.3):
                out[lab == rp.label] = 2
        return out


def assess_artifact(
    corrected_rgb: np.ndarray,
    tissue_mask: np.ndarray,
    cfg,
    detector: ArtifactDetector | None = None,
) -> CriterionResult:
    detector = detector or HeuristicArtifactDetector()
    mask = detector.predict_mask(corrected_rgb)
    # only artifacts that touch tissue matter for adequacy
    relevant = ndi.binary_dilation(tissue_mask, iterations=3)
    tissue_px = max(int(tissue_mask.sum()), 1)
    per_class = {}
    for k, name in ARTIFACT_CLASSES.items():
        if k == 0:
            continue
        per_class[name] = float(((mask == k) & relevant).sum() / tissue_px)
    frac = float(sum(per_class.values()))
    reasons = [
        _REASON_FOR_CLASS[k] for k in _REASON_FOR_CLASS if per_class[ARTIFACT_CLASSES[k]] > 0.05
    ]
    score = float(np.clip(1.0 - frac / max(cfg.artifact_max_fraction, 1e-6), 0, 1))
    return CriterionResult(
        name="artifact",
        value=frac,
        unit="fraction_tissue_affected",
        score=score,
        passed=bool(frac <= cfg.artifact_max_fraction),
        threshold=cfg.artifact_max_fraction,
        details={
            "per_class_fraction": per_class,
            "reasons": [r.value for r in reasons],
            "detector": type(detector).__name__,
        },
    )
