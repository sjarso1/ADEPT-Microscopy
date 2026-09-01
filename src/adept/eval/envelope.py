"""Generalisation-envelope characterisation (S2-ID-2).

Controlled perturbations along three stress axes, applied to *canonical-space
inputs or raw inputs* as the experiment requires, and a sweep that increases
the stress until a diagnostic target is crossed.

Axes
----
stain         hue/intensity: rotate the stain vectors by θ degrees and scale
              concentrations by g (θ, g are the stress parameters)
focus         Gaussian blur σ in pixels, calibrated against the focus metric
illumination  multiplicative vignetting strength v and white-balance tilt

Every perturbation is a pure function ``perturb(img, axis, level, rng)``; the
sweep takes a *scoring callback* so the same harness works for any model arm.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from adept.calibration.od import as_float_rgb, od_to_rgb, rgb_to_od
from adept.calibration.stain import deconvolve, reconstruct_od
from adept.eval.metrics import TARGETS_ADEQUACY, DiagnosticMetrics, diagnostic_metrics

AXES = ("stain", "focus", "illumination")


def _rotate_stains(M: np.ndarray, deg: float) -> np.ndarray:
    th = np.radians(deg)
    h, e = M[0], M[1]
    e_perp = e - h * np.dot(e, h)
    e_perp /= np.linalg.norm(e_perp) + 1e-9
    h2 = np.cos(th) * h + np.sin(th) * e_perp
    M2 = np.stack([h2, e])
    M2 = np.clip(M2, 0, None)
    return M2 / np.linalg.norm(M2, axis=1, keepdims=True)


def perturb(
    img: np.ndarray,
    axis: str,
    level: float,
    rng: np.random.Generator | None = None,
    stain_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a perturbation of strength ``level`` along ``axis`` to an RGB image.

    stain         level = stain-vector rotation in degrees; intensity is scaled
                  by (1 + level/45) to couple hue and intensity stress
    focus         level = Gaussian blur σ (px)
    illumination  level ∈ [0, 1] = vignetting strength (+ a mild colour tilt)
    """
    rgb = as_float_rgb(img)
    if level <= 0:
        return rgb
    if axis == "stain":
        from adept.calibration.canonical import CANONICAL_STAIN_MATRIX

        M = CANONICAL_STAIN_MATRIX if stain_matrix is None else stain_matrix
        C = deconvolve(rgb_to_od(rgb), M) * (1.0 + level / 45.0)
        return od_to_rgb(reconstruct_od(C, _rotate_stains(M, level)))
    if axis == "focus":
        return cv2.GaussianBlur(rgb, (0, 0), float(level))
    if axis == "illumination":
        H, W = rgb.shape[:2]
        yy, xx = np.mgrid[0:H, 0:W]
        r2 = ((yy - H / 2) / (H / 2)) ** 2 + ((xx - W / 2) / (W / 2)) ** 2
        f = 1.0 - level * r2 / 2.0
        tilt = np.array([1.0, 1.0 - 0.1 * level, 1.0 - 0.2 * level], np.float32)
        return np.clip(rgb * f[..., None] * tilt[None, None, :], 0, 1).astype(np.float32)
    raise ValueError(f"unknown axis {axis!r}; choose from {AXES}")


@dataclass
class EnvelopeResult:
    axis: str
    levels: list[float]
    metrics: list[DiagnosticMetrics]
    boundary_level: float | None  # first level at which any target is crossed
    boundary_metric: str | None


def sweep_envelope(
    axis: str,
    levels: Sequence[float],
    score_fn: Callable[[float], tuple[np.ndarray, np.ndarray]],
    threshold: float = 0.5,
    targets: dict[str, float] | None = None,
    n_boot: int = 500,
    seed: int = 0,
) -> EnvelopeResult:
    """Increase stress along ``axis`` and record where a target is crossed.

    ``score_fn(level) -> (y_true, y_score)`` runs the model arm on the test set
    perturbed at that level. The boundary is the first level whose *point
    estimate* misses a target; the report should also show the CI band.
    """
    targets = targets or TARGETS_ADEQUACY
    out_metrics, boundary, bmetric = [], None, None
    for lv in levels:
        y_true, y_score = score_fn(float(lv))
        m = diagnostic_metrics(y_true, y_score, threshold, n_boot, seed)
        out_metrics.append(m)
        if boundary is None:
            for k, v in targets.items():
                if (
                    hasattr(m, k)
                    and np.isfinite(getattr(m, k).estimate)
                    and getattr(m, k).estimate < v
                ):
                    boundary, bmetric = float(lv), k
                    break
    return EnvelopeResult(axis, [float(x) for x in levels], out_metrics, boundary, bmetric)
