"""Criterion 4 — Focus quality: tile-wise sharpness map, calibrated per instrument.

Sharpness is the **contrast-normalised Laplacian variance** of the
haematoxylin concentration image, ``var(∇²h) / var(h)``, computed per tile.
Dividing by the tile's own variance makes the metric insensitive to how much
stain is present, so a sparsely-stained tile and a dense one are judged on
edge sharpness alone. Because the absolute value still depends on optics,
camera and magnification, the map is expressed *relative to the instrument's
best achievable sharpness*, measured on the resolution target during
calibration (``focus_reference``). That is what makes
the criterion instrument-invariant: 0.5 means "half as sharp as this
instrument can be", on every instrument.

The tile map is also the signal that drives the closed loop: tiles below the
threshold become ``refocus`` / ``rescan`` ROIs.
"""

from __future__ import annotations

import cv2
import numpy as np

from adept.schema import CriterionResult


def laplacian_variance(img: np.ndarray) -> float:
    """Plain variance of the Laplacian of a 2-D (or RGB→grey) image."""
    if img.ndim == 3:
        img = img.mean(axis=-1)
    img = img.astype(np.float32)
    return float(cv2.Laplacian(img, cv2.CV_32F, ksize=3).var())


def normalized_sharpness(img: np.ndarray, eps: float = 1e-4) -> float:
    """Contrast-normalised sharpness ``var(∇²img) / (var(img) + eps)`` of a 2-D image."""
    if img.ndim == 3:
        img = img.mean(axis=-1)
    img = img.astype(np.float32)
    v = float(img.var())
    if v < eps:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_32F, ksize=3).var() / (v + eps))


def sharpness_map(channel: np.ndarray, tile_px: int = 64) -> np.ndarray:
    """(nH, nW) contrast-normalised sharpness per non-overlapping tile."""
    H, W = channel.shape
    nH, nW = max(H // tile_px, 1), max(W // tile_px, 1)
    out = np.zeros((nH, nW), np.float32)
    for i in range(nH):
        for j in range(nW):
            out[i, j] = normalized_sharpness(
                channel[i * tile_px : (i + 1) * tile_px, j * tile_px : (j + 1) * tile_px]
            )
    return out


def tile_tissue_fraction(mask: np.ndarray, tile_px: int) -> np.ndarray:
    H, W = mask.shape
    nH, nW = max(H // tile_px, 1), max(W // tile_px, 1)
    out = np.zeros((nH, nW), np.float32)
    for i in range(nH):
        for j in range(nW):
            out[i, j] = mask[
                i * tile_px : (i + 1) * tile_px, j * tile_px : (j + 1) * tile_px
            ].mean()
    return out


def assess_focus(
    concentrations: np.ndarray,
    tissue_mask: np.ndarray,
    cfg,
    focus_reference: float | None = None,
) -> CriterionResult:
    h = concentrations[..., 0]
    smap = sharpness_map(h, cfg.focus_tile_px)
    tfrac = tile_tissue_fraction(tissue_mask, cfg.focus_tile_px)
    tissue_tiles = tfrac > 0.2
    if focus_reference is None or focus_reference <= 0:
        # absolute fallback: relative to the sharpest tissue tile in this image
        ref = float(smap[tissue_tiles].max()) if tissue_tiles.any() else 1.0
        ref_kind = "self_max"
    else:
        ref, ref_kind = focus_reference, "instrument_reference"
    rel = np.clip(smap / max(ref, 1e-9), 0, 1)
    in_focus = rel >= cfg.focus_min_relative_sharpness
    frac_in_focus = float(in_focus[tissue_tiles].mean()) if tissue_tiles.any() else 0.0
    score = float(np.clip(frac_in_focus / max(cfg.focus_min_fraction_in_focus, 1e-6), 0, 1))
    return CriterionResult(
        name="focus",
        value=frac_in_focus,
        unit="fraction_tissue_tiles_in_focus",
        score=score,
        passed=bool(frac_in_focus >= cfg.focus_min_fraction_in_focus),
        threshold=cfg.focus_min_fraction_in_focus,
        details={
            "tile_px": cfg.focus_tile_px,
            "reference": ref,
            "reference_kind": ref_kind,
            "relative_sharpness_map": rel.round(3).tolist(),
            "tissue_tile_map": tissue_tiles.tolist(),
        },
    )
