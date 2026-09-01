"""Illumination flat-field derivation and correction.

The flat-field captures the instrument's spatial illumination non-uniformity
(vignetting, LED hot-spot, condenser misalignment) from images of a **blank**
slide. It is stored per instrument and applied *before* the OD transform.

    corrected = raw / flat

The flat is kept in **absolute** units (it *is* the background intensity
``I0(x, y)`` of the Beer–Lambert law), so a corrected blank slide is ≈ 1.0
everywhere and the corrected image is directly the transmittance ``I / I0``.
White balance is therefore absorbed into the flat; the separate spectral gain
in :class:`InstrumentCalibration` only matters when no flat is available.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter

from adept.calibration.od import as_float_rgb


def derive_flatfield(
    blank_images: Sequence[np.ndarray],
    smooth_sigma: float = 5.0,
    dark: np.ndarray | None = None,
    normalise: bool = False,
) -> np.ndarray:
    """Estimate the illumination flat-field from one or more blank-slide images.

    Parameters
    ----------
    blank_images : images of a blank (glass + coverslip, no tissue) slide, all the
        same shape. Several frames at different stage positions average out dust.
    smooth_sigma : Gaussian smoothing (pixels) applied to suppress dust and sensor
        noise while preserving the low-frequency vignetting profile.
    dark : optional dark frame (shutter closed / LED off), subtracted first.
    normalise : if True, rescale so the per-channel median is 1 (shape only);
        default False keeps absolute units so that ``blank / flat ≈ 1``.

    Returns
    -------
    (H, W, 3) float32 flat-field.
    """
    if len(blank_images) == 0:
        raise ValueError("derive_flatfield needs at least one blank image")
    stack = np.stack([as_float_rgb(im) for im in blank_images], axis=0)
    if dark is not None:
        stack = np.clip(stack - as_float_rgb(dark)[None], 0.0, None)
    flat = np.median(stack, axis=0)
    if smooth_sigma > 0:
        flat = np.stack(
            [gaussian_filter(flat[..., c], sigma=smooth_sigma, mode="nearest") for c in range(3)],
            axis=-1,
        )
    if normalise:
        med = np.median(flat.reshape(-1, 3), axis=0)
        flat = flat / np.maximum(med, 1e-6)
    return np.clip(flat, 0.05, None).astype(np.float32)


def apply_flatfield(img: np.ndarray, flat: np.ndarray) -> np.ndarray:
    """Divide an image by the flat-field (resizing the flat if needed).

    Returns float32 RGB clipped to [0, 1].
    """
    rgb = as_float_rgb(img)
    if flat.shape[:2] != rgb.shape[:2]:
        import cv2

        flat = cv2.resize(flat, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(rgb / flat, 0.0, 1.0).astype(np.float32)


def flatfield_nonuniformity(flat: np.ndarray) -> float:
    """Scalar summary: (p99 − p1) / median of the luminance flat, i.e. how much
    the illumination varies across the field. Reported in calibration reports."""
    lum = flat.mean(axis=-1)
    p1, p99 = np.percentile(lum, [1, 99])
    return float((p99 - p1) / max(np.median(lum), 1e-6))
