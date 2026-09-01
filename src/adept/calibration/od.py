"""Optical-density (Beer–Lambert) transform.

All calibration mathematics is done in optical density (OD), where stain
concentrations add linearly:

    OD_c = -log10( I_c / I0_c )     for each colour channel c

Conventions
-----------
* RGB images are ``float32`` arrays of shape ``(H, W, 3)`` in ``[0, 1]``; ``uint8``
  inputs are converted transparently.
* OD arrays are ``float32`` of shape ``(H, W, 3)`` with non-negative values.
* ``I0`` is the per-channel white (background) intensity. After flat-field
  correction it should be ≈1; it is exposed so that uncorrected images can be
  handled explicitly.
"""

from __future__ import annotations

import numpy as np

_EPS = 1.0 / 255.0


def as_float_rgb(img: np.ndarray) -> np.ndarray:
    """Return ``img`` as float32 RGB in [0, 1] with shape (H, W, 3)."""
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    out = img.astype(np.float32, copy=False)
    if out.max() > 1.0 + 1e-6:
        out = out / 255.0
    return out


def rgb_to_od(img: np.ndarray, i0: np.ndarray | float = 1.0) -> np.ndarray:
    """Convert an RGB image to optical density.

    Parameters
    ----------
    img : (H, W, 3) RGB, uint8 or float in [0, 1].
    i0 : per-channel background intensity (scalar or shape (3,)). Default 1.0
        assumes a flat-field-corrected image.

    Returns
    -------
    (H, W, 3) float32 OD, clipped at 0.
    """
    rgb = as_float_rgb(img)
    i0 = np.asarray(i0, dtype=np.float32)
    ratio = np.clip(rgb / np.maximum(i0, _EPS), _EPS, 1.0)
    return (-np.log10(ratio)).astype(np.float32)


def od_to_rgb(od: np.ndarray, i0: np.ndarray | float = 1.0) -> np.ndarray:
    """Inverse of :func:`rgb_to_od`. Returns float32 RGB in [0, 1]."""
    i0 = np.asarray(i0, dtype=np.float32)
    rgb = i0 * np.power(10.0, -np.asarray(od, dtype=np.float32))
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def od_magnitude(od: np.ndarray) -> np.ndarray:
    """Per-pixel Euclidean OD magnitude, shape (H, W). Used for tissue masks."""
    return np.linalg.norm(od, axis=-1)
