"""The canonical image space.

Every instrument's images are mapped into one shared representation so that a
single classifier can be trained and frozen once. The canonical space is
defined by:

* a fixed **canonical stain matrix** (target H and E OD directions),
* concentration images ``(H, W, 2)`` in canonical units, and
* an optional canonical RGB rendering for models that expect 3 channels.

Mapping an instrument image is: flat-field → spectral correction → OD →
deconvolve with *that instrument's* stain matrix → (optional per-stain gain)
→ re-render with the canonical matrix. The concentration image is the
primary canonical representation; the RGB rendering is derived from it.

The default canonical matrix is the widely used Ruifrok & Johnston H&E
reference. Semester 1 (S1-ID-1) may *replace* it with a matrix justified by the
measured spectral differences between the project's instruments — that
decision is recorded as an ADR and the value below updated accordingly.
"""

from __future__ import annotations

import numpy as np

from adept.calibration.od import od_to_rgb
from adept.calibration.stain import reconstruct_od

# Ruifrok & Johnston (2001) H&E OD vectors, unit-normalised. Rows: [H; E].
CANONICAL_STAIN_MATRIX: np.ndarray = np.array(
    [
        [0.6500286, 0.704031, 0.2860126],
        [0.07, 0.99, 0.11],
    ],
    dtype=np.float32,
)
CANONICAL_STAIN_MATRIX /= np.linalg.norm(CANONICAL_STAIN_MATRIX, axis=1, keepdims=True)


def canonical_rgb_from_concentrations(
    concentrations: np.ndarray,
    stain_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Render (H, W, 2) canonical concentrations as float32 RGB in [0, 1]."""
    M = CANONICAL_STAIN_MATRIX if stain_matrix is None else stain_matrix
    return od_to_rgb(reconstruct_od(concentrations, M))


def to_canonical(img: np.ndarray, calibration) -> tuple[np.ndarray, np.ndarray]:
    """Map a raw instrument image to canonical space.

    Thin functional alias for :meth:`InstrumentCalibration.to_canonical`;
    returns ``(concentrations (H, W, 2), canonical_rgb (H, W, 3))``.
    """
    return calibration.to_canonical(img)
