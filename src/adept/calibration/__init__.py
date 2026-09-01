"""Instrument calibration and the canonical image space (Semester 1, S1-ID-1).

Each acquisition system is characterised **once** from a label-free calibration
kit (pure-H, pure-E, blank slides; a resolution target) into an
:class:`InstrumentCalibration`. Applying it maps a raw RGB image into the
*canonical image space* in which every downstream measurement and model lives.

    raw RGB  --flat-field-->  --spectral-->  --OD-->  --stain deconvolution-->  (H, E) OD
                                                                       └── canonical RGB
"""

from adept.calibration.canonical import (
    CANONICAL_STAIN_MATRIX,
    canonical_rgb_from_concentrations,
    to_canonical,
)
from adept.calibration.flatfield import apply_flatfield, derive_flatfield
from adept.calibration.instrument import InstrumentCalibration, derive_calibration
from adept.calibration.od import od_to_rgb, rgb_to_od
from adept.calibration.stain import (
    deconvolve,
    estimate_stain_vectors_macenko,
    stain_matrix_from_single_stains,
)

__all__ = [
    "CANONICAL_STAIN_MATRIX",
    "InstrumentCalibration",
    "apply_flatfield",
    "canonical_rgb_from_concentrations",
    "deconvolve",
    "derive_calibration",
    "derive_flatfield",
    "estimate_stain_vectors_macenko",
    "od_to_rgb",
    "rgb_to_od",
    "stain_matrix_from_single_stains",
    "to_canonical",
]
