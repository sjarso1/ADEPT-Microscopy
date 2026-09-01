"""`InstrumentCalibration`: the single object that carries per-instrument state.

Derived once from the calibration kit (:func:`derive_calibration`), serialised
to JSON under ``artifacts/calibrations/<instrument>.json`` (small enough to be
git-tracked), and applied to every image of that instrument.

Contents
--------
stain_matrix        (2, 3) unit OD vectors for H and E *on this instrument*
spectral_gain       (3,) per-channel multiplicative correction so that the
                    blank-slide white point is neutral (grey-world on the kit)
stain_gain          (2,) per-stain concentration scaling that maps this
                    instrument's H/E concentration scale onto the canonical scale
flatfield           (H, W, 3) illumination profile (stored as a small
                    down-sampled array; up-sampled on application)
focus_reference     contrast-normalised sharpness of the resolution target at best focus —
                    lets Criterion 4 express sharpness relative to *this*
                    instrument's achievable optimum
metadata            free-form provenance (kit ids, date, operator, firmware)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from adept.calibration.canonical import CANONICAL_STAIN_MATRIX, canonical_rgb_from_concentrations
from adept.calibration.flatfield import apply_flatfield, derive_flatfield, flatfield_nonuniformity
from adept.calibration.od import as_float_rgb, rgb_to_od
from adept.calibration.stain import deconvolve, stain_angle_deg, stain_matrix_from_single_stains

_FLAT_STORE_SIZE = 64  # flat-field is stored down-sampled to this many pixels per side


@dataclass
class InstrumentCalibration:
    instrument: str
    stain_matrix: np.ndarray
    spectral_gain: np.ndarray = field(default_factory=lambda: np.ones(3, np.float32))
    stain_gain: np.ndarray = field(default_factory=lambda: np.ones(2, np.float32))
    flatfield: np.ndarray | None = None
    focus_reference: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ apply
    def correct_illumination(self, img: np.ndarray) -> np.ndarray:
        """Flat-field + spectral (white-balance) correction → float32 RGB [0, 1]."""
        rgb = as_float_rgb(img)
        if self.flatfield is not None:
            rgb = apply_flatfield(rgb, self.flatfield)
        rgb = np.clip(rgb * self.spectral_gain[None, None, :], 0.0, 1.0)
        return rgb.astype(np.float32)

    def to_od(self, img: np.ndarray) -> np.ndarray:
        """Corrected optical density (H, W, 3)."""
        return rgb_to_od(self.correct_illumination(img))

    def to_concentrations(self, img: np.ndarray) -> np.ndarray:
        """Canonical-scale (H, W, 2) stain concentrations [H, E]."""
        C = deconvolve(self.to_od(img), self.stain_matrix)
        return (C * self.stain_gain[None, None, :]).astype(np.float32)

    def to_canonical(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(concentrations (H, W, 2), canonical RGB (H, W, 3))``."""
        C = self.to_concentrations(img)
        return C, canonical_rgb_from_concentrations(C, CANONICAL_STAIN_MATRIX)

    # ------------------------------------------------------------- diagnostics
    def summary(self) -> dict[str, Any]:
        """Scalar diagnostics reported in S1-ID-1 calibration tables."""
        angles = stain_angle_deg(self.stain_matrix, CANONICAL_STAIN_MATRIX)
        return {
            "instrument": self.instrument,
            "stain_matrix": self.stain_matrix.round(4).tolist(),
            "angle_to_canonical_deg": {"H": float(angles[0]), "E": float(angles[1])},
            "spectral_gain": self.spectral_gain.round(4).tolist(),
            "stain_gain": self.stain_gain.round(4).tolist(),
            "flatfield_nonuniformity": (
                None if self.flatfield is None else flatfield_nonuniformity(self.flatfield)
            ),
            "focus_reference": self.focus_reference,
        }

    # ------------------------------------------------------------- persistence
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["stain_matrix"] = self.stain_matrix.tolist()
        d["spectral_gain"] = self.spectral_gain.tolist()
        d["stain_gain"] = self.stain_gain.tolist()
        d["flatfield"] = None if self.flatfield is None else self.flatfield.tolist()
        d["schema_version"] = 1
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> InstrumentCalibration:
        return cls(
            instrument=d["instrument"],
            stain_matrix=np.asarray(d["stain_matrix"], np.float32),
            spectral_gain=np.asarray(d.get("spectral_gain", [1, 1, 1]), np.float32),
            stain_gain=np.asarray(d.get("stain_gain", [1, 1]), np.float32),
            flatfield=None
            if d.get("flatfield") is None
            else np.asarray(d["flatfield"], np.float32),
            focus_reference=d.get("focus_reference"),
            metadata=d.get("metadata", {}),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1))
        return path

    @classmethod
    def load(cls, path: str | Path) -> InstrumentCalibration:
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def identity(cls, instrument: str = "canonical") -> InstrumentCalibration:
        """A no-op calibration (canonical stain matrix, unit gains). Useful for
        images that are already canonical, and as a baseline in ablations."""
        return cls(instrument=instrument, stain_matrix=CANONICAL_STAIN_MATRIX.copy())


# ---------------------------------------------------------------------- derive
def _downsample(arr: np.ndarray, size: int) -> np.ndarray:
    import cv2

    return cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)


def derive_calibration(
    instrument: str,
    pure_h: list[np.ndarray],
    pure_e: list[np.ndarray],
    blank: list[np.ndarray],
    resolution_target: np.ndarray | None = None,
    reference_stain_gain: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> InstrumentCalibration:
    """Derive an :class:`InstrumentCalibration` from calibration-kit images.

    Steps
    -----
    1. Flat-field from the blank slide(s).
    2. Spectral gain: grey-world white balance from the flat-field-corrected blank.
    3. Stain matrix: dominant OD direction of the corrected pure-H and pure-E slides.
    4. Stain gain: scale so that the *median* stained-pixel concentration of the
       kit slides equals ``reference_stain_gain`` (default: 1.0 per stain). When
       the same kit is imaged on the reference scanner, pass the scanner's
       medians here to put all instruments on the scanner's concentration scale.
    5. Focus reference: contrast-normalised sharpness of the resolution target's
       canonical H channel (if given) — the same metric Criterion 4 uses.
    """
    flat = derive_flatfield(blank)
    blank_corr = np.stack([apply_flatfield(b, flat) for b in blank]).reshape(-1, 3)
    white = np.percentile(blank_corr, 50, axis=0)
    spectral_gain = (white.mean() / np.maximum(white, 1e-6)).astype(np.float32)

    def corr(im):
        return np.clip(apply_flatfield(im, flat) * spectral_gain, 0, 1)

    h_imgs = [corr(im) for im in pure_h]
    e_imgs = [corr(im) for im in pure_e]
    M = stain_matrix_from_single_stains(h_imgs, e_imgs)

    # per-stain concentration scale of the kit on this instrument
    def median_conc(imgs, idx):
        vals = []
        for im in imgs:
            C = deconvolve(rgb_to_od(im), M)[..., idx]
            vals.append(np.median(C[C > 0.05]) if np.any(C > 0.05) else np.nan)
        return float(np.nanmean(vals))

    med = np.array([median_conc(h_imgs, 0), median_conc(e_imgs, 1)], np.float32)
    target = np.ones(2, np.float32) if reference_stain_gain is None else reference_stain_gain
    stain_gain = np.where(np.isfinite(med) & (med > 0), target / med, 1.0).astype(np.float32)

    cal = InstrumentCalibration(
        instrument=instrument,
        stain_matrix=M,
        spectral_gain=spectral_gain,
        stain_gain=stain_gain,
        flatfield=_downsample(flat, _FLAT_STORE_SIZE),
        focus_reference=None,
        metadata={"kit_median_concentration": med.tolist(), **(metadata or {})},
    )
    if resolution_target is not None:
        # measured in the same domain as Criterion 4 uses: the canonical H channel
        from adept.criteria.focus import normalized_sharpness

        cal.focus_reference = float(
            normalized_sharpness(cal.to_concentrations(resolution_target)[..., 0])
        )
    return cal
