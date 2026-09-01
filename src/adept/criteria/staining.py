"""Criterion 3 — Staining quality from calibrated optical densities.

Measured from the **calibrated, un-normalised** stain concentrations — i.e.
the canonical-space concentrations produced by the instrument calibration
(flat-field, spectral correction, the instrument's own stain matrix and
per-stain gain) but *before* any stain normalisation. Normalisation
(Macenko / Vahadane and the like, Semester 2) maps every image toward a
target stain appearance and would erase a real staining failure — weak
haematoxylin, over-eosinophilic sections, faded slides — so it must never
sit upstream of this criterion.

Outputs the mean H and E concentration over tissue and their ratio.
"""

from __future__ import annotations

import numpy as np

from adept.schema import CriterionResult


def he_optical_densities(
    concentrations: np.ndarray, tissue_mask: np.ndarray
) -> tuple[float, float]:
    """Mean H and E calibrated concentration over tissue.

    The mean (not the median) is used so that sparse nuclei contribute to the
    haematoxylin figure — the median of H over tissue is dominated by stroma.
    """
    if not tissue_mask.any():
        return 0.0, 0.0
    h = float(np.mean(concentrations[..., 0][tissue_mask]))
    e = float(np.mean(concentrations[..., 1][tissue_mask]))
    return h, e


def assess_staining(concentrations: np.ndarray, tissue_mask: np.ndarray, cfg) -> CriterionResult:
    h, e = he_optical_densities(concentrations, tissue_mask)
    ratio = h / e if e > 1e-6 else float("inf") if h > 0 else 0.0
    lo, hi = cfg.staining_he_ratio_range
    in_range = (lo <= ratio <= hi) and h >= cfg.staining_min_h_od
    # score: 1 inside the range, decays outside (log-ratio distance)
    if ratio <= 0 or not np.isfinite(ratio):
        score = 0.0
    else:
        centre = np.sqrt(lo * hi)
        half_width = np.log(hi / centre)
        dist = abs(np.log(ratio / centre)) / max(half_width, 1e-6)
        score = float(np.clip(1.5 - dist, 0, 1))  # 1 within range, 0 at 2.5× the half-width
    score *= float(np.clip(h / max(cfg.staining_min_h_od, 1e-6), 0, 1))
    return CriterionResult(
        name="staining",
        value=float(ratio if np.isfinite(ratio) else 0.0),
        unit="H_over_E_OD_ratio",
        score=score,
        passed=bool(in_range),
        threshold=None,
        details={"mean_h_od": h, "mean_e_od": e, "ratio_range": [lo, hi]},
    )
