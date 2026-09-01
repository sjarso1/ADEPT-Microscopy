"""Criterion 1 — Tissue: is there a biopsy core at all, and how much?

Connected-component labelling of the tissue mask gives core count, per-core
length (major-axis length of the component) and total tissue length; the
adequacy of a core-needle biopsy is conventionally judged on total core length
and the number of intact cores.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label, regionprops

from adept.schema import CriterionResult


def tissue_mask_from_concentrations(
    concentrations: np.ndarray, od_threshold: float = 0.15, min_area_px: int = 64
) -> np.ndarray:
    """Boolean (H, W) tissue mask from canonical concentrations.

    A pixel is tissue when its total stain concentration exceeds the threshold;
    the mask is then closed and cleaned of small specks.
    """
    total = concentrations.sum(axis=-1)
    mask = total > od_threshold
    mask = ndi.binary_closing(mask, structure=np.ones((5, 5)))
    mask = ndi.binary_fill_holes(mask)
    if mask.any():
        lab, n = ndi.label(mask)
        sizes = np.bincount(lab.ravel())
        keep = sizes >= min_area_px
        keep[0] = False
        mask = keep[lab]
    return mask.astype(bool)


def detect_cores(mask: np.ndarray, microns_per_pixel: float, min_core_length_um: float = 500.0):
    """Label cores in the tissue mask.

    Returns a list of dicts with ``label, area_px, length_um, width_um, bbox``,
    sorted by length descending. Components shorter than ``min_core_length_um``
    are treated as fragments and excluded from the core count (but still
    counted in total tissue area).
    """
    lab = label(mask, connectivity=2)
    cores = []
    for rp in regionprops(lab):
        length_um = rp.axis_major_length * microns_per_pixel
        if length_um < min_core_length_um:
            continue
        cores.append(
            {
                "label": int(rp.label),
                "area_px": int(rp.area),
                "length_um": float(length_um),
                "width_um": float(rp.axis_minor_length * microns_per_pixel),
                "bbox": [int(v) for v in rp.bbox],
            }
        )
    cores.sort(key=lambda c: -c["length_um"])
    return cores


def assess_tissue(mask: np.ndarray, cfg) -> CriterionResult:
    cores = detect_cores(mask, cfg.microns_per_pixel, cfg.tissue_min_core_length_um)
    total_len = float(sum(c["length_um"] for c in cores))
    area_mm2 = float(mask.sum() * (cfg.microns_per_pixel / 1000.0) ** 2)
    score = float(np.clip(total_len / max(cfg.tissue_min_total_length_um, 1e-6), 0, 1))
    passed = len(cores) >= cfg.tissue_min_core_count and total_len >= cfg.tissue_min_total_length_um
    return CriterionResult(
        name="tissue",
        value=total_len,
        unit="um_total_core_length",
        score=score,
        passed=bool(passed),
        threshold=cfg.tissue_min_total_length_um,
        details={"core_count": len(cores), "tissue_area_mm2": area_mm2, "cores": cores[:10]},
    )
