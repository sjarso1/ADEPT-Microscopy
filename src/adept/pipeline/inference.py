"""The stable inference interface.

    AdequacyPipeline(calibration, combiner, criteria_cfg).assess(image, slide_id)
        -> AdequacyReport   (schema: adept.schema)

Design points (from the Semester 2 course agreement):

* The **calibration front-end is an explicit, swappable stage** — pass a
  different :class:`InstrumentCalibration` and nothing else changes. That is
  what "onboarding a new microscope costs a calibration session" means in code.
* The classifier is swappable too (any object with ``report(...)``), so the
  transparent combiner, the CNN arms and the adapted model all sit behind the
  same call.
* The Semester 3 suspicion model is an *optional* hook that populates
  ``RegionOfInterest.suspicion_score`` and re-orders the capture list; the
  schema and every existing consumer are untouched when it is absent.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from adept.calibration import InstrumentCalibration
from adept.criteria import CriteriaConfig, assess_all
from adept.criteria.focus import tile_tissue_fraction
from adept.criteria.tissue import tissue_mask_from_concentrations
from adept.models.adequacy import CriterionCombiner
from adept.models.roi import rank_rois
from adept.schema import AdequacyReport

SuspicionHook = Callable[[np.ndarray, np.ndarray], np.ndarray]
"""(concentrations (H,W,2), tissue_tile_map (nH,nW)) -> suspicion map (nH,nW) in [0,1]"""


@dataclass
class AdequacyPipeline:
    calibration: InstrumentCalibration
    classifier: CriterionCombiner
    criteria_cfg: CriteriaConfig
    artifact_model: object | None = None
    suspicion_hook: SuspicionHook | None = None
    max_capture: int = 8
    max_rescan: int = 4

    def assess(self, image: np.ndarray, slide_id: str) -> AdequacyReport:
        cal = self.calibration
        corrected = cal.correct_illumination(image)
        C, _canonical_rgb = cal.to_canonical(image)
        criteria = assess_all(
            C,
            corrected,
            self.criteria_cfg,
            focus_reference=cal.focus_reference,
            artifact_model=self.artifact_model,
        )
        # tile maps for ROI ranking (reuse the focus criterion's tiling)
        tile_px = self.criteria_cfg.focus_tile_px
        mask = tissue_mask_from_concentrations(C, self.criteria_cfg.tissue_od_threshold)
        tissue_tiles = tile_tissue_fraction(mask, tile_px) > 0.2
        focus = next(c for c in criteria if c.name == "focus")
        rel_sharp = np.asarray(focus.details["relative_sharpness_map"], dtype=float)
        cell_map = _tile_mean(C[..., 0], tile_px)
        cell_map = cell_map / max(float(cell_map.max()), 1e-6)
        susp = self.suspicion_hook(C, tissue_tiles) if self.suspicion_hook else None
        rois = rank_rois(
            tissue_tiles,
            rel_sharp,
            tile_px,
            cellularity_map=cell_map,
            suspicion_map=susp,
            max_capture=self.max_capture,
            max_rescan=self.max_rescan,
            min_relative_sharpness=self.criteria_cfg.focus_min_relative_sharpness,
        )
        return self.classifier.report(
            slide_id=slide_id,
            instrument=cal.instrument,
            criteria=criteria,
            calibration_id=cal.metadata.get("calibration_id", cal.instrument),
            rois=rois,
        )

    def with_calibration(self, calibration: InstrumentCalibration) -> AdequacyPipeline:
        """Swap the calibration front-end (new-instrument onboarding)."""
        return AdequacyPipeline(
            calibration,
            self.classifier,
            self.criteria_cfg,
            self.artifact_model,
            self.suspicion_hook,
            self.max_capture,
            self.max_rescan,
        )


def _tile_mean(arr: np.ndarray, tile_px: int) -> np.ndarray:
    H, W = arr.shape
    nH, nW = max(H // tile_px, 1), max(W // tile_px, 1)
    out = np.zeros((nH, nW), np.float32)
    for i in range(nH):
        for j in range(nW):
            out[i, j] = arr[i * tile_px : (i + 1) * tile_px, j * tile_px : (j + 1) * tile_px].mean()
    return out


__all__ = ["AdequacyPipeline"]
