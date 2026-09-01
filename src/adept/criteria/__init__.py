"""The five adequacy criteria as canonical-space measurement modules (S1-ID-2).

Each module is a *measurement instrument*: a pure function from canonical-space
inputs to a :class:`~adept.schema.CriterionResult`, with thresholds supplied by
config so that the same code measures and the classifier / report decide.

| # | module        | measures the slide via                         |
|---|---------------|------------------------------------------------|
| 1 | tissue        | connected-component core detection             |
| 2 | cellularity   | nucleus density per tissue area                |
| 3 | staining      | calibrated H/E OD ratio, before normalisation  |
| 4 | focus         | tile-wise Laplacian variance, per-instrument   |
| 5 | artifact      | fold / bubble / crush / desiccation masks      |

:func:`assess_all` runs all five and returns the assembled list.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from adept.criteria.artifact import assess_artifact
from adept.criteria.cellularity import assess_cellularity
from adept.criteria.focus import assess_focus
from adept.criteria.staining import assess_staining
from adept.criteria.tissue import assess_tissue, tissue_mask_from_concentrations
from adept.schema import CriterionResult

CRITERIA = ("tissue", "cellularity", "staining", "focus", "artifact")


@dataclass
class CriteriaConfig:
    """Thresholds for the five criteria. Defaults are *placeholders* to be
    replaced by values justified in S1-ID-2; every field maps to a YAML key in
    ``configs/models/criteria.yaml``."""

    microns_per_pixel: float = 2.0  # overview-scan resolution
    tissue_min_total_length_um: float = 3000.0
    tissue_min_core_count: int = 1
    tissue_min_core_length_um: float = 500.0
    tissue_od_threshold: float = 0.15
    cellularity_min_nuclei_per_mm2: float = 300.0
    staining_he_ratio_range: tuple[float, float] = (0.35, 2.5)
    staining_min_h_od: float = 0.12
    focus_tile_px: int = 64
    focus_min_relative_sharpness: float = 0.35
    focus_min_fraction_in_focus: float = 0.6
    artifact_max_fraction: float = 0.25
    extras: dict = field(default_factory=dict)


def assess_all(
    concentrations: np.ndarray,
    corrected_rgb: np.ndarray,
    cfg: CriteriaConfig | None = None,
    focus_reference: float | None = None,
    artifact_model=None,
) -> list[CriterionResult]:
    """Run all five criteria.

    Parameters
    ----------
    concentrations : (H, W, 2) canonical stain concentrations [H, E].
    corrected_rgb : (H, W, 3) illumination-corrected RGB, used by the artifact
        detector (folds and bubbles are luminance phenomena).
    cfg : thresholds.
    focus_reference : instrument's best-focus Laplacian variance (from the
        calibration); ``None`` falls back to an absolute scale.
    artifact_model : optional trained artifact detector implementing
        ``predict_mask(rgb) -> (H, W) int mask`` (0 = clean, 1..4 = classes).
    """
    cfg = cfg or CriteriaConfig()
    mask = tissue_mask_from_concentrations(concentrations, cfg.tissue_od_threshold)
    return [
        assess_tissue(mask, cfg),
        assess_cellularity(concentrations, mask, cfg),
        assess_staining(concentrations, mask, cfg),
        assess_focus(concentrations, mask, cfg, focus_reference),
        assess_artifact(corrected_rgb, mask, cfg, artifact_model),
    ]


__all__ = ["CRITERIA", "CriteriaConfig", "assess_all"]
