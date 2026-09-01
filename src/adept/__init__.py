"""ADEPT — Automated Diagnostic-adequacy Evaluation and Prioritization of Tissue.

Closed-loop autonomous microscopy for breast core-needle biopsy telepathology on
the OpenFlexure platform. The package is organised around one principle:

    calibrate once  →  canonical image space  →  one frozen classifier
                    →  validated leave-one-instrument-out.

Sub-packages
------------
calibration   Per-instrument stain matrix, flat-field, spectral correction; canonical mapping.
criteria      The five adequacy criteria as canonical-space measurement modules.
data          Slide registry, no-leakage stratified splits, LOIO folds, synthetic data.
eval          Diagnostic-accuracy metrics with bootstrap CIs, invariance, envelope sweeps.
models        Adequacy classifier, ROI ranking, MIL suspicion model (torch optional).
pipeline      Inference API and output packages.
schema        Pydantic output schema (AdequacyReport, ROI list, SuspicionReport).
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("adept-microscopy")
except PackageNotFoundError:  # pragma: no cover - editable/uninstalled
    __version__ = "0.0.0+local"

INSTRUMENTS = ("openflexure", "printed3d", "wsi", "scanner")
"""Canonical instrument identifiers used across configs, splits and reports.

openflexure  OpenFlexure motorised microscope (primary deployment instrument)
printed3d    3D-printed low-cost microscope (same site as openflexure → separates
             instrument variation from site variation)
wsi          Engineered whole-slide imager (treated as *new* for onboarding trials)
scanner      Reference commercial slide scanner
"""

__all__ = ["__version__", "INSTRUMENTS"]
