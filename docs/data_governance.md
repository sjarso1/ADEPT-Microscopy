# Data governance

## What lives where

| Data | Location | In git? |
|------|----------|---------|
| Patient slide images (overview scans, HR tiles) | project object store; local mirror under `data/raw/` | **Never** |
| Pathologist reference labels (B1 / B2+, reason codes) | `data/labels/` (access-controlled) | Never |
| Clinical pathology reports and extracted diagnostic labels (Semester 3) | tissue-bank system; extracted table under `data/labels/` | Never |
| Calibration-kit images | `data/calibration_kits/` | Never (large); derived JSON is tracked |
| Per-instrument calibration JSON | `artifacts/calibrations/` | Yes (small, no patient content) |
| Locked split files | `data/splits/` locally; a copy under `configs/splits/` once used in a report | Yes — immutable once reported on |
| Trained weights | `artifacts/weights/` and Zenodo | No (Zenodo DOI recorded in the release notes) |
| Synthetic data | generated on demand (`adept synth make-dataset`) | No (regenerable) |

## Rules

1. Slide identifiers in the registry are the tissue-bank accession codes
   re-keyed through a project-internal mapping held by the clinical lead. No
   names, dates of birth, or hospital numbers appear anywhere in this
   repository, in experiment logs, or in file names.
2. Access to `data/labels/` and to the archived pathology reports (Semester 3)
   requires the ethics amendment covering diagnostic use; confirm its status
   before S3-ID-1 begins.
3. The open Zenodo release of the multi-instrument dataset contains only
   de-identified images and slide-level labels approved for release by the
   sponsor and the clinical lead.
4. Weights & Biases runs log metrics and configs only — never images that
   contain patient tissue.
5. The pre-commit hook `check-added-large-files` (2 MB) is a guard, not a
   policy: if you find yourself wanting to raise the limit, stop and ask.

## Reference-standard integrity

* Inter-rater κ ≥ 0.75 is required before any classifier training begins;
  disagreements are adjudicated and the adjudication logged.
* Labels attach to the **physical slide**; every acquisition of that slide
  inherits them. This is why splits are slide-level (`adept.data.splits`).
* Semester 3 diagnostic labels: provenance (report date, reporting pathologist
  role, original microscopy) is recorded in the metadata table; a blinded
  re-review sample estimates label noise.
