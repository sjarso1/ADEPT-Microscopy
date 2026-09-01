# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
semantic versioning. Semester archive tags (`s1-id1` … `s3-final`) mark the
exact commits behind each report.

## [0.1.0] — 2026-09-01

### Added
- Repository scaffold for the three-semester ADEPT computational workstream.
- `adept.calibration`: optical-density transform, single-stain and Macenko
  stain-vector estimation, absolute-unit flat-field, per-instrument
  `InstrumentCalibration` (JSON-serialisable) and the canonical image space.
- `adept.criteria`: the five adequacy criteria (tissue, cellularity, staining,
  focus, artifact) as canonical-space measurement modules with a shared
  `CriteriaConfig`.
- `adept.schema`: Pydantic output schema v1.1.0 (`AdequacyReport`,
  `RegionOfInterest`, `SuspicionReport`) shared with the hardware and
  validation workstreams.
- `adept.data`: slide registry, slide-level stratified locked splits with
  hash verification, leave-one-instrument-out folds, and a synthetic
  multi-instrument dataset generator with calibration kits.
- `adept.eval`: sensitivity / specificity / AUROC / Cohen's κ with stratified
  bootstrap CIs, paired-acquisition invariance statistics (inter/intra ratio,
  ICC), and generalisation-envelope perturbation sweeps.
- `adept.models`: transparent five-criterion combiner, adequacy-driven ROI
  ranking with closed-loop `refocus`/`rescan` actions and an optional
  suspicion term; torch-optional CNN, artifact-patch and attention-MIL models.
- `adept.pipeline`: `AdequacyPipeline` with the calibration front-end as a
  swappable stage; output packages with completeness manifest and QC.
- `adept` CLI, `scripts/s1_train_combiner.py` reference experiment, CI
  end-to-end smoke test, 43 unit tests on synthetic data.
- Documentation: architecture, roadmap, calibration-kit and new-instrument
  onboarding protocols, output schema, data governance, TRIPOD+AI and STARD
  checklists, ADR template and first two ADRs.
