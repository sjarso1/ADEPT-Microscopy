# Research roadmap — three sequential MSE semesters

Each semester has three interim deliverables (ID-1..3, 10 % each) and a final
report (70 %), with Minimum / Target / Maximum endpoints. Every deliverable
ends with a tagged commit (`s1-id1`, …, `s3-final`) whose code, configs,
splits, weights and logs produced the numbers in the report.

## Performance budget (fixed by the clinical task)

| | Sensitivity | Specificity | AUROC | Concordance |
|-|-|-|-|-|
| Adequacy (B1 detection) | ≥ 0.90 | ≥ 0.85 | ≥ 0.92 | κ ≥ 0.75 |
| Malignancy-suspicion triage | ≥ 0.95 | ≥ 0.80 | ≥ 0.90 | top-tile hit-rate ≥ 0.80 |

Labels are trusted only after inter-rater κ ≥ 0.75.

---

## Semester 1 — Instrument calibration, the canonical space, and the multi-criterion adequacy classifier

| Deliverable | Week | Minimum | Target | Maximum | Code entry point |
|-------------|------|---------|--------|---------|------------------|
| **S1-ID-1** Calibrations & canonical space | 5 | OpenFlexure + scanner calibrations; canonical mapping validated on one paired set | + 3D-printed microscope; inter- vs intra-instrument residuals with error bars; documented protocol | + drift check on kit re-run; protocol executed by an independent user | `adept calibrate derive/validate`, `adept.eval.invariance` |
| **S1-ID-2** Five criteria & invariance | 9 | Criteria 1, 3, 4 + single-type artifact detector; invariance OpenFlexure vs scanner | + cellularity; four-class artifact classifier; five-criterion assembly; full invariance table | + two nucleus detectors compared; reason codes vs pathologist reasons | `adept.criteria`, `adept.models.artifact_cnn` |
| **S1-ID-3** Classifier, validation, ROI | 12 | Trained canonical classifier: sens/spec/AUROC + basic ROI ranking | + per-instrument breakdown, κ, bootstrap CIs, ablation, calibration analysis | + reason-code validation; cost-based threshold; failure gallery by instrument | `scripts/s1_train_combiner.py`, `adept.models.adequacy_cnn`, `adept.models.roi` |
| **Final** | 14 | One validation story from calibration to concordance; TRIPOD-AI + STARD | | | tag `s1-final` |

Weeks 1–2: environment, literature review (adequacy, colour deconvolution, stain normalisation, diagnostic-accuracy methods), repository, confirm kit images and κ ≥ 0.75 on labels.

## Semester 2 — Cross-instrument generalisation, envelope characterisation, and system integration

| Deliverable | Week | Minimum | Target | Maximum | Code entry point |
|-------------|------|---------|--------|---------|------------------|
| **S2-ID-1** LOIO harness & three-arm experiment | 5 | LOIO harness; cross-instrument gap for the canonical classifier; cross-site evaluation | + per-instrument vs pooled vs canonical arms with CIs; Macenko on pooled arm; recorded architecture decision | + foundation-encoder arm; second normalisation (Vahadane); per-site/per-instrument gap breakdowns | `adept.data.loio_folds`, `scripts/s2_three_arm.py` |
| **S2-ID-2** Envelope & domain adaptation | 9 | Baseline envelope on stain/focus/illumination, primary instrument | + domain adaptation; re-measured envelope; baseline-vs-adapted; per-instrument | + second adaptation method; anchoring vs real LMIC variation | `adept.eval.envelope`, `scripts/s2_envelope.py` |
| **S2-ID-3** Inference API, packages, onboarding | 12 | Inference API + packages for a subset of reading-study slides | + full reading-study set with QC; written onboarding protocol | + live end-to-end demo; re-scan paired comparison; onboarding by an independent user on the WSI treated as new | `adept.pipeline`, `docs/protocols/new_instrument_onboarding.md` |
| **Final** | 14 | One generalisation story; Zenodo dataset + descriptor; draft manuscript | | | tag `s2-final` |

## Semester 3 — Malignancy-suspicion triage from targeted HR tiles, with cross-instrument validation

| Deliverable | Week | Minimum | Target | Maximum | Code entry point |
|-------------|------|---------|--------|---------|------------------|
| **S3-ID-1** Diagnostic dataset & encoder baseline | 5 | Labels extracted & audited; scope decision (binary vs three-way); locked split | + adequacy-QC'd canonical tile set; two-encoder baseline with CIs | + blinded re-review label-noise estimate; per-instrument baseline | `scripts/s3_extract_labels.py`, `adept.data.registry` (diagnosis column) |
| **S3-ID-2** MIL suspicion model | 9 | Trained MIL model: sens/spec/AUROC on held-out | + CIs, calibration, cost-based threshold, attention vs top-k, per-instrument | + LOIO triage; three-way if counts allow; failure gallery | `adept.models.mil`, `scripts/s3_train_mil.py` |
| **S3-ID-3** Attention validation & integration | 12 | Attention concordance on validation subset; suspicion folded into ROI ranking | + cross-instrument attention stability; regenerated packages with QC; hit-rate target with CIs | + pathologist usefulness judgment; end-to-end via onboarding protocol on a new instrument | `adept.pipeline.AdequacyPipeline(suspicion_hook=…)`, `scripts/s3_attention_concordance.py` |
| **Final** | 14 | One triage story; claim ceiling stated (triage, not diagnosis); Zenodo label extension | | | tag `s3-final` |

Out of scope for all semesters: Nottingham grading, mitotic counting, histological subtype classification.

## Dependencies between workstreams

* Hardware → computational: kit images and paired acquisitions (S1 W3), LMIC partner image set (S2 W1), HR tile acquisitions (S3 W4).
* Validation → computational: reference labels with κ ≥ 0.75 (S1 W2), reading-study slide list (S2 W11), pathologist-marked diagnostic regions (S3 W10).
* Computational → hardware: `RegionOfInterest` schema and ROI actions (S1 W12 onward).
