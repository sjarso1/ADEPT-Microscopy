# ADEPT architecture

ADEPT is a closed loop: **scan → assess → act → capture → package**. The
computational workstream (this repository) owns everything between the image
arriving from the microscope and the package leaving for the pathologist. The
hardware workstream owns the OpenFlexure stage, autofocus and acquisition; the
validation workstream owns the reference standard and the reading study.

```mermaid
flowchart LR
    subgraph HW["Hardware workstream (OpenFlexure)"]
        A[Low-mag overview scan] --> B[Stage controller]
        B --> C[Targeted HR tiles ≥20× eq.]
    end
    subgraph SW["This repository"]
        A --> D[Calibration front-end<br/>InstrumentCalibration]
        D --> E[Canonical image space<br/>H/E concentrations]
        E --> F1[Tissue]
        E --> F2[Cellularity]
        E --> F3[Staining]
        E --> F4[Focus]
        E --> F5[Artifact]
        F1 & F2 & F3 & F4 & F5 --> G[Adequacy classifier<br/>B1 vs B2+ + reason codes]
        E --> H[ROI ranking]
        G --> H
        H -->|refocus / rescan ROIs| B
        H -->|capture ROIs| B
        C --> I[Suspicion model (S3)<br/>attention-MIL]
        I --> H
        G & H --> J[AdequacyReport<br/>schema v1.1]
        J --> K[Output package<br/>report · mosaic · ranked tiles]
    end
    K --> L[Remote pathologist]
```

## The one rule

**Raw instrument colour never reaches a model.** Every measurement and every
learned model consumes canonical-space inputs produced by
`adept.calibration`. The only deliberate exception is Criterion 3 (staining),
which reads the *calibrated but un-normalised* concentrations so that true
staining failures are not erased by later normalisation.

## Package responsibilities

| Package | Owns | Must not |
|---------|------|----------|
| `adept.calibration` | Kit → `InstrumentCalibration`; raw → canonical | Know anything about adequacy or labels |
| `adept.criteria` | Five measurement modules → `CriterionResult` | Decide adequacy (that is the classifier's job) |
| `adept.models` | Classifier arms, ROI ranking, MIL | Read raw images |
| `adept.data` | Registry, locked splits, LOIO folds, synthetic data | Be bypassed when reporting a number |
| `adept.eval` | Metrics + CIs, invariance, envelope | Return bare point estimates |
| `adept.pipeline` | Stable inference API, output packages | Change the schema without a version bump + ADR |
| `adept.schema` | The contract with the other two workstreams | Remove or rename a field |

## The canonical image space

```
raw RGB ──/ flat-field (I0(x,y)) ──► transmittance ──► OD = −log10(T)
        ──► deconvolve with the INSTRUMENT's stain matrix ──► (c_H, c_E) instrument units
        ──► × per-stain gain (kit-derived)                ──► (c_H, c_E) canonical units
        ──► re-render with the CANONICAL stain matrix     ──► canonical RGB (for CNNs)
```

The per-instrument quantities (stain matrix, gains, flat-field, focus
reference) are derived **once** from the label-free kit
(`docs/protocols/calibration_kit.md`) and stored as a small JSON under
`artifacts/calibrations/`. Swapping that JSON is the *entire* cost of moving to
a new microscope (`AdequacyPipeline.with_calibration`).

## Closed-loop signals

`RegionOfInterest.action` is the interface to the stage controller:

* `refocus` / `rescan` — tissue tiles whose relative sharpness is below
  threshold (Criterion 4). Ranked first so the loop closes before capture.
* `capture_hr` — the best tissue tiles by adequacy quality, re-ordered by the
  Semester 3 suspicion attention when present.

## Validation architecture

* **Locked split** at the physical-slide level, stratified by site × label;
  hash-verified on load (`adept.data.splits.LockedSplit`).
* **LOIO** folds layered on the split (`loio_folds`): each instrument held out
  of training entirely.
* **Every metric with a 95 % bootstrap CI** (`adept.eval.metrics`), stratified
  resampling; per-instrument, per-site and overall.
* **Invariance** (`adept.eval.invariance`): inter/intra-instrument ratio and
  ICC(2,1) on paired acquisitions, before and after calibration.
* **Envelope** (`adept.eval.envelope`): stress sweeps along stain / focus /
  illumination until a target is crossed.

## Where the semesters plug in

| Semester | Adds | Touches |
|----------|------|---------|
| 1 | Calibrations, criteria, combiner + CNN arms, ROI ranking | everything above |
| 2 | LOIO harness runs, three-arm experiment, normalisation, domain adaptation, envelope, inference API, packages, onboarding protocol | `eval`, `models`, `pipeline`, `docs/protocols` |
| 3 | Label extraction, tile curation, encoders, attention-MIL, attention concordance, schema v1.x suspicion fields | `models.mil`, `schema`, `pipeline` (suspicion hook) |
