# New-instrument onboarding protocol (S2-ID-3)

**Claim being made practical.** A site with a microscope the models have never
seen can run ADEPT after a calibration session — no retraining.

**Status.** Draft v0.1. The Target endpoint of S2-ID-3 delivers the written
protocol; the Maximum endpoint has it executed by someone other than the
student, on the engineered whole-slide imager treated as new.

## Inputs the site needs

* The ADEPT calibration kit (`docs/protocols/calibration_kit.md`).
* ≥ 10 physical H&E slides that have already been imaged on a calibrated
  instrument (the *paired verification set*), or a shipped set from the
  consortium.
* The frozen classifier (`artifacts/weights/`), the criteria config, and the
  published generalisation envelope (`docs/reporting/envelope_<release>.json`).

## Steps

### 1. Calibrate
Run the kit protocol; produce `artifacts/calibrations/<site>-<instrument>.json`.

### 2. Verify the canonical mapping
`adept calibrate validate` on the paired verification set. Acceptance as in the
kit protocol (inter/intra < 1, CI upper bound < 1, ICC reported).

### 3. Check the site's conditions against the envelope
Measure, on the site's own slides:

| Axis | Site statistic | Compare with |
|------|----------------|--------------|
| Stain | stain-vector angle to canonical; mean H and E concentration distribution | envelope boundary on the stain axis |
| Focus | distribution of relative sharpness on tissue tiles | envelope boundary on the focus axis |
| Illumination | flat-field non-uniformity; spectral gain | envelope boundary on the illumination axis |

If every site statistic lies inside the published envelope, the frozen
classifier is expected to meet the diagnostic targets on this instrument. If
not, the protocol says so **before** any clinical use: the site either fixes
the acquisition condition (e.g. illumination) or the consortium adds the
instrument to the next training round.

### 4. Swap the front-end
```python
pipe = AdequacyPipeline.load(...)  # frozen classifier
pipe = pipe.with_calibration(InstrumentCalibration.load("artifacts/calibrations/<site>.json"))
```
Nothing else changes: the schema, the ROI actions and the packages are
identical to those produced on the training instruments.

### 5. Local acceptance run
Assess the paired verification set on the new instrument; report sensitivity,
specificity and AUROC with bootstrap CIs against the existing labels
(`adept eval loio` with the new instrument as the group). This is a
*spot check*, not a validation study — it uses ≥ 10 slides, so CIs are wide;
its purpose is to catch gross failures.

## Outputs

* `artifacts/calibrations/<site>-<instrument>.json`
* `reports/onboarding_<site>-<instrument>.md` containing the calibration
  summary, the invariance table, the envelope check table, and the acceptance
  run metrics — generated from `scripts/s2_onboarding_report.py`.
* A GitHub issue from the *New instrument onboarding* template, closed with a
  link to the report.
