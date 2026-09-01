# Calibration-kit protocol (label-free, per instrument)

**Purpose.** Characterise one acquisition system once, producing an
`InstrumentCalibration` JSON that maps its images into the canonical image
space. Requires no annotated tissue. Takes roughly 30 minutes per instrument.

**Status.** Draft v0.1 — to be refined and validated in S1-ID-1. The Target
endpoint requires this document to be re-runnable; the Maximum endpoint
requires it to be executed end-to-end by someone other than the author.

## Kit contents

| Item | Purpose | Notes |
|------|---------|-------|
| Pure-haematoxylin slide | H stain vector | Standard section stained with haematoxylin only; uniform, moderately stained |
| Pure-eosin slide | E stain vector | Section stained with eosin only |
| Blank slide | Illumination flat-field, white point | Glass + mounting medium + coverslip, no tissue; clean |
| Resolution target | Focus reference, µm/px | USAF 1951 or Ronchi ruling; printed targets acceptable at 4× |
| (optional) H&E control slide | Cross-check via Macenko | The same physical slide across all instruments |

The *same physical kit* is imaged on every instrument so that per-stain gains
share one reference scale.

## Acquisition

1. Warm up the illumination for ≥ 5 min. Fix exposure / gain / white balance
   for the session and record them in `metadata`.
2. **Blank**: 3–5 fields at different stage positions, coverslip clean. Focus
   on dust, then move to a clean area. Do not saturate: the brightest pixel
   should be < 95 % of full scale.
3. **Pure-H, pure-E**: 2–3 fields each in uniformly stained regions.
4. **Resolution target**: 1 field at best focus (use the instrument's autofocus
   if available; otherwise a manual through-focus stack and keep the sharpest).
5. Save as lossless PNG (or 16-bit TIFF) using the file-name convention
   `pure_h_<i>.png`, `pure_e_<i>.png`, `blank_<i>.png`, `target_<i>.png` in
   `data/calibration_kits/<instrument>/`.

## Derivation

```bash
adept calibrate derive --kit data/calibration_kits/<instrument> \
     --instrument <instrument> --out artifacts/calibrations/<instrument>.json
```

Steps performed (see `adept.calibration.instrument.derive_calibration`):

1. **Flat-field** `I0(x, y)`: median of blank frames, light Gaussian smoothing
   (σ = 5 px) to suppress dust, kept in absolute units.
2. **Spectral gain**: grey-world white balance from the corrected blank
   (≈ 1 when the flat is used; retained for flat-less operation).
3. **Stain matrix**: dominant OD direction (SVD) of the corrected pure-H and
   pure-E images; rows unit-normalised.
4. **Per-stain gain**: scale such that the median kit concentration equals the
   reference (1.0, or the reference scanner's medians via `--reference`).
5. **Focus reference**: contrast-normalised sharpness of the target's canonical
   H channel — the same metric Criterion 4 uses.

Record the summary printed by the command (stain angles to canonical, flat-field
non-uniformity, focus reference) in the calibration table of the S1-ID-1 report.

## Verification (paired slides)

Image ≥ 10 physical H&E slides on this instrument and on at least one already
calibrated instrument, then:

```bash
adept calibrate validate --pairs data/pairs.csv --calibrations artifacts/calibrations --raw
```

**Acceptance:** for canonical mean-H and mean-E, the inter/intra-instrument
ratio must be **< 1** with the 95 % CI upper bound < 1, and must be smaller than
the raw (`--raw`) ratio. ICC(2,1) is reported alongside.

## Stability (Maximum endpoint)

Re-image the kit after ≥ 1 week and re-derive. Report the stain-vector angle
between the two calibrations (`adept.calibration.stain.stain_angle_deg`), the
change in flat-field non-uniformity, and the change in focus reference. Drift
> 2° or > 5 % triggers re-calibration in deployment.
