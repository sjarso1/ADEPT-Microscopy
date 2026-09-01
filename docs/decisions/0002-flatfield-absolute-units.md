# 0002 — Flat-field kept in absolute units, lightly smoothed

**Date:** 2026-09-01 · **Deliverable:** S1-ID-1 · **Status:** accepted

## Context
Two implementation choices for the illumination flat-field were tried on the
synthetic multi-instrument data: (a) a median-normalised shape-only flat plus
a separate white point; (b) the smoothed blank in absolute units, i.e. the
Beer–Lambert `I0(x, y)` itself. Smoothing strength was also examined.

## Decision
(b), with Gaussian σ = 5 px. White balance is thereby absorbed into the flat;
the separate spectral gain is retained only for flat-less operation.

## Evidence
With (a), a background OD offset of ≈0.05 leaked into the canonical H channel
on the strongly vignetted instruments and tilted the eosin vector estimate by
≈9°; with (b) the estimate error fell below 2° on every instrument. With
σ = 25 px the vignette itself was blurred, leaving a ≈0.03 mean colour
residual between paired acquisitions; σ = 5 px reduced it to ≈0.003
(blur-free) — see `tests/test_calibration.py::test_paired_slide_agrees_in_canonical_space`.

## Consequences
Blank frames must not saturate (kit protocol step 2). Dust is handled by the
median across frames, not by smoothing.
