# 0001 — Default canonical stain matrix

**Date:** 2026-09-01 · **Deliverable:** S1-ID-1 · **Status:** accepted (provisional)

## Context
The canonical image space needs a target H and E stain-vector pair. The
course agreement requires the canonical space to be "justified by measured
spectral differences rather than borrowed constants".

## Decision
Until S1-ID-1 measures the project's instruments, `CANONICAL_STAIN_MATRIX`
is the Ruifrok & Johnston (2001) H&E reference, unit-normalised. S1-ID-1
will replace it with a matrix chosen from the measured per-instrument
vectors (e.g. the reference scanner's, or the mean over instruments) and
supersede this ADR.

## Evidence
None yet beyond literature precedent; the value is a placeholder by design.

## Consequences
All canonical-space measurements before the replacement must be re-run
after it; no reported number may span the change.
