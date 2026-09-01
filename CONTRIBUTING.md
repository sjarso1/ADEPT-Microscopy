# Contributing to ADEPT

Thank you for helping build an adequacy-assessment system that travels between
microscopes. This document covers the engineering conventions that keep the
project reproducible and the results trustworthy.

## Ground rules

1. **No patient data in git.** Images, labels, pathology reports, and slide
   metadata that can identify a case live under `data/` (git-ignored) or on
   the project object store. See `docs/data_governance.md`.
2. **Locked splits are immutable.** A split file under `data/splits/` that has
   been used to report a number is never edited. Create a new, versioned split
   and record the reason in `docs/decisions/`.
3. **No physical slide crosses a split.** Splits are made at the *slide* level
   and inherited by every instrument's acquisition of that slide. The split
   utilities enforce this; do not bypass them.
4. **Raw instrument colour never reaches a model.** Everything downstream of
   `adept.calibration` operates in canonical space. Staining quality
   (Criterion 3) is the one deliberate exception — it is measured from
   calibrated optical densities *before* normalisation.
5. **Every reported number has an error bar.** Use `adept.eval.metrics`
   (bootstrap CIs) — do not hand-compute point estimates for reports.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # + ,torch for the DL stack
pre-commit install
pytest
```

## Branching and pull requests

- `main` is always releasable. Work on branches named `s<semester>/<id>-<topic>`,
  e.g. `s1/id2-focus-map`, or `fix/<topic>`.
- Open a PR early; CI runs `ruff`, `pytest`, and a build check.
- Each interim deliverable (S1-ID-1, …, S3-ID-3) and each final report is
  tagged: `s1-id1`, `s1-final`, and so on. Tags are annotated and point at the
  exact commit whose code produced the numbers in the report.

## Code style

- Python ≥ 3.10, type-hinted, `ruff`-clean (line length 100).
- Public functions carry a docstring that states **units** and **array
  layouts** (`HxWx3 float32 in [0,1]`, `HxW OD`, …).
- Prefer pure functions over stateful classes for measurements. Calibration
  state is carried by `InstrumentCalibration`, nothing else.
- Tests run on synthetic data (`adept.data.synthetic`) and must not need a GPU.

## Experiment discipline

- One experiment = one YAML under `configs/experiments/` + one script under
  `scripts/`. The script writes a results JSON with the config hash embedded.
- Log runs to Weights & Biases (academic tier) with the same config hash.
- Record every architecture / scope / threshold decision as an ADR in
  `docs/decisions/` using the template there.

## Reporting

Interim and final reports are written by the student (see the course
agreements' academic-integrity clause). Code, configs, splits, weights and
logs referenced by a report must be in the tagged commit.
