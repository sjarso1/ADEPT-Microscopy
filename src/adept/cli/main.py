"""`adept` CLI.

Every command is a thin wrapper over an importable function, so the same
code path runs in notebooks, scripts and CI.

    adept synth make-dataset      synthetic multi-instrument dataset + kits
    adept calibrate derive        InstrumentCalibration from a kit directory
    adept calibrate validate      paired-slide invariance in canonical space
    adept assess run              AdequacyReport for one image
    adept splits make             locked, stratified, leak-free slide split
    adept eval loio               LOIO metrics with bootstrap CIs from a predictions CSV
    adept package write           output package for remote review
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import typer

from adept import INSTRUMENTS, __version__

app = typer.Typer(
    help="ADEPT — adequacy assessment that travels between microscopes.", no_args_is_help=True
)
synth_app = typer.Typer(help="Synthetic data for tests and dry runs.")
cal_app = typer.Typer(help="Instrument calibration and canonical space.")
assess_app = typer.Typer(help="Adequacy assessment.")
splits_app = typer.Typer(help="Locked data splits.")
eval_app = typer.Typer(help="Evaluation with bootstrap CIs.")
pkg_app = typer.Typer(help="Output packages.")
for name, sub in [
    ("synth", synth_app),
    ("calibrate", cal_app),
    ("assess", assess_app),
    ("splits", splits_app),
    ("eval", eval_app),
    ("package", pkg_app),
]:
    app.add_typer(sub, name=name)


@app.callback(invoke_without_command=True)
def _root(version: bool = typer.Option(False, "--version", help="Print version and exit.")):
    if version:
        typer.echo(f"adept {__version__}")
        raise typer.Exit()


# ------------------------------------------------------------------- synth
@synth_app.command("make-dataset")
def synth_make_dataset(
    out: Path = typer.Option(..., help="Output directory."),
    n_slides: int = typer.Option(24),
    instruments: str = typer.Option(",".join(INSTRUMENTS), help="Comma-separated instrument ids."),
    size: int = typer.Option(256),
    seed: int = typer.Option(0),
):
    from adept.data.synthetic import make_dataset

    records, params = make_dataset(out, n_slides, instruments.split(","), size, seed)
    n_b1 = sum(p.label == "B1" for p in params)
    typer.echo(f"wrote {len(records)} acquisitions of {n_slides} slides ({n_b1} B1) to {out}")


# --------------------------------------------------------------- calibrate
@cal_app.command("derive")
def calibrate_derive(
    kit: Path = typer.Option(
        ..., help="Kit directory with pure_h_*, pure_e_*, blank_*, target_* PNGs."
    ),
    instrument: str = typer.Option(...),
    out: Path = typer.Option(...),
    reference: Path | None = typer.Option(
        None, help="Reference-instrument calibration JSON to match stain scale."
    ),
):
    from adept.calibration import InstrumentCalibration, derive_calibration
    from adept.data.synthetic import load_kit

    k = load_kit(kit)
    ref_gain = None
    if reference is not None:
        ref = InstrumentCalibration.load(reference)
        ref_gain = np.asarray(ref.metadata.get("kit_median_concentration", [1, 1]), np.float32)
    cal = derive_calibration(
        instrument,
        k["pure_h"],
        k["pure_e"],
        k["blank"],
        resolution_target=k["target"][0] if k["target"] else None,
        reference_stain_gain=ref_gain,
        metadata={"kit_dir": str(kit)},
    )
    cal.save(out)
    typer.echo(json.dumps(cal.summary(), indent=1))


@cal_app.command("validate")
def calibrate_validate(
    pairs: Path = typer.Option(
        ..., help="CSV with slide_id, instrument, path (paired acquisitions)."
    ),
    calibrations: Path = typer.Option(..., help="Directory of <instrument>.json calibrations."),
    out: Path | None = typer.Option(None, help="Write the invariance table as JSON."),
    raw: bool = typer.Option(False, help="Also report the uncalibrated (raw OD) baseline."),
):
    """Inter- vs intra-instrument residuals of canonical-space statistics on paired slides."""
    from adept.calibration import InstrumentCalibration
    from adept.calibration.od import rgb_to_od
    from adept.data.synthetic import read_rgb
    from adept.eval import invariance_table

    root = pairs.parent
    cals = {p.stem: InstrumentCalibration.load(p) for p in calibrations.glob("*.json")}
    rows = list(csv.DictReader(open(pairs)))
    insts = sorted({r["instrument"] for r in rows} & set(cals))
    slides = sorted({r["slide_id"] for r in rows})
    by = {(r["slide_id"], r["instrument"]): r["path"] for r in rows}
    stats = {
        "mean_H": np.full((len(slides), len(insts)), np.nan),
        "mean_E": np.full((len(slides), len(insts)), np.nan),
    }
    raw_stats = {
        "raw_mean_OD_R": np.full_like(stats["mean_H"], np.nan),
        "raw_mean_OD_B": np.full_like(stats["mean_H"], np.nan),
    }
    for i, s in enumerate(slides):
        for j, k in enumerate(insts):
            if (s, k) not in by:
                continue
            img = read_rgb(root / by[(s, k)])
            C, _ = cals[k].to_canonical(img)
            tissue = C.sum(-1) > 0.15
            if tissue.sum() < 20:
                continue
            stats["mean_H"][i, j] = C[..., 0][tissue].mean()
            stats["mean_E"][i, j] = C[..., 1][tissue].mean()
            if raw:
                od = rgb_to_od(img)
                raw_stats["raw_mean_OD_R"][i, j] = od[..., 0][tissue].mean()
                raw_stats["raw_mean_OD_B"][i, j] = od[..., 2][tissue].mean()
    table = invariance_table({**stats, **(raw_stats if raw else {})})
    typer.echo(json.dumps(table, indent=1))
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(table, indent=1))


# ------------------------------------------------------------------ assess
@assess_app.command("run")
def assess_run(
    image: Path = typer.Option(...),
    calibration: Path = typer.Option(...),
    out: Path | None = typer.Option(None),
    slide_id: str | None = typer.Option(None),
    model: Path | None = typer.Option(None, help="CriterionCombiner JSON; default = heuristic."),
    criteria_config: Path | None = typer.Option(
        None, help="YAML overriding CriteriaConfig fields."
    ),
):
    from adept.calibration import InstrumentCalibration
    from adept.criteria import CriteriaConfig
    from adept.data.synthetic import read_rgb
    from adept.models import CriterionCombiner
    from adept.pipeline import AdequacyPipeline

    cfg = CriteriaConfig()
    if criteria_config:
        import yaml

        for k, v in (yaml.safe_load(criteria_config.read_text()) or {}).items():
            setattr(cfg, k, tuple(v) if isinstance(v, list) else v)
    clf = CriterionCombiner.load(model) if model else CriterionCombiner.heuristic()
    pipe = AdequacyPipeline(InstrumentCalibration.load(calibration), clf, cfg)
    rep = pipe.assess(read_rgb(image), slide_id or image.stem)
    js = rep.model_dump_json(indent=1)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(js)
    typer.echo(
        f"{rep.slide_id} [{rep.instrument}] → {rep.adequacy.value}  "
        f"P(B1)={rep.probability_inadequate:.2f}  reasons={[r.value for r in rep.reasons]}  "
        f"rois={len(rep.rois)}"
    )
    if not out:
        typer.echo(js)


# ------------------------------------------------------------------ splits
@splits_app.command("make")
def splits_make(
    registry: Path = typer.Option(...),
    out: Path = typer.Option(...),
    seed: int = typer.Option(0),
    train: float = typer.Option(0.6),
    val: float = typer.Option(0.15),
    test: float = typer.Option(0.25),
    name: str = typer.Option("split"),
):
    from adept.data import assert_no_slide_leakage, load_registry, make_stratified_split

    recs = load_registry(registry)
    split = make_stratified_split(recs, {"train": train, "val": val, "test": test}, seed, name=name)
    assert_no_slide_leakage(split, recs)
    split.save(out)
    typer.echo(
        f"locked split {split.name} (hash {split.content_hash}): "
        + ", ".join(f"{k}={len(v)} slides" for k, v in split.slides.items())
    )


# -------------------------------------------------------------------- eval
@eval_app.command("loio")
def eval_loio(
    registry: Path = typer.Option(...),
    predictions: Path = typer.Option(..., help="CSV: acq_id, y_score  (P(inadequate))."),
    threshold: float = typer.Option(0.5),
    n_boot: int = typer.Option(1000),
    out: Path | None = typer.Option(None),
):
    """Metrics overall and per instrument (the LOIO view when each instrument's
    predictions come from the model that held it out)."""
    from adept.data import load_registry
    from adept.eval import TARGETS_ADEQUACY, meets_targets, per_group_metrics

    recs = {r.acq_id: r for r in load_registry(registry)}
    preds = {r["acq_id"]: float(r["y_score"]) for r in csv.DictReader(open(predictions))}
    ids = [a for a in preds if a in recs and recs[a].label]
    y = np.array([1 if recs[a].is_inadequate else 0 for a in ids])
    s = np.array([preds[a] for a in ids])
    g = np.array([recs[a].instrument for a in ids])
    res = per_group_metrics(y, s, g, threshold, n_boot)
    table = {
        k: {**m.to_dict(), "targets": meets_targets(m, TARGETS_ADEQUACY)} for k, m in res.items()
    }
    for k, m in res.items():
        typer.echo(
            f"{k:>12}  n={m.n:4d}  sens {m.sensitivity}  spec {m.specificity}  "
            f"auroc {m.auroc}  κ {m.kappa}"
        )
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(table, indent=1))


# ----------------------------------------------------------------- package
@pkg_app.command("write")
def package_write(
    report: Path = typer.Option(..., help="AdequacyReport JSON from `adept assess run`."),
    image: Path = typer.Option(...),
    calibration: Path = typer.Option(...),
    out: Path = typer.Option(...),
):
    from adept.calibration import InstrumentCalibration
    from adept.data.synthetic import read_rgb
    from adept.pipeline import write_output_package
    from adept.pipeline.package import verify_package
    from adept.schema import AdequacyReport

    rep = AdequacyReport.model_validate_json(report.read_text())
    _, canonical_rgb = InstrumentCalibration.load(calibration).to_canonical(read_rgb(image))
    pkg = write_output_package(out, rep, canonical_rgb)
    typer.echo(json.dumps(verify_package(pkg)))


if __name__ == "__main__":
    app()
