#!/usr/bin/env python
"""S1-ID-3 (Minimum endpoint) reference experiment on a synthetic or real dataset.

1. Derive one calibration per instrument from its kit.
2. Measure the five criteria for every acquisition (canonical space).
3. Lock a site- and instrument-stratified slide split (or load an existing one).
4. Train the transparent CriterionCombiner on train, pick the threshold on val
   for the target sensitivity, and evaluate on the locked test set — overall
   and per instrument — with bootstrap 95 % CIs.
5. Leave-one-instrument-out: retrain with each instrument held out and test on it.
6. Per-criterion ablation.

Writes results/<name>/{features.csv, split.json, model.json, metrics.json}.

Usage (synthetic):
    adept synth make-dataset --out data/synth --n-slides 60
    python scripts/s1_train_combiner.py --data data/synth \
        --criteria configs/models/criteria_synthetic.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from adept.calibration import InstrumentCalibration, derive_calibration
from adept.criteria import CRITERIA, CriteriaConfig, assess_all
from adept.data import (
    LockedSplit,
    assert_no_slide_leakage,
    load_registry,
    loio_folds,
    make_stratified_split,
)
from adept.data.synthetic import load_kit, read_rgb
from adept.eval import TARGETS_ADEQUACY, meets_targets, per_group_metrics
from adept.eval.metrics import threshold_for_sensitivity
from adept.models import CriterionCombiner
from adept.models.adequacy import criteria_to_features


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data", type=Path, required=True, help="dataset root with registry.csv and kits/"
    )
    ap.add_argument("--criteria", type=Path, default=None, help="CriteriaConfig YAML")
    ap.add_argument("--split", type=Path, default=None, help="existing locked split JSON")
    ap.add_argument("--out", type=Path, default=Path("results/s1_combiner"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--target-sensitivity", type=float, default=TARGETS_ADEQUACY["sensitivity"])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = CriteriaConfig()
    if args.criteria:
        for k, v in (yaml.safe_load(args.criteria.read_text()) or {}).items():
            setattr(cfg, k, tuple(v) if isinstance(v, list) else v)

    records = load_registry(args.data / "registry.csv")
    instruments = sorted({r.instrument for r in records})

    # 1. calibrations
    cals: dict[str, InstrumentCalibration] = {}
    for inst in instruments:
        kit = load_kit(args.data / "kits" / inst)
        cals[inst] = derive_calibration(
            inst,
            kit["pure_h"],
            kit["pure_e"],
            kit["blank"],
            resolution_target=kit["target"][0] if kit["target"] else None,
        )
        cals[inst].save(args.out / "calibrations" / f"{inst}.json")
    print(
        "calibrations:",
        {
            k: {kk: round(vv, 2) for kk, vv in v.summary()["angle_to_canonical_deg"].items()}
            for k, v in cals.items()
        },
    )

    # 2. features
    feats: dict[str, np.ndarray] = {}
    with open(args.out / "features.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["acq_id", "slide_id", "instrument", "site", "label", *CRITERIA])
        for r in records:
            cal = cals[r.instrument]
            img = read_rgb(args.data / r.path)
            C, _ = cal.to_canonical(img)
            crit = assess_all(
                C, cal.correct_illumination(img), cfg, focus_reference=cal.focus_reference
            )
            x = criteria_to_features(crit)
            feats[r.acq_id] = x
            w.writerow([r.acq_id, r.slide_id, r.instrument, r.site, r.label, *np.round(x, 4)])
    print(f"features for {len(feats)} acquisitions")

    # 3. split
    if args.split:
        split = LockedSplit.load(args.split)
    else:
        split = make_stratified_split(records, seed=args.seed, name="s1_synthetic")
        split.save(args.out / "split.json")
    assert_no_slide_leakage(split, records)

    def xy(recs):
        X = np.stack([feats[r.acq_id] for r in recs])
        y = np.array([1 if r.is_inadequate else 0 for r in recs])
        g = np.array([r.instrument for r in recs])
        return X, y, g

    # 4. pooled canonical-space combiner
    Xtr, ytr, _ = xy(split.records_in("train", records))
    Xva, yva, _ = xy(split.records_in("val", records))
    Xte, yte, gte = xy(split.records_in("test", records))
    clf = CriterionCombiner().fit(Xtr, ytr)
    clf.threshold = threshold_for_sensitivity(
        yva, clf.predict_proba_inadequate(Xva), args.target_sensitivity
    )
    clf.save(args.out / "model.json")
    pooled = per_group_metrics(
        yte, clf.predict_proba_inadequate(Xte), gte, clf.threshold, args.n_boot, args.seed
    )
    results = {"threshold": clf.threshold, "pooled": {k: m.to_dict() for k, m in pooled.items()}}
    print("\n== pooled canonical combiner, locked test set ==")
    for k, m in pooled.items():
        print(
            f"{k:>12} n={m.n:3d}  sens {m.sensitivity}  spec {m.specificity}  "
            f"auroc {m.auroc}  κ {m.kappa}  targets={meets_targets(m)}"
        )

    # 5. LOIO
    results["loio"] = {}
    print("\n== leave-one-instrument-out ==")
    for fold in loio_folds(records, split):
        if not fold.test:
            continue
        Xa, ya, _ = xy(fold.train)
        Xv, yv, _ = xy(fold.val) if fold.val else (Xa, ya, None)
        Xt, yt, _ = xy(fold.test)
        m_ = CriterionCombiner().fit(Xa, ya)
        thr = threshold_for_sensitivity(
            yv, m_.predict_proba_inadequate(Xv), args.target_sensitivity
        )
        met = per_group_metrics(
            yt,
            m_.predict_proba_inadequate(Xt),
            np.array([fold.held_out_instrument] * len(yt)),
            thr,
            args.n_boot,
            args.seed,
        )["overall"]
        results["loio"][fold.held_out_instrument] = {**met.to_dict(), "threshold": thr}
        print(
            f"held out {fold.held_out_instrument:>12} n={met.n:3d}  sens {met.sensitivity}  "
            f"spec {met.specificity}  auroc {met.auroc}"
        )

    # 6. ablation (drop one criterion at a time)
    results["ablation"] = {}
    print("\n== per-criterion ablation (AUROC on test, pooled) ==")
    for drop in CRITERIA:
        use = tuple(c for c in CRITERIA if c != drop)
        idx = [CRITERIA.index(c) for c in use]
        m_ = CriterionCombiner(use=use).fit(Xtr[:, idx], ytr)
        met = per_group_metrics(
            yte, m_.predict_proba_inadequate(Xte[:, idx]), gte, 0.5, args.n_boot, args.seed
        )["overall"]
        results["ablation"][f"without_{drop}"] = met.to_dict()
        print(f"without {drop:>12}: auroc {met.auroc}")

    (args.out / "metrics.json").write_text(json.dumps(results, indent=1))
    print(f"\nwrote {args.out / 'metrics.json'}")


if __name__ == "__main__":
    main()
