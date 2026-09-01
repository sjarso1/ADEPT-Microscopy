import numpy as np
import pytest

from adept.data import (
    LockedSplit,
    assert_no_slide_leakage,
    load_registry,
    loio_folds,
    make_stratified_split,
)
from adept.data.registry import paired_slides
from adept.eval import (
    bootstrap_ci,
    cohens_kappa,
    diagnostic_metrics,
    paired_invariance,
    perturb,
    sweep_envelope,
)
from adept.eval.metrics import threshold_for_sensitivity


def test_split_is_slide_level_and_leak_free(synth_dir):
    recs = load_registry(synth_dir / "registry.csv")
    split = make_stratified_split(recs, seed=3)
    assert_no_slide_leakage(split, recs)
    # every acquisition of a slide lands in the same split
    for r in recs:
        assert split.split_of(r.slide_id) is not None
    total = sum(len(v) for v in split.slides.values())
    assert total == len({r.slide_id for r in recs})


def test_locked_split_detects_tampering(synth_dir, tmp_path):
    recs = load_registry(synth_dir / "registry.csv")
    split = make_stratified_split(recs, seed=3)
    p = split.save(tmp_path / "split.json")
    txt = p.read_text().replace('"seed": 3', '"seed": 4')
    p.write_text(txt)
    with pytest.raises(ValueError):
        LockedSplit.load(p)


def test_loio_folds_hold_out_instrument_completely(synth_dir):
    recs = load_registry(synth_dir / "registry.csv")
    split = make_stratified_split(recs, seed=5)
    for fold in loio_folds(recs, split):
        assert all(r.instrument != fold.held_out_instrument for r in fold.train + fold.val)
        assert all(r.instrument == fold.held_out_instrument for r in fold.test)
        test_slides = {r.slide_id for r in fold.test}
        assert test_slides.isdisjoint({r.slide_id for r in fold.train})


def test_paired_slides_all_instruments(synth_dir):
    recs = load_registry(synth_dir / "registry.csv")
    assert len(paired_slides(recs)) == len({r.slide_id for r in recs})


def test_kappa_perfect_and_chance():
    a = np.array([0, 1, 0, 1, 1, 0])
    assert cohens_kappa(a, a) == pytest.approx(1.0)
    assert abs(cohens_kappa(a, 1 - a) - (-1.0)) < 1e-9


def test_diagnostic_metrics_perfect_classifier():
    y = np.array([0] * 20 + [1] * 20)
    s = np.array([0.1] * 20 + [0.9] * 20)
    m = diagnostic_metrics(y, s, 0.5, n_boot=200)
    assert m.sensitivity.estimate == 1.0 and m.specificity.estimate == 1.0
    assert m.auroc.estimate == 1.0 and m.kappa.estimate == 1.0
    assert m.sensitivity.lower <= m.sensitivity.estimate <= m.sensitivity.upper


def test_bootstrap_ci_brackets_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = np.clip(y * 0.6 + rng.normal(0, 0.3, 200), 0, 1)
    m = diagnostic_metrics(y, s, 0.5, n_boot=300)
    for k in ("sensitivity", "specificity", "auroc", "kappa"):
        ci = getattr(m, k)
        assert ci.lower <= ci.estimate <= ci.upper
        assert ci.upper - ci.lower < 0.5


def test_threshold_for_sensitivity():
    y = np.array([1] * 10 + [0] * 10)
    s = np.concatenate([np.linspace(0.3, 0.9, 10), np.linspace(0.0, 0.4, 10)])
    thr = threshold_for_sensitivity(y, s, 0.9)
    assert ((s[y == 1] >= thr).mean()) >= 0.9


def test_bootstrap_ci_generic():
    ci = bootstrap_ci(lambda i: float(np.mean(np.arange(10)[i])), np.ones(10), n_boot=100)
    assert ci.lower <= 4.5 <= ci.upper


def test_paired_invariance_detects_instrument_effect():
    rng = np.random.default_rng(0)
    slide_effect = rng.normal(0, 1, (30, 1))
    invariant = slide_effect + rng.normal(0, 0.05, (30, 4))
    variant = slide_effect + np.array([[0, 1, 2, 3]]) + rng.normal(0, 0.05, (30, 4))
    a, b = paired_invariance(invariant, n_boot=100), paired_invariance(variant, n_boot=100)
    assert a["invariant"] and a["ratio"] < 0.2
    assert not b["invariant"] and b["icc_2_1"] < a["icc_2_1"]


@pytest.mark.parametrize("axis", ["stain", "focus", "illumination"])
def test_perturbations_change_image_monotonically(axis, good_slide):
    from adept.calibration import canonical_rgb_from_concentrations

    rgb = canonical_rgb_from_concentrations(good_slide)
    d = [np.abs(perturb(rgb, axis, lv) - rgb).mean() for lv in (0.0, 0.3, 0.8)]
    assert d[0] == 0.0 and d[1] < d[2]


def test_envelope_sweep_finds_boundary():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 120)

    def score_fn(level):
        noise = rng.normal(0, 0.1 + level, 120)
        return y, np.clip(0.5 + (y - 0.5) * 0.8 + noise, 0, 1)

    res = sweep_envelope("focus", [0.0, 0.5, 1.0, 2.0, 4.0], score_fn, n_boot=50)
    assert res.boundary_level is not None
    assert len(res.metrics) == 5
