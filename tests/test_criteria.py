import numpy as np

from adept.calibration import canonical_rgb_from_concentrations
from adept.criteria import CRITERIA, CriteriaConfig, assess_all
from adept.criteria.artifact import HeuristicArtifactDetector
from adept.criteria.focus import normalized_sharpness
from adept.criteria.tissue import detect_cores, tissue_mask_from_concentrations
from adept.data.synthetic import SlideParams, render_canonical_slide

CFG = CriteriaConfig(
    microns_per_pixel=10.0,
    tissue_min_total_length_um=1500,
    tissue_min_core_length_um=800,
    cellularity_min_nuclei_per_mm2=40,
    staining_he_ratio_range=(0.2, 3.0),
    staining_min_h_od=0.1,
    focus_tile_px=32,
    focus_min_relative_sharpness=0.25,
)


def _run(C):
    rgb = canonical_rgb_from_concentrations(C)
    return {c.name: c for c in assess_all(C, rgb, CFG)}


def test_all_five_criteria_present_and_valid(good_slide):
    res = _run(good_slide)
    assert set(res) == set(CRITERIA)
    for c in res.values():
        assert 0.0 <= c.score <= 1.0


def test_good_slide_passes_everything(good_slide):
    res = _run(good_slide)
    failed = [n for n, c in res.items() if not c.passed]
    assert failed == [], {n: res[n].model_dump(exclude={"details"}) for n in failed}


def test_empty_slide_fails_tissue(empty_slide):
    res = _run(empty_slide)
    assert not res["tissue"].passed
    assert res["tissue"].details["core_count"] == 0


def test_core_detection_counts_cores(good_slide):
    mask = tissue_mask_from_concentrations(good_slide)
    cores = detect_cores(mask, microns_per_pixel=10.0, min_core_length_um=800)
    assert len(cores) == 3
    assert all(c["length_um"] > 1000 for c in cores)


def test_low_cellularity_is_detected(rng):
    sparse = render_canonical_slide(
        SlideParams("S", 3, 150, 1.5, 1.0, 0.8, False, False, 0.0, "B1"), rng, 192
    )
    dense = render_canonical_slide(
        SlideParams("D", 3, 150, 9.0, 1.0, 0.8, False, False, 0.0, "B2+"), rng, 192
    )
    assert _run(sparse)["cellularity"].value < _run(dense)["cellularity"].value


def test_weak_haematoxylin_lowers_staining_score(rng):
    weak = render_canonical_slide(
        SlideParams("W", 3, 150, 8.0, 0.2, 0.9, False, False, 0.0, "B1"), rng, 192
    )
    good = render_canonical_slide(
        SlideParams("G", 3, 150, 8.0, 1.0, 0.8, False, False, 0.0, "B2+"), rng, 192
    )
    assert _run(weak)["staining"].score < _run(good)["staining"].score


def test_blur_reduces_focus_and_normalised_sharpness(rng):
    sharp = render_canonical_slide(
        SlideParams("F", 3, 150, 8.0, 1.0, 0.8, False, False, 0.0, "B2+"), rng, 192
    )
    blurred = render_canonical_slide(
        SlideParams("B", 3, 150, 8.0, 1.0, 0.8, False, False, 4.0, "B1"), rng, 192
    )
    assert normalized_sharpness(blurred[..., 0]) < 0.3 * normalized_sharpness(sharp[..., 0])
    assert _run(blurred)["focus"].value < _run(sharp)["focus"].value


def test_heuristic_artifact_detector_finds_fold(rng):
    folded = render_canonical_slide(
        SlideParams("A", 2, 160, 6.0, 1.0, 0.8, True, False, 0.0, "B1"), rng, 192
    )
    rgb = canonical_rgb_from_concentrations(folded)
    mask = HeuristicArtifactDetector().predict_mask(rgb)
    assert (mask == 1).sum() > 50


def test_criteria_are_instrument_invariant_on_paired_slide(calibrations, good_slide, rng):
    from adept.data.synthetic import DEFAULT_INSTRUMENTS

    vals = {}
    for name in ("openflexure", "scanner", "wsi"):
        img = DEFAULT_INSTRUMENTS[name].acquire(good_slide, rng)
        cal = calibrations[name]
        C, _ = cal.to_canonical(img)
        res = {
            c.name: c
            for c in assess_all(C, cal.correct_illumination(img), CFG, cal.focus_reference)
        }
        vals[name] = res
    for crit in ("tissue", "cellularity", "staining"):
        v = np.array([vals[n][crit].value for n in vals])
        assert v.std() / max(v.mean(), 1e-6) < 0.25, (crit, v)
