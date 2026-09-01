import numpy as np
import pytest

from adept.calibration import (
    CANONICAL_STAIN_MATRIX,
    InstrumentCalibration,
    deconvolve,
    derive_flatfield,
    estimate_stain_vectors_macenko,
    od_to_rgb,
    rgb_to_od,
)
from adept.calibration.stain import reconstruct_od, stain_angle_deg
from adept.data.synthetic import DEFAULT_INSTRUMENTS


def test_od_roundtrip():
    rgb = np.random.default_rng(0).uniform(0.05, 1.0, (8, 8, 3)).astype(np.float32)
    assert np.allclose(od_to_rgb(rgb_to_od(rgb)), rgb, atol=1e-4)


def test_od_of_white_is_zero_and_uint8_ok():
    white = np.full((4, 4, 3), 255, np.uint8)
    assert np.allclose(rgb_to_od(white), 0.0)


def test_deconvolution_recovers_concentrations():
    rng = np.random.default_rng(1)
    C = rng.uniform(0, 1.5, (16, 16, 2)).astype(np.float32)
    od = reconstruct_od(C, CANONICAL_STAIN_MATRIX)
    assert np.allclose(deconvolve(od, CANONICAL_STAIN_MATRIX), C, atol=1e-4)


def test_macenko_finds_canonical_vectors_on_synthetic_he():
    rng = np.random.default_rng(2)
    C = np.zeros((64, 64, 2), np.float32)
    C[:32, :, 0] = rng.uniform(0.5, 1.5, (32, 64))  # H-dominant region
    C[32:, :, 1] = rng.uniform(0.5, 1.5, (32, 64))  # E-dominant region
    rgb = od_to_rgb(reconstruct_od(C, CANONICAL_STAIN_MATRIX))
    M = estimate_stain_vectors_macenko(rgb, beta=0.1, alpha=1.0)
    ang = stain_angle_deg(M, CANONICAL_STAIN_MATRIX)
    assert ang.max() < 5.0, ang


def test_flatfield_corrects_vignetting():
    inst = DEFAULT_INSTRUMENTS["printed3d"]
    rng = np.random.default_rng(3)
    blank = [inst.acquire(np.zeros((96, 96, 2), np.float32), rng) for _ in range(3)]
    flat = derive_flatfield(blank, smooth_sigma=10)
    corrected = blank[0] / flat
    assert abs(corrected.mean() - 1.0) < 0.05
    assert corrected.std() < 0.05


@pytest.mark.parametrize("name", list(DEFAULT_INSTRUMENTS))
def test_kit_calibration_recovers_instrument_stain_matrix(calibrations, name):
    true = DEFAULT_INSTRUMENTS[name].stain_matrix()
    err = stain_angle_deg(true, calibrations[name].stain_matrix)
    assert err.max() < 2.0, f"{name}: stain-vector error {err} deg"


def test_calibration_roundtrip_json(calibrations, tmp_path):
    cal = calibrations["openflexure"]
    p = cal.save(tmp_path / "of.json")
    back = InstrumentCalibration.load(p)
    assert back.instrument == "openflexure"
    assert np.allclose(back.stain_matrix, cal.stain_matrix)
    assert np.allclose(back.flatfield, cal.flatfield)
    assert back.focus_reference == pytest.approx(cal.focus_reference)


def test_paired_slide_agrees_in_canonical_space(calibrations, good_slide, rng):
    """The falsifiable claim: the same slide on two instruments must agree after
    calibration far better than before."""
    a, b = "openflexure", "scanner"
    img_a = DEFAULT_INSTRUMENTS[a].acquire(good_slide, rng)
    img_b = DEFAULT_INSTRUMENTS[b].acquire(good_slide, rng)
    _, canon_a = calibrations[a].to_canonical(img_a)
    _, canon_b = calibrations[b].to_canonical(img_b)
    tissue = good_slide.sum(-1) > 0.2
    # compare slide-level tissue colour (pixel-wise residuals are dominated by
    # the low-cost instrument's optical blur, which calibration does not touch)
    after = np.abs(canon_a[tissue].mean(0) - canon_b[tissue].mean(0)).max()
    before = np.abs(img_a[tissue].mean(0) - img_b[tissue].mean(0)).max()
    assert after < 0.3 * before, (before, after)


def test_identity_calibration_is_noop_on_canonical_rgb(good_slide):
    from adept.calibration import canonical_rgb_from_concentrations

    rgb = canonical_rgb_from_concentrations(good_slide)
    C, _ = InstrumentCalibration.identity().to_canonical(rgb)
    assert np.allclose(C, good_slide, atol=0.05)
