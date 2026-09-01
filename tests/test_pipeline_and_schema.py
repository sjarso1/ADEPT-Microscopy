import json

import numpy as np
import pytest
from pydantic import ValidationError

from adept.criteria import CriteriaConfig
from adept.data.synthetic import DEFAULT_INSTRUMENTS
from adept.models import CriterionCombiner, rank_rois
from adept.pipeline import AdequacyPipeline, write_output_package
from adept.pipeline.package import verify_package
from adept.schema import (
    SCHEMA_VERSION,
    AdequacyClass,
    AdequacyReport,
    CriterionResult,
    RegionOfInterest,
)

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


def test_schema_rejects_duplicate_criteria():
    c = CriterionResult(name="tissue", value=1, score=1, passed=True)
    with pytest.raises(ValidationError):
        AdequacyReport(
            slide_id="s",
            instrument="i",
            adequacy=AdequacyClass.B2_PLUS_ADEQUATE,
            probability_inadequate=0.1,
            decision_threshold=0.5,
            criteria=[c, c],
        )


def test_schema_roundtrip_json():
    rep = AdequacyReport(
        slide_id="s",
        instrument="openflexure",
        adequacy=AdequacyClass.B1_INADEQUATE,
        probability_inadequate=0.9,
        decision_threshold=0.5,
        rois=[RegionOfInterest(rank=1, x=0, y=0, width=64, height=64, score=0.8)],
    )
    back = AdequacyReport.model_validate_json(rep.model_dump_json())
    assert back == rep and back.schema_version == SCHEMA_VERSION


def test_suspicion_field_is_optional_for_backward_compatibility():
    d = json.loads(
        RegionOfInterest(rank=1, x=0, y=0, width=1, height=1, score=0.5).model_dump_json()
    )
    assert d["suspicion_score"] is None


def test_rank_rois_rescans_come_first_and_ranks_are_contiguous():
    tissue = np.ones((4, 4), bool)
    sharp = np.full((4, 4), 0.9)
    sharp[0, 0] = 0.1  # one poor tile
    rois = rank_rois(tissue, sharp, 32, max_capture=5, max_rescan=2)
    assert rois[0].action == "refocus"
    assert [r.rank for r in rois] == list(range(1, len(rois) + 1))
    assert sum(r.action == "capture_hr" for r in rois) == 5


def test_rank_rois_suspicion_reorders_capture_list():
    tissue = np.ones((2, 2), bool)
    sharp = np.full((2, 2), 0.9)
    susp = np.zeros((2, 2))
    susp[1, 1] = 1.0
    rois = rank_rois(tissue, sharp, 16, suspicion_map=susp, max_capture=4, max_rescan=0)
    assert (rois[0].x, rois[0].y) == (16, 16)
    assert rois[0].suspicion_score == 1.0


def test_pipeline_end_to_end_and_calibration_swap(
    calibrations, good_slide, empty_slide, rng, tmp_path
):
    pipe = AdequacyPipeline(calibrations["openflexure"], CriterionCombiner.heuristic(), CFG)
    img = DEFAULT_INSTRUMENTS["openflexure"].acquire(good_slide, rng)
    rep = pipe.assess(img, "GOOD")
    assert rep.adequacy == AdequacyClass.B2_PLUS_ADEQUATE
    assert len(rep.rois) > 0 and rep.instrument == "openflexure"

    img2 = DEFAULT_INSTRUMENTS["openflexure"].acquire(empty_slide, rng)
    rep2 = pipe.assess(img2, "EMPTY")
    assert rep2.adequacy == AdequacyClass.B1_INADEQUATE
    assert "no_tissue" in [r.value for r in rep2.reasons]

    # onboarding = swap the calibration front-end only
    pipe_sc = pipe.with_calibration(calibrations["scanner"])
    rep3 = pipe_sc.assess(DEFAULT_INSTRUMENTS["scanner"].acquire(good_slide, rng), "GOOD")
    assert rep3.adequacy == AdequacyClass.B2_PLUS_ADEQUATE and rep3.instrument == "scanner"

    # package + QC
    _, canon = calibrations["openflexure"].to_canonical(img)
    pkg = write_output_package(tmp_path / "pk", rep, canon)
    qc = verify_package(pkg)
    assert qc["ok"], qc
    with pytest.raises(FileExistsError):
        write_output_package(tmp_path / "pk", rep, canon)


def test_combiner_fit_save_load(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (80, 5))
    y = (X.min(axis=1) < 0.3).astype(int)
    clf = CriterionCombiner().fit(X, y)
    p = clf.save(tmp_path / "m.json")
    back = CriterionCombiner.load(p)
    assert np.allclose(back.predict_proba_inadequate(X), clf.predict_proba_inadequate(X))


def test_cli_help_runs():
    from typer.testing import CliRunner

    from adept.cli.main import app

    r = CliRunner().invoke(app, ["--help"])
    assert r.exit_code == 0 and "calibrate" in r.output
