"""Shared fixtures: a small synthetic multi-instrument dataset built once per session."""

from __future__ import annotations

import numpy as np
import pytest

from adept.calibration import InstrumentCalibration, derive_calibration
from adept.data.synthetic import (
    DEFAULT_INSTRUMENTS,
    SlideParams,
    make_dataset,
    render_canonical_slide,
    render_kit,
)


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(123)


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth")
    make_dataset(out, n_slides=16, size=192, seed=1)
    return out


@pytest.fixture(scope="session")
def kits(rng):
    return {name: render_kit(inst, rng, size=160) for name, inst in DEFAULT_INSTRUMENTS.items()}


@pytest.fixture(scope="session")
def calibrations(kits) -> dict[str, InstrumentCalibration]:
    return {
        name: derive_calibration(
            name, k["pure_h"], k["pure_e"], k["blank"], resolution_target=k["target"][0]
        )
        for name, k in kits.items()
    }


@pytest.fixture(scope="session")
def good_slide(rng):
    """A clearly adequate canonical slide: 3 cores, cellular, well stained, sharp."""
    p = SlideParams("GOOD", 3, 150, 8.0, 1.0, 0.8, False, False, 0.0, "B2+")
    return render_canonical_slide(p, rng, size=192)


@pytest.fixture(scope="session")
def empty_slide(rng):
    p = SlideParams("EMPTY", 0, 150, 8.0, 1.0, 0.8, False, False, 0.0, "B1", ["no_tissue"])
    return render_canonical_slide(p, rng, size=192)
