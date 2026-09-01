"""Slide registry, leak-free splits, LOIO folds and synthetic data."""

from adept.data.registry import SlideRecord, load_registry, save_registry
from adept.data.splits import (
    LockedSplit,
    assert_no_slide_leakage,
    loio_folds,
    make_stratified_split,
)

__all__ = [
    "LockedSplit",
    "SlideRecord",
    "assert_no_slide_leakage",
    "load_registry",
    "loio_folds",
    "make_stratified_split",
    "save_registry",
]
