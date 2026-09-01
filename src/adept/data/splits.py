"""Locked, stratified, leak-free data splits and leave-one-instrument-out folds.

Rules enforced here (and tested in ``tests/test_splits.py``):

1. Splitting happens at the **physical-slide** level. Every acquisition of a
   slide inherits the slide's split, so the same tissue can never appear in
   train via one instrument and in test via another.
2. Stratification is by **site × label** (and instrument coverage), so each
   split has comparable prevalence and site mix.
3. A split is **locked**: it is saved with a content hash and the seed, and
   :func:`assert_no_slide_leakage` is run before any metric is reported.
4. LOIO folds are built *on top of* the locked split: for held-out instrument
   *k*, train on the train-split acquisitions of every other instrument and
   test on the test-split acquisitions of instrument *k* (and optionally its
   train-split acquisitions too, since the model never saw them — configurable).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from adept.data.registry import SlideRecord


@dataclass
class LockedSplit:
    name: str
    seed: int
    fractions: dict[str, float]
    slides: dict[str, list[str]]  # split name -> sorted slide ids
    stratify_on: list[str]
    content_hash: str = ""
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def split_of(self, slide_id: str) -> str | None:
        for name, ids in self.slides.items():
            if slide_id in ids:
                return name
        return None

    def records_in(self, split: str, records: Iterable[SlideRecord]) -> list[SlideRecord]:
        ids = set(self.slides[split])
        return [r for r in records if r.slide_id in ids]

    def compute_hash(self) -> str:
        payload = json.dumps(
            {"seed": self.seed, "fractions": self.fractions, "slides": self.slides}, sort_keys=True
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.content_hash = self.compute_hash()
        path.write_text(json.dumps(self.__dict__, indent=1, sort_keys=True))
        return path

    @classmethod
    def load(cls, path: str | Path, verify: bool = True) -> LockedSplit:
        d = json.loads(Path(path).read_text())
        s = cls(**d)
        if verify and s.content_hash and s.content_hash != s.compute_hash():
            raise ValueError(f"split file {path} has been modified since it was locked")
        return s


def _strata(records: list[SlideRecord], keys: list[str]) -> dict[str, str]:
    """slide_id -> stratum string built from slide-level attributes."""
    out: dict[str, str] = {}
    for r in records:
        parts = []
        for k in keys:
            v = getattr(r, k, "")
            parts.append(str(v))
        out.setdefault(r.slide_id, "|".join(parts))
    return out


def make_stratified_split(
    records: list[SlideRecord],
    fractions: dict[str, float] | None = None,
    seed: int = 0,
    stratify_on: list[str] | None = None,
    name: str = "split",
) -> LockedSplit:
    """Slide-level stratified split.

    Parameters
    ----------
    fractions : e.g. ``{"train": 0.6, "val": 0.15, "test": 0.25}``; must sum to 1.
    stratify_on : slide-level attributes (default ``["site", "label"]``).
    """
    fractions = fractions or {"train": 0.6, "val": 0.15, "test": 0.25}
    if abs(sum(fractions.values()) - 1.0) > 1e-6:
        raise ValueError("fractions must sum to 1")
    stratify_on = stratify_on or ["site", "label"]
    strata = _strata(records, stratify_on)
    rng = np.random.default_rng(seed)
    names = list(fractions)
    cum = np.cumsum([fractions[n] for n in names])
    out: dict[str, list[str]] = {n: [] for n in names}
    by_stratum: dict[str, list[str]] = {}
    for sid, st in strata.items():
        by_stratum.setdefault(st, []).append(sid)
    for _, sids in sorted(by_stratum.items()):
        sids = sorted(sids)
        rng.shuffle(sids)
        n = len(sids)
        bounds = np.round(cum * n).astype(int)
        start = 0
        for nm, b in zip(names, bounds, strict=True):
            out[nm].extend(sids[start:b])
            start = b
    split = LockedSplit(
        name=name,
        seed=seed,
        fractions=fractions,
        slides={k: sorted(v) for k, v in out.items()},
        stratify_on=stratify_on,
    )
    split.content_hash = split.compute_hash()
    assert_no_slide_leakage(split)
    return split


def assert_no_slide_leakage(split: LockedSplit, records: list[SlideRecord] | None = None) -> None:
    """Raise if any slide id appears in more than one split, or (with records)
    if acquisitions of one slide would land in different splits."""
    seen: dict[str, str] = {}
    for name, ids in split.slides.items():
        for s in ids:
            if s in seen:
                raise AssertionError(f"slide {s} in both {seen[s]} and {name}")
            seen[s] = name
    if records is not None:
        for r in records:
            if r.slide_id not in seen:
                raise AssertionError(f"acquisition {r.acq_id}: slide {r.slide_id} not in split")


@dataclass
class LOIOFold:
    held_out_instrument: str
    train: list[SlideRecord]
    val: list[SlideRecord]
    test: list[SlideRecord]


def loio_folds(
    records: list[SlideRecord],
    split: LockedSplit,
    instruments: list[str] | None = None,
    test_on_all_heldout_acquisitions: bool = False,
) -> list[LOIOFold]:
    """Leave-one-instrument-out folds layered on a locked slide split.

    For each held-out instrument *k*:
      train = train-split acquisitions on instruments ≠ k
      val   = val-split acquisitions on instruments ≠ k
      test  = test-split acquisitions on instrument k
              (or *all* acquisitions on k if ``test_on_all_heldout_acquisitions``;
              legitimate because no image of k was used for training — but the
              *tissue* of train-split slides was seen via other instruments, so
              the default is the stricter test-split-only choice)
    """
    instruments = instruments or sorted({r.instrument for r in records})
    folds = []
    for k in instruments:
        train = [r for r in split.records_in("train", records) if r.instrument != k]
        val = (
            [r for r in split.records_in("val", records) if r.instrument != k]
            if "val" in split.slides
            else []
        )
        if test_on_all_heldout_acquisitions:
            test = [r for r in records if r.instrument == k]
        else:
            test = [r for r in split.records_in("test", records) if r.instrument == k]
        folds.append(LOIOFold(k, train, val, test))
    return folds
