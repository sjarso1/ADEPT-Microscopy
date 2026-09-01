"""The slide registry: one row per *acquisition* (physical slide × instrument).

The registry is the single source of truth joining images, instruments, sites
and labels. Labels attach to the **physical slide**, so every acquisition of a
slide shares its label — this is what lets one annotation serve every
instrument, and what makes slide-level splitting mandatory.

CSV columns (``registry.csv``)
-------------------------------
acq_id        unique acquisition id
slide_id      physical slide id (shared across instruments)
instrument    one of adept.INSTRUMENTS (or a registered new instrument)
site          acquisition site (e.g. "mulago", "mbarara", "jhu")
path          image path relative to the data root
label         adequacy reference label: "B1" or "B2+" (may be empty pre-annotation)
diagnosis     Semester 3: B-category from the clinical report (B1..B5b) or empty
paired_set    optional id grouping acquisitions imaged in the same paired session
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class SlideRecord:
    acq_id: str
    slide_id: str
    instrument: str
    site: str
    path: str
    label: str = ""
    diagnosis: str = ""
    paired_set: str = ""

    @property
    def is_inadequate(self) -> bool | None:
        if self.label == "":
            return None
        return self.label.strip().upper() == "B1"


def load_registry(path: str | Path) -> list[SlideRecord]:
    names = {f.name for f in fields(SlideRecord)}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [SlideRecord(**{k: (v or "") for k, v in r.items() if k in names}) for r in rows]


def save_registry(records: list[SlideRecord], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[fl.name for fl in fields(SlideRecord)])
        w.writeheader()
        for r in records:
            w.writerow(asdict(r))
    return path


def paired_slides(records: list[SlideRecord], instruments: list[str] | None = None) -> list[str]:
    """Slide ids imaged on *all* of the given instruments (default: all present)."""
    by_slide: dict[str, set[str]] = {}
    for r in records:
        by_slide.setdefault(r.slide_id, set()).add(r.instrument)
    if instruments is None:
        instruments = sorted({r.instrument for r in records})
    want = set(instruments)
    return sorted(s for s, ins in by_slide.items() if want <= ins)
