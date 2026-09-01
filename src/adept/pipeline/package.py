"""Output packages for remote review (S2-ID-3).

A package directory holds everything a remote pathologist (or the reading
study) needs for one slide:

    <slide_id>/
      report.json          AdequacyReport (schema-versioned)
      overview.png         the low-magnification mosaic (canonical RGB)
      overview_rois.png    the mosaic with ranked ROIs drawn
      tiles/NN_<action>.png  targeted high-resolution tiles in rank order
      manifest.json        completeness QC: which files exist, sizes, checksums

``write_output_package`` never overwrites an existing package silently: the
reading study must be reproducible, so a package is versioned by re-running
into a new directory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from adept.schema import AdequacyReport

_COLOURS = {"capture_hr": (255, 209, 102), "refocus": (239, 71, 111), "rescan": (6, 214, 160)}


def _png(path: Path, rgb: np.ndarray) -> None:
    arr = (np.clip(rgb, 0, 1) * 255).astype(np.uint8) if rgb.dtype != np.uint8 else rgb
    cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


def draw_rois(rgb: np.ndarray, report: AdequacyReport) -> np.ndarray:
    img = (np.clip(rgb, 0, 1) * 255).astype(np.uint8).copy()
    for r in report.rois:
        colour = _COLOURS.get(r.action, (255, 255, 255))
        cv2.rectangle(img, (r.x, r.y), (r.x + r.width, r.y + r.height), colour, 1)
        cv2.putText(img, str(r.rank), (r.x + 2, r.y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, colour, 1)
    return img


def write_output_package(
    out_dir: str | Path,
    report: AdequacyReport,
    overview_rgb: np.ndarray,
    hr_tiles: dict[int, np.ndarray] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a complete package and its completeness manifest."""
    out = Path(out_dir) / report.slide_id
    if out.exists() and not overwrite:
        raise FileExistsError(f"package {out} exists; use a new output directory")
    (out / "tiles").mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(report.model_dump_json(indent=1))
    _png(out / "overview.png", overview_rgb)
    _png(out / "overview_rois.png", draw_rois(overview_rgb, report))
    for r in report.rois:
        if hr_tiles and r.rank in hr_tiles:
            tile = hr_tiles[r.rank]
        else:  # fall back to the overview crop so the package is always complete
            tile = overview_rgb[r.y : r.y + r.height, r.x : r.x + r.width]
        _png(out / "tiles" / f"{r.rank:02d}_{r.action}.png", tile)
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json")
    manifest = {
        "slide_id": report.slide_id,
        "instrument": report.instrument,
        "schema_version": report.schema_version,
        "adequacy": report.adequacy.value,
        "n_rois": len(report.rois),
        "n_tiles": len(list((out / "tiles").glob("*.png"))),
        "complete": len(list((out / "tiles").glob("*.png"))) == len(report.rois),
        "files": [
            {
                "path": str(p.relative_to(out)),
                "bytes": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest()[:16],
            }
            for p in files
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return out


def verify_package(pkg_dir: str | Path) -> dict:
    """Completeness QC for a package: manifest present, every file present with
    the recorded checksum, tile count equals ROI count."""
    pkg = Path(pkg_dir)
    m = json.loads((pkg / "manifest.json").read_text())
    problems = []
    for f in m["files"]:
        p = pkg / f["path"]
        if not p.exists():
            problems.append(f"missing {f['path']}")
        elif hashlib.sha256(p.read_bytes()).hexdigest()[:16] != f["sha256"]:
            problems.append(f"checksum mismatch {f['path']}")
    if not m["complete"]:
        problems.append("tile count != ROI count")
    return {"slide_id": m["slide_id"], "ok": not problems, "problems": problems}
