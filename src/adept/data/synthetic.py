"""Synthetic multi-instrument H&E data for tests, CI and dry runs.

Nothing here resembles real tissue closely enough to train a clinical model —
that is not the purpose. The purpose is to give every pipeline stage a
deterministic, label-free-to-generate input with the *structure* of the real
problem:

* a **physical slide** is a latent canonical concentration image (cores of
  nucleus-dense tissue on a blank background, optionally with a fold or
  bubble, low cellularity, weak stain, or blur) with a B1 / B2+ label derived
  from the same generative parameters;
* an **instrument** is a simulated acquisition: its own stain vectors
  (spectral shift), illumination vignetting, white balance, blur, and noise;
* a **calibration kit** for that instrument is the pure-H, pure-E, blank and
  resolution-target images rendered through the same instrument model.

So the same slide imaged on two instruments differs only by the instrument
model — exactly the paired-acquisition design — and the calibration derived
from the kit *should* undo it. Tests assert that it does.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from skimage.draw import disk, polygon

from adept.calibration.canonical import CANONICAL_STAIN_MATRIX
from adept.calibration.od import od_to_rgb
from adept.calibration.stain import reconstruct_od
from adept.data.registry import SlideRecord, save_registry


# --------------------------------------------------------------- instruments
@dataclass
class InstrumentModel:
    """A simulated acquisition system."""

    name: str
    stain_rotation_deg: float = 0.0  # spectral shift of the stain vectors
    white_balance: tuple[float, float, float] = (1.0, 1.0, 1.0)
    vignetting: float = 0.0  # 0 = flat; 0.5 = corners at 50 % brightness
    blur_sigma: float = 0.0
    noise_sigma: float = 0.005
    concentration_gain: float = 1.0
    site: str = "mulago"

    def stain_matrix(self) -> np.ndarray:
        """Rotate the canonical H and E vectors slightly toward each other / away
        in OD space to mimic a different spectral response."""
        M = CANONICAL_STAIN_MATRIX.copy()
        th = np.radians(self.stain_rotation_deg)
        # rotate in the plane spanned by (H, E) by angle th (H toward E), E by -th/2
        h, e = M[0], M[1]
        e_perp = e - h * np.dot(e, h)
        e_perp /= np.linalg.norm(e_perp)
        h2 = np.cos(th) * h + np.sin(th) * e_perp
        h_perp = h - e * np.dot(h, e)
        h_perp /= np.linalg.norm(h_perp)
        e2 = np.cos(th / 2) * e - np.sin(th / 2) * h_perp
        M2 = np.stack([h2, e2])
        M2 = np.clip(M2, 0, None)
        return (M2 / np.linalg.norm(M2, axis=1, keepdims=True)).astype(np.float32)

    def flatfield(self, shape: tuple[int, int]) -> np.ndarray:
        H, W = shape
        yy, xx = np.mgrid[0:H, 0:W]
        r2 = ((yy - H / 2) / (H / 2)) ** 2 + ((xx - W / 2) / (W / 2)) ** 2
        f = 1.0 - self.vignetting * r2 / 2.0
        return np.repeat(f[..., None], 3, axis=-1).astype(np.float32)

    def acquire(self, concentrations: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Render canonical concentrations as this instrument would see them."""
        C = concentrations * self.concentration_gain
        od = reconstruct_od(C.astype(np.float32), self.stain_matrix())
        rgb = od_to_rgb(od)
        rgb = rgb * np.asarray(self.white_balance, np.float32)[None, None, :]
        rgb = rgb * self.flatfield(rgb.shape[:2])
        if self.blur_sigma > 0:
            rgb = cv2.GaussianBlur(rgb, (0, 0), self.blur_sigma)
        rgb = rgb + rng.normal(0, self.noise_sigma, rgb.shape).astype(np.float32)
        return np.clip(rgb, 0, 1).astype(np.float32)


DEFAULT_INSTRUMENTS: dict[str, InstrumentModel] = {
    "openflexure": InstrumentModel(
        "openflexure",
        stain_rotation_deg=8.0,
        white_balance=(1.0, 0.96, 0.90),
        vignetting=0.35,
        blur_sigma=0.8,
        noise_sigma=0.01,
        concentration_gain=0.85,
        site="mulago",
    ),
    "printed3d": InstrumentModel(
        "printed3d",
        stain_rotation_deg=-6.0,
        white_balance=(0.92, 0.98, 1.0),
        vignetting=0.5,
        blur_sigma=1.2,
        noise_sigma=0.015,
        concentration_gain=0.75,
        site="mulago",
    ),
    "wsi": InstrumentModel(
        "wsi",
        stain_rotation_deg=3.0,
        white_balance=(1.0, 1.0, 0.97),
        vignetting=0.1,
        blur_sigma=0.3,
        noise_sigma=0.006,
        concentration_gain=1.05,
        site="jhu",
    ),
    "scanner": InstrumentModel(
        "scanner",
        stain_rotation_deg=0.0,
        white_balance=(1.0, 1.0, 1.0),
        vignetting=0.02,
        blur_sigma=0.0,
        noise_sigma=0.003,
        concentration_gain=1.0,
        site="jhu",
    ),
}


# -------------------------------------------------------------------- slides
@dataclass
class SlideParams:
    slide_id: str
    n_cores: int
    core_length_px: int
    nucleus_density: float  # nuclei per 1000 tissue px
    h_strength: float
    e_strength: float
    fold: bool
    bubble: bool
    blur_sigma: float
    label: str  # "B1" or "B2+"
    reasons: list[str] = field(default_factory=list)


def _sample_slide_params(rng: np.random.Generator, slide_id: str) -> SlideParams:
    # failure modes are sampled so that roughly half of the slides are B1
    n_cores = int(rng.choice([0, 1, 2, 3], p=[0.12, 0.30, 0.30, 0.28]))
    core_len = int(rng.integers(120, 220))
    density = float(rng.uniform(1.5, 10.0))
    h = float(rng.uniform(0.4, 1.2))
    e = float(rng.uniform(0.4, 1.0))
    fold = rng.random() < 0.2
    bubble = rng.random() < 0.2
    blur = float(rng.choice([0.0, 2.5, 4.0], p=[0.75, 0.10, 0.15]))
    reasons = []
    if n_cores == 0:
        reasons.append("no_tissue")
    elif n_cores * core_len < 250:
        reasons.append("insufficient_tissue")
    if density < 2.5:
        reasons.append("low_cellularity")
    if h < 0.5 or h / max(e, 1e-3) < 0.5 or h / max(e, 1e-3) > 2.5:
        reasons.append("poor_staining")
    if blur >= 4.0:
        reasons.append("out_of_focus")
    if fold and rng.random() < 0.5:
        reasons.append("artifact_fold")
    label = "B1" if reasons else "B2+"
    return SlideParams(
        slide_id, n_cores, core_len, density, h, e, fold, bubble, blur, label, reasons
    )


def render_canonical_slide(p: SlideParams, rng: np.random.Generator, size: int = 256) -> np.ndarray:
    """Latent canonical concentration image (size, size, 2) for a slide."""
    C = np.zeros((size, size, 2), np.float32)
    tissue = np.zeros((size, size), bool)
    for i in range(p.n_cores):
        y0 = int(30 + i * (size - 60) / max(p.n_cores, 1) + rng.integers(-5, 5))
        x0 = int(rng.integers(10, max(size - p.core_length_px - 10, 11)))
        w = int(rng.integers(14, 24))
        rr, cc = polygon(
            [y0, y0, y0 + w, y0 + w],
            [x0, x0 + p.core_length_px, x0 + p.core_length_px, x0],
            C.shape[:2],
        )
        tissue[rr, cc] = True
    # cytoplasm / stroma: eosin everywhere in tissue, faint haematoxylin
    C[..., 1][tissue] = p.e_strength * rng.uniform(0.7, 1.0, tissue.sum())
    C[..., 0][tissue] = 0.15 * p.h_strength
    # nuclei: small discs of haematoxylin
    n_nuclei = int(p.nucleus_density * tissue.sum() / 1000.0)
    ys, xs = np.nonzero(tissue)
    if len(ys) and n_nuclei > 0:
        idx = rng.choice(len(ys), size=min(n_nuclei, len(ys)), replace=False)
        for y, x in zip(ys[idx], xs[idx], strict=True):
            rr, cc = disk((y, x), 2.2, shape=C.shape[:2])
            C[rr, cc, 0] = p.h_strength * rng.uniform(0.8, 1.2)
    if p.fold and tissue.any():
        # a dark, elongated over-stained ridge across a core
        y = int(np.median(ys))
        x = int(np.median(xs))
        rr, cc = polygon(
            [y - 3, y + 3, y + 3, y - 3], [x - 40, x - 40, x + 40, x + 40], C.shape[:2]
        )
        C[rr, cc, :] = [3.0, 2.5]
    if p.bubble and tissue.any():
        y = int(np.percentile(ys, 30))
        x = int(np.percentile(xs, 70))
        rr, cc = disk((y, x), 6, shape=C.shape[:2])
        C[rr, cc, :] = 0.0
        ring_r, ring_c = disk((y, x), 8, shape=C.shape[:2])
        ring = np.zeros_like(tissue)
        ring[ring_r, ring_c] = True
        ring[rr, cc] = False
        C[ring, 0] = 2.5
    if p.blur_sigma > 0:
        C = cv2.GaussianBlur(C, (0, 0), p.blur_sigma)
    return np.clip(C, 0, None).astype(np.float32)


# ---------------------------------------------------------- calibration kits
def render_kit(
    inst: InstrumentModel, rng: np.random.Generator, size: int = 256
) -> dict[str, list[np.ndarray]]:
    """Pure-H, pure-E, blank and resolution-target images for an instrument."""
    blank = np.zeros((size, size, 2), np.float32)
    h_only = blank.copy()
    e_only = blank.copy()
    yy, xx = np.mgrid[0:size, 0:size]
    # single-stain kit slides are uniformly stained sections: a smooth ±10 %
    # spatial variation keeps the median concentration well defined under blur
    pattern = 0.7 * (1.0 + 0.1 * np.sin(xx / 40.0) * np.cos(yy / 50.0)).astype(np.float32)
    h_only[..., 0] = pattern
    e_only[..., 1] = pattern
    target = blank.copy()
    bars = ((xx // 6) % 2 == 0).astype(np.float32)
    target[..., 0] = 1.6 * bars
    target[..., 1] = 0.4 * bars
    return {
        "pure_h": [inst.acquire(h_only, rng) for _ in range(2)],
        "pure_e": [inst.acquire(e_only, rng) for _ in range(2)],
        "blank": [inst.acquire(blank, rng) for _ in range(3)],
        "target": [inst.acquire(target, rng)],
    }


# --------------------------------------------------------------- the dataset
def make_dataset(
    out_dir: str | Path,
    n_slides: int = 24,
    instruments: list[str] | None = None,
    size: int = 256,
    seed: int = 0,
) -> tuple[list[SlideRecord], list[SlideParams]]:
    """Write a synthetic multi-instrument dataset.

    Layout::

        out_dir/
          registry.csv                 one row per acquisition
          pairs.csv                    slide_id, instrument, path  (all paired)
          slides/<slide>_<inst>.png    acquisitions
          canonical/<slide>.npy        latent canonical concentrations (ground truth)
          kits/<inst>/{pure_h,pure_e,blank,target}_<i>.png
          labels.csv                   slide_id, label, reasons
    """
    out = Path(out_dir)
    instruments = instruments or list(DEFAULT_INSTRUMENTS)
    rng = np.random.default_rng(seed)
    (out / "slides").mkdir(parents=True, exist_ok=True)
    (out / "canonical").mkdir(exist_ok=True)
    records: list[SlideRecord] = []
    params: list[SlideParams] = []
    for i in range(n_slides):
        sid = f"S{i + 1:04d}"
        p = _sample_slide_params(rng, sid)
        params.append(p)
        C = render_canonical_slide(p, rng, size)
        np.save(out / "canonical" / f"{sid}.npy", C)
        for name in instruments:
            inst = DEFAULT_INSTRUMENTS[name]
            rgb = inst.acquire(C, rng)
            rel = f"slides/{sid}_{name}.png"
            cv2.imwrite(
                str(out / rel), cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            )
            records.append(
                SlideRecord(
                    acq_id=f"{sid}_{name}",
                    slide_id=sid,
                    instrument=name,
                    site=inst.site,
                    path=rel,
                    label=p.label,
                    paired_set="synthetic",
                )
            )
    for name in instruments:
        kit_dir = out / "kits" / name
        kit_dir.mkdir(parents=True, exist_ok=True)
        kit = render_kit(DEFAULT_INSTRUMENTS[name], rng, size)
        for kind, imgs in kit.items():
            for j, im in enumerate(imgs):
                cv2.imwrite(
                    str(kit_dir / f"{kind}_{j}.png"),
                    cv2.cvtColor((im * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                )
    save_registry(records, out / "registry.csv")
    with open(out / "pairs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slide_id", "instrument", "path"])
        for r in records:
            w.writerow([r.slide_id, r.instrument, r.path])
    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slide_id", "label", "reasons"])
        for p in params:
            w.writerow([p.slide_id, p.label, ";".join(p.reasons)])
    return records, params


def load_kit(kit_dir: str | Path) -> dict[str, list[np.ndarray]]:
    """Read a calibration kit directory written by :func:`make_dataset` (or the
    real kit protocol, which uses the same file-name convention)."""
    kit_dir = Path(kit_dir)
    out: dict[str, list[np.ndarray]] = {}
    for kind in ("pure_h", "pure_e", "blank", "target"):
        files = sorted(kit_dir.glob(f"{kind}_*.png"))
        out[kind] = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in files]
    return out


def read_rgb(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
