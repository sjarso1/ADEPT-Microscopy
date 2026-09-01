"""Stain-vector estimation and colour deconvolution.

Two routes to a per-instrument stain matrix:

1. **Single-stain kit slides** (preferred, label-free, ADEPT calibration kit):
   the dominant OD direction of a *pure haematoxylin* slide and of a *pure eosin*
   slide, measured on the instrument being calibrated, give the H and E stain
   vectors directly (:func:`stain_matrix_from_single_stains`).

2. **Macenko** (Macenko et al., ISBI 2009) — unsupervised estimation from an
   H&E image: project OD pixels onto the plane spanned by the two leading
   singular vectors and take the extreme angular percentiles as the stain
   directions (:func:`estimate_stain_vectors_macenko`). Used when kit slides
   are unavailable and as a consistency check against route 1.

A stain matrix ``M`` has shape (2, 3): rows are unit-norm OD vectors for H and E.
Deconvolution solves ``OD = C @ M`` for the concentration image ``C`` with the
pseudo-inverse.
"""

from __future__ import annotations

import numpy as np

from adept.calibration.od import as_float_rgb, rgb_to_od


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _tissue_od_pixels(od: np.ndarray, beta: float = 0.15) -> np.ndarray:
    """Flatten OD and drop near-transparent pixels (|OD| < beta)."""
    px = od.reshape(-1, 3)
    return px[np.linalg.norm(px, axis=1) > beta]


def dominant_od_direction(img: np.ndarray, beta: float = 0.15, i0=1.0) -> np.ndarray:
    """Principal OD direction (unit vector, shape (3,)) of a single-stain image.

    The sign is fixed so that all components are non-negative, as physical
    absorbance must be.
    """
    od = rgb_to_od(as_float_rgb(img), i0=i0)
    px = _tissue_od_pixels(od, beta)
    if px.shape[0] < 50:
        raise ValueError("too few stained pixels to estimate a stain direction; lower beta?")
    # first right-singular vector of the centred-free OD cloud == direction of max energy
    _, _, vt = np.linalg.svd(px, full_matrices=False)
    v = vt[0]
    if v.sum() < 0:
        v = -v
    return _unit(np.clip(v, 0.0, None)).astype(np.float32)


def stain_matrix_from_single_stains(
    pure_h_images: list[np.ndarray],
    pure_e_images: list[np.ndarray],
    beta: float = 0.15,
    i0=1.0,
) -> np.ndarray:
    """Per-instrument stain matrix from pure-H and pure-E calibration slides.

    Returns
    -------
    (2, 3) float32 matrix, rows = unit OD vectors [H; E].
    """
    h = np.mean([dominant_od_direction(im, beta, i0) for im in pure_h_images], axis=0)
    e = np.mean([dominant_od_direction(im, beta, i0) for im in pure_e_images], axis=0)
    return np.stack([_unit(h), _unit(e)]).astype(np.float32)


def estimate_stain_vectors_macenko(
    img: np.ndarray,
    beta: float = 0.15,
    alpha: float = 1.0,
    i0=1.0,
    max_pixels: int = 200_000,
    seed: int = 0,
) -> np.ndarray:
    """Macenko stain-vector estimation from an H&E image.

    Parameters
    ----------
    beta : OD magnitude threshold below which pixels are treated as background.
    alpha : angular percentile (in %) defining the extreme stain directions.

    Returns
    -------
    (2, 3) float32 matrix, rows = [H; E], ordered so that H is the vector with
    the larger red-channel OD (haematoxylin absorbs red; eosin hardly does).
    """
    od = rgb_to_od(as_float_rgb(img), i0=i0)
    px = _tissue_od_pixels(od, beta)
    if px.shape[0] < 100:
        raise ValueError("too few stained pixels for Macenko estimation")
    if px.shape[0] > max_pixels:
        rng = np.random.default_rng(seed)
        px = px[rng.choice(px.shape[0], max_pixels, replace=False)]

    # plane of the two leading eigenvectors of the OD covariance
    cov = np.cov(px, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    basis = evecs[:, np.argsort(evals)[::-1][:2]]  # (3, 2)
    # make the basis vectors point into the positive octant on average
    if basis[:, 0].sum() < 0:
        basis[:, 0] *= -1
    if basis[:, 1].sum() < 0:
        basis[:, 1] *= -1

    proj = px @ basis  # (N, 2)
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    lo, hi = np.percentile(phi, [alpha, 100 - alpha])
    v1 = basis @ np.array([np.cos(lo), np.sin(lo)])
    v2 = basis @ np.array([np.cos(hi), np.sin(hi)])
    v1, v2 = _unit(np.clip(v1, 0, None)), _unit(np.clip(v2, 0, None))

    # haematoxylin absorbs strongly in red (OD_R ≈ 0.65) whereas eosin barely does
    # (OD_R ≈ 0.07): the vector with the larger red OD component is H
    h, e = (v1, v2) if v1[0] > v2[0] else (v2, v1)
    return np.stack([h, e]).astype(np.float32)


def deconvolve(od: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
    """Colour deconvolution: OD (H, W, 3) → stain concentrations (H, W, 2).

    Solves ``OD = C @ M`` in the least-squares sense with the Moore–Penrose
    pseudo-inverse of the (2, 3) stain matrix. Negative concentrations
    (numerical noise) are clipped at 0.
    """
    M = np.asarray(stain_matrix, dtype=np.float32)
    pinv = np.linalg.pinv(M)  # (3, 2)
    C = od.reshape(-1, 3) @ pinv
    return np.clip(C, 0.0, None).reshape(od.shape[0], od.shape[1], 2).astype(np.float32)


def reconstruct_od(concentrations: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
    """Inverse of :func:`deconvolve`: (H, W, 2) concentrations → (H, W, 3) OD."""
    M = np.asarray(stain_matrix, dtype=np.float32)
    H, W, _ = concentrations.shape
    return (concentrations.reshape(-1, 2) @ M).reshape(H, W, 3).astype(np.float32)


def stain_angle_deg(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """Angle (degrees) between corresponding stain vectors of two matrices.

    Used to quantify spectral differences between instruments and to detect
    calibration drift on kit re-runs.
    """
    out = []
    for a, b in zip(m1, m2, strict=True):
        c = float(np.clip(np.dot(_unit(a), _unit(b)), -1.0, 1.0))
        out.append(np.degrees(np.arccos(c)))
    return np.asarray(out, dtype=np.float32)
