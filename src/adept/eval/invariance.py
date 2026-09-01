"""Instrument-invariance analysis on paired acquisitions (S1-ID-1 / S1-ID-2).

The falsifiable calibration claim: *after* mapping to canonical space, the
inter-instrument variation of a measurement on the same physical slide must
be smaller than the intra-instrument slide-to-slide variation.

For a measurement ``m`` (a criterion value, or a canonical-space image
statistic) over slides *s* and instruments *k*:

    inter  = RMS over slides of  std_k  m[s, k]        (same slide, different instruments)
    intra  = mean over instruments of std_s m[s, k]    (same instrument, different slides)
    ratio  = inter / intra                              (want ≪ 1)

Also reported: the intraclass correlation ICC(2,1) treating instruments as
raters, and a bootstrap CI on the ratio over slides.
"""

from __future__ import annotations

import numpy as np


def _icc_2_1(M: np.ndarray) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single rater."""
    n, k = M.shape
    if n < 2 or k < 2:
        return float("nan")
    grand = M.mean()
    ms_r = k * ((M.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_c = n * ((M.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    resid = M - M.mean(axis=1, keepdims=True) - M.mean(axis=0, keepdims=True) + grand
    ms_e = (resid**2).sum() / ((n - 1) * (k - 1))
    denom = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    return float((ms_r - ms_e) / denom) if denom > 0 else float("nan")


def paired_invariance(M: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """Invariance statistics for a (n_slides, n_instruments) measurement matrix.

    Rows with any NaN are dropped.
    """
    M = np.asarray(M, dtype=float)
    M = M[np.isfinite(M).all(axis=1)]
    n, k = M.shape

    def ratio(idx):
        sub = M[idx]
        inter = np.sqrt((sub.std(axis=1, ddof=1) ** 2).mean()) if k > 1 else float("nan")
        intra = sub.std(axis=0, ddof=1).mean() if len(sub) > 1 else float("nan")
        return inter / intra if intra > 0 else float("nan")

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        boots.append(ratio(rng.choice(n, n, replace=True)))
    boots = np.asarray([b for b in boots if np.isfinite(b)])
    est = ratio(np.arange(n))
    inter = float(np.sqrt((M.std(axis=1, ddof=1) ** 2).mean())) if k > 1 else float("nan")
    intra = float(M.std(axis=0, ddof=1).mean()) if n > 1 else float("nan")
    return {
        "n_slides": int(n),
        "n_instruments": int(k),
        "inter_instrument_rms": inter,
        "intra_instrument_sd": intra,
        "ratio": float(est),
        "ratio_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        if len(boots)
        else [float("nan"), float("nan")],
        "icc_2_1": _icc_2_1(M),
        "invariant": bool(np.isfinite(est) and est < 1.0),
    }


def invariance_table(measurements: dict[str, np.ndarray], **kw) -> dict[str, dict]:
    """Run :func:`paired_invariance` for several named measurements
    (e.g. one per criterion, before and after calibration)."""
    return {name: paired_invariance(M, **kw) for name, M in measurements.items()}
