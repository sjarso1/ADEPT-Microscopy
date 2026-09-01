"""Sensitivity, specificity, AUROC and Cohen's κ with bootstrap confidence intervals.

Conventions
-----------
* The **positive class is "inadequate" (B1)** for adequacy and **"malignant"**
  for triage — i.e. the class whose miss is clinically costly. Sensitivity is
  therefore the metric that dominates the performance budget.
* ``y_true`` is 0/1, ``y_score`` is the model's probability of the positive
  class, ``threshold`` converts scores to decisions.
* Every public function returns point estimates *and* 95 % bootstrap CIs
  (percentile method, stratified resampling so that both classes are always
  present). Nothing here returns a bare point estimate by default.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import roc_auc_score

TARGETS_ADEQUACY = {"sensitivity": 0.90, "specificity": 0.85, "auroc": 0.92, "kappa": 0.75}
TARGETS_TRIAGE = {"sensitivity": 0.95, "specificity": 0.80, "auroc": 0.90}


@dataclass
class CI:
    estimate: float
    lower: float
    upper: float

    def as_list(self) -> list[float]:
        return [self.estimate, self.lower, self.upper]

    def __str__(self) -> str:
        return f"{self.estimate:.3f} [{self.lower:.3f}, {self.upper:.3f}]"


@dataclass
class DiagnosticMetrics:
    n: int
    prevalence: float
    threshold: float
    sensitivity: CI
    specificity: CI
    auroc: CI
    kappa: CI
    ppv: CI
    npv: CI
    confusion: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "n": self.n,
            "prevalence": self.prevalence,
            "threshold": self.threshold,
            "confusion": self.confusion,
        }
        for k in ("sensitivity", "specificity", "auroc", "kappa", "ppv", "npv"):
            ci: CI = getattr(self, k)
            d[k] = {"estimate": ci.estimate, "ci95": [ci.lower, ci.upper]}
        return d


# ------------------------------------------------------------- primitives
def _confusion(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn


def sensitivity(y_true, y_pred) -> float:
    tp, _, _, fn = _confusion(y_true, y_pred)
    return tp / (tp + fn) if tp + fn else float("nan")


def specificity(y_true, y_pred) -> float:
    _, tn, fp, _ = _confusion(y_true, y_pred)
    return tn / (tn + fp) if tn + fp else float("nan")


def ppv(y_true, y_pred) -> float:
    tp, _, fp, _ = _confusion(y_true, y_pred)
    return tp / (tp + fp) if tp + fp else float("nan")


def npv(y_true, y_pred) -> float:
    _, tn, _, fn = _confusion(y_true, y_pred)
    return tn / (tn + fn) if tn + fn else float("nan")


def cohens_kappa(a, b) -> float:
    """Cohen's κ between two binary (or categorical) label vectors.

    Used both for inter-rater agreement on the reference standard (must clear
    0.75 before training) and for system-versus-reference concordance.
    """
    a, b = np.asarray(a), np.asarray(b)
    cats = np.unique(np.concatenate([a, b]))
    n = len(a)
    if n == 0:
        return float("nan")
    po = float((a == b).mean())
    pe = sum(((a == c).mean()) * ((b == c).mean()) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def auroc(y_true, y_score) -> float:
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


# ----------------------------------------------------------------- bootstrap
def bootstrap_ci(
    stat: Callable[[np.ndarray], float],
    y_true: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    stratified: bool = True,
) -> CI:
    """Percentile bootstrap CI for ``stat(idx) -> float`` where ``idx`` is a
    resampled index array into the evaluation set."""
    y_true = np.asarray(y_true)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    est = float(stat(np.arange(n)))
    if n == 0:
        return CI(est, float("nan"), float("nan"))
    pos, neg = np.flatnonzero(y_true == 1), np.flatnonzero(y_true == 0)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        if stratified and len(pos) and len(neg):
            idx = np.concatenate(
                [rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
            )
        else:
            idx = rng.choice(n, n, replace=True)
        vals[b] = stat(idx)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return CI(est, float("nan"), float("nan"))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return CI(est, float(lo), float(hi))


def diagnostic_metrics(
    y_true, y_score, threshold: float = 0.5, n_boot: int = 2000, seed: int = 0
) -> DiagnosticMetrics:
    """Full diagnostic-accuracy panel with 95 % bootstrap CIs."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    tp, tn, fp, fn = _confusion(y_true, y_pred)

    def mk(fn_):
        return bootstrap_ci(lambda i: fn_(y_true[i], y_pred[i]), y_true, n_boot, seed)

    return DiagnosticMetrics(
        n=int(len(y_true)),
        prevalence=float(y_true.mean()) if len(y_true) else float("nan"),
        threshold=threshold,
        sensitivity=mk(sensitivity),
        specificity=mk(specificity),
        auroc=bootstrap_ci(lambda i: auroc(y_true[i], y_score[i]), y_true, n_boot, seed),
        kappa=mk(cohens_kappa),
        ppv=mk(ppv),
        npv=mk(npv),
        confusion={"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    )


def per_group_metrics(
    y_true, y_score, groups, threshold: float = 0.5, n_boot: int = 1000, seed: int = 0
) -> dict[str, DiagnosticMetrics]:
    """Metrics overall and per group (instrument, site, LOIO fold …)."""
    groups = np.asarray(groups)
    out = {"overall": diagnostic_metrics(y_true, y_score, threshold, n_boot, seed)}
    for g in sorted(np.unique(groups)):
        m = groups == g
        out[str(g)] = diagnostic_metrics(
            np.asarray(y_true)[m], np.asarray(y_score)[m], threshold, n_boot, seed
        )
    return out


def meets_targets(m: DiagnosticMetrics, targets: dict[str, float] | None = None) -> dict[str, bool]:
    """Pass/fail of each target on the *point estimate*; the report should
    also state whether the CI lower bound clears it."""
    targets = targets or TARGETS_ADEQUACY
    return {k: bool(getattr(m, k).estimate >= v) for k, v in targets.items() if hasattr(m, k)}


def threshold_for_sensitivity(y_true, y_score, target_sensitivity: float = 0.90) -> float:
    """Largest threshold achieving at least the target sensitivity on this set
    (used for the cost-based threshold analysis; must be chosen on *validation*
    data, never on the locked test set)."""
    y_true = np.asarray(y_true)
    s = np.sort(np.asarray(y_score)[y_true == 1])
    if len(s) == 0:
        return 0.5
    k = int(np.floor((1 - target_sensitivity) * len(s)))
    return float(s[min(k, len(s) - 1)])
