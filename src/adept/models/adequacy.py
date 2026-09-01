"""Transparent five-criterion adequacy combiner (baseline and ablation reference).

A logistic regression over the five criterion *scores* (each in [0, 1]).
It is the model against which the CNN arms are compared, the natural vehicle
for the per-criterion ablation (drop a feature, refit, re-measure), and a
perfectly serviceable deployable classifier on its own for the Minimum
endpoint. Because it consumes canonical-space measurements only, it inherits
their instrument invariance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from adept.criteria import CRITERIA
from adept.schema import AdequacyClass, AdequacyReport, CriterionResult, ReasonCode

_REASON_FOR_CRITERION = {
    "tissue": ReasonCode.INSUFFICIENT_TISSUE,
    "cellularity": ReasonCode.LOW_CELLULARITY,
    "staining": ReasonCode.POOR_STAINING,
    "focus": ReasonCode.OUT_OF_FOCUS,
}


def criteria_to_features(
    criteria: list[CriterionResult], use: tuple[str, ...] = CRITERIA
) -> np.ndarray:
    by_name = {c.name: c for c in criteria}
    return np.array([by_name[n].score if n in by_name else 0.0 for n in use], dtype=np.float32)


class CriterionCombiner:
    def __init__(self, use: tuple[str, ...] = CRITERIA, class_weight="balanced", C: float = 1.0):
        self.use = tuple(use)
        self.model = LogisticRegression(class_weight=class_weight, C=C, max_iter=1000)
        self.threshold = 0.5
        self.model_id = "criterion-combiner-v1"

    # ----------------------------------------------------------------- fit
    def fit(self, X: np.ndarray, y_inadequate: np.ndarray) -> CriterionCombiner:
        self.model.fit(X, y_inadequate)
        return self

    def predict_proba_inadequate(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, list(self.model.classes_).index(1)]

    # --------------------------------------------------------------- report
    def report(
        self,
        slide_id: str,
        instrument: str,
        criteria: list[CriterionResult],
        calibration_id: str | None = None,
        rois=None,
    ) -> AdequacyReport:
        x = criteria_to_features(criteria, self.use)[None]
        p = float(self.predict_proba_inadequate(x)[0])
        inadequate = p >= self.threshold
        reasons = []
        if inadequate:
            for c in criteria:
                if c.passed:
                    continue
                if c.name == "artifact":
                    reasons += [ReasonCode(r) for r in c.details.get("reasons", [])]
                elif c.name == "tissue" and c.details.get("core_count", 0) == 0:
                    reasons.append(ReasonCode.NO_TISSUE)
                elif c.name in _REASON_FOR_CRITERION:
                    reasons.append(_REASON_FOR_CRITERION[c.name])
        return AdequacyReport(
            slide_id=slide_id,
            instrument=instrument,
            calibration_id=calibration_id,
            adequacy=AdequacyClass.B1_INADEQUATE if inadequate else AdequacyClass.B2_PLUS_ADEQUATE,
            probability_inadequate=p,
            decision_threshold=self.threshold,
            reasons=list(dict.fromkeys(reasons)),
            criteria=criteria,
            rois=rois or [],
            rescan_requested=any(r.action in ("rescan", "refocus") for r in (rois or [])),
            model_id=self.model_id,
        )

    # ---------------------------------------------------------- persistence
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "model_id": self.model_id,
                    "use": self.use,
                    "threshold": self.threshold,
                    "coef": self.model.coef_.tolist(),
                    "intercept": self.model.intercept_.tolist(),
                    "classes": self.model.classes_.tolist(),
                },
                indent=1,
            )
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> CriterionCombiner:
        d = json.loads(Path(path).read_text())
        obj = cls(use=tuple(d["use"]))
        obj.model_id = d["model_id"]
        obj.threshold = d["threshold"]
        obj.model.coef_ = np.asarray(d["coef"])
        obj.model.intercept_ = np.asarray(d["intercept"])
        obj.model.classes_ = np.asarray(d["classes"])
        return obj

    @classmethod
    def heuristic(cls) -> CriterionCombiner:
        """An untrained combiner with hand-set weights: P(inadequate) is high
        when any criterion score is low. Used before labels exist."""
        obj = cls()
        obj.model_id = "criterion-combiner-heuristic"
        obj.model.coef_ = -np.array([[4.0, 3.0, 3.0, 3.0, 2.5]])
        obj.model.intercept_ = np.array([9.0])
        obj.model.classes_ = np.array([0, 1])
        return obj
