"""Pydantic models for the ADEPT output schema.

This is the contract between the classifier (computational workstream), the
stage controller (hardware workstream) and the reading study (validation
workstream). Semester 2 wraps it behind the inference API; Semester 3
*extends* it with the suspicion score without breaking existing consumers —
hence every addition is optional with a default.

Schema changes bump ``SCHEMA_VERSION`` and are recorded as an ADR.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

SCHEMA_VERSION = "1.1.0"


class AdequacyClass(str, Enum):
    """UK NHSBSP B-classification collapsed to the adequacy decision."""

    B1_INADEQUATE = "B1"
    B2_PLUS_ADEQUATE = "B2+"


class ReasonCode(str, Enum):
    """Why a specimen was called inadequate — mirrors pathologist-stated reasons."""

    NO_TISSUE = "no_tissue"
    INSUFFICIENT_TISSUE = "insufficient_tissue"
    LOW_CELLULARITY = "low_cellularity"
    POOR_STAINING = "poor_staining"
    OUT_OF_FOCUS = "out_of_focus"
    ARTIFACT_FOLD = "artifact_fold"
    ARTIFACT_BUBBLE = "artifact_bubble"
    ARTIFACT_CRUSH = "artifact_crush"
    ARTIFACT_DESICCATION = "artifact_desiccation"


class CriterionResult(BaseModel):
    """One of the five adequacy criteria, measured in canonical space."""

    name: str = Field(description="tissue | cellularity | staining | focus | artifact")
    value: float = Field(description="Primary scalar measurement (units in `unit`).")
    unit: str = ""
    score: float = Field(
        ge=0.0, le=1.0, description="Normalised adequacy score, 1 = fully adequate."
    )
    passed: bool
    threshold: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class RegionOfInterest(BaseModel):
    """A region proposed for targeted high-resolution capture.

    Coordinates are in overview-scan pixel space; the hardware workstream
    converts to stage coordinates using the scan's affine metadata.
    """

    rank: int = Field(ge=1)
    x: int
    y: int
    width: int
    height: int
    score: float = Field(ge=0.0, le=1.0, description="Priority score used for ranking.")
    adequacy_score: float | None = Field(default=None, ge=0.0, le=1.0)
    suspicion_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Semester 3 malignancy-suspicion attention weight.",
    )
    action: str = Field(default="capture_hr", description="capture_hr | rescan | refocus")
    reasons: list[ReasonCode] = Field(default_factory=list)


class AdequacyReport(BaseModel):
    """Slide-level adequacy decision plus the ranked ROI list."""

    schema_version: str = SCHEMA_VERSION
    slide_id: str
    instrument: str
    calibration_id: str | None = None
    adequacy: AdequacyClass
    probability_inadequate: float = Field(ge=0.0, le=1.0)
    decision_threshold: float = Field(ge=0.0, le=1.0)
    reasons: list[ReasonCode] = Field(default_factory=list)
    criteria: list[CriterionResult] = Field(default_factory=list)
    rois: list[RegionOfInterest] = Field(default_factory=list)
    rescan_requested: bool = False
    model_id: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("criteria")
    @classmethod
    def _unique_criteria(cls, v: list[CriterionResult]) -> list[CriterionResult]:
        names = [c.name for c in v]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate criterion names: {names}")
        return v

    def criterion(self, name: str) -> CriterionResult | None:
        return next((c for c in self.criteria if c.name == name), None)


class SuspicionReport(BaseModel):
    """Semester 3 extension: slide-level malignancy-suspicion triage.

    The score *prioritises*; it does not diagnose. Consumers must render it as
    a queue-ordering aid, never as a diagnostic label.
    """

    schema_version: str = SCHEMA_VERSION
    slide_id: str
    instrument: str
    suspicion_probability: float = Field(ge=0.0, le=1.0)
    suspicion_flag: bool
    decision_threshold: float = Field(ge=0.0, le=1.0)
    scope: str = Field(default="binary", description="binary (malignant vs benign) | three_way")
    class_probabilities: dict[str, float] | None = None
    top_tiles: list[RegionOfInterest] = Field(default_factory=list)
    model_id: str | None = None
    claim_ceiling: str = "triage_not_diagnosis"
