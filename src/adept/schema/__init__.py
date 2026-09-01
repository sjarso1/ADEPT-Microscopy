"""The agreed output schema shared by the computational, hardware and validation workstreams."""

from adept.schema.report import (
    SCHEMA_VERSION,
    AdequacyClass,
    AdequacyReport,
    CriterionResult,
    ReasonCode,
    RegionOfInterest,
    SuspicionReport,
)

__all__ = [
    "SCHEMA_VERSION",
    "AdequacyClass",
    "AdequacyReport",
    "CriterionResult",
    "ReasonCode",
    "RegionOfInterest",
    "SuspicionReport",
]
