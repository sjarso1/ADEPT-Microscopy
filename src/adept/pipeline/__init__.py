"""Inference interface and output packages (S2-ID-3, extended in S3-ID-3)."""

from adept.pipeline.inference import AdequacyPipeline
from adept.pipeline.package import write_output_package

__all__ = ["AdequacyPipeline", "write_output_package"]
