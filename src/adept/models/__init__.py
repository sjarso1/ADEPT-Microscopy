"""Models.

Torch-free (always available)
    adequacy.CriterionCombiner   logistic combination of the five criterion scores —
                                 the transparent baseline every DL arm is compared to
    roi.rank_rois                adequacy-driven ROI ranking (S1) with an optional
                                 suspicion term (S3)

Torch (``pip install adept-microscopy[torch]``)
    adequacy_cnn.AdequacyCNN     EfficientNet-B0 (timm) fine-tune vs from-scratch arms
    artifact_cnn.ArtifactPatchNet  four-class artifact patch classifier with mask output
    mil.AttentionMIL             attention-based multiple-instance model (S3)
"""

from adept.models.adequacy import CriterionCombiner
from adept.models.roi import rank_rois

__all__ = ["CriterionCombiner", "rank_rois"]
