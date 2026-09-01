"""Region-of-interest detection and prioritisation.

Semester 1: adequacy-driven. Tiles are scored by tissue presence × cellularity
× sharpness; the lowest-quality tissue tiles become ``rescan`` / ``refocus``
targets (closing the loop), and the highest-quality, most cellular tiles
become ``capture_hr`` targets.

Semester 3: the optional ``suspicion`` map (per-tile MIL attention) re-orders
the capture list so the first tiles sent are the ones a pathologist would look
at first. The interface is unchanged — the field is simply populated.
"""

from __future__ import annotations

import numpy as np

from adept.schema import ReasonCode, RegionOfInterest


def rank_rois(
    tissue_tile_map: np.ndarray,
    sharpness_map: np.ndarray,
    tile_px: int,
    cellularity_map: np.ndarray | None = None,
    suspicion_map: np.ndarray | None = None,
    max_capture: int = 8,
    max_rescan: int = 4,
    min_relative_sharpness: float = 0.35,
    suspicion_weight: float = 0.6,
) -> list[RegionOfInterest]:
    """Build a ranked ROI list from tile-level maps of identical shape (nH, nW)."""
    tissue = np.asarray(tissue_tile_map, dtype=bool)
    sharp = np.clip(np.asarray(sharpness_map, dtype=float), 0, 1)
    cell = np.ones_like(sharp) if cellularity_map is None else np.clip(cellularity_map, 0, 1)
    quality = tissue * sharp * (0.5 + 0.5 * cell)
    rois: list[RegionOfInterest] = []

    # closed loop: poor tissue tiles → rescan / refocus
    poor = tissue & (sharp < min_relative_sharpness)
    for i, j in sorted(zip(*np.nonzero(poor), strict=True), key=lambda ij: sharp[ij])[:max_rescan]:
        rois.append(
            RegionOfInterest(
                rank=1,
                x=int(j * tile_px),
                y=int(i * tile_px),
                width=tile_px,
                height=tile_px,
                score=float(1.0 - sharp[i, j]),
                adequacy_score=float(quality[i, j]),
                action="refocus",
                reasons=[ReasonCode.OUT_OF_FOCUS],
            )
        )

    # targeted high-res capture: best tissue tiles, re-weighted by suspicion if present
    priority = quality.copy()
    if suspicion_map is not None:
        s = np.clip(np.asarray(suspicion_map, dtype=float), 0, 1)
        priority = (1 - suspicion_weight) * quality + suspicion_weight * s * tissue
    order = np.argsort(priority, axis=None)[::-1]
    n = 0
    for flat in order:
        i, j = np.unravel_index(flat, priority.shape)
        if not tissue[i, j] or priority[i, j] <= 0:
            break
        rois.append(
            RegionOfInterest(
                rank=1,
                x=int(j * tile_px),
                y=int(i * tile_px),
                width=tile_px,
                height=tile_px,
                score=float(priority[i, j]),
                adequacy_score=float(quality[i, j]),
                suspicion_score=None if suspicion_map is None else float(suspicion_map[i, j]),
                action="capture_hr",
            )
        )
        n += 1
        if n >= max_capture:
            break

    # final ranking: rescans first (they drive the loop), then captures by score
    rois.sort(key=lambda r: (r.action == "capture_hr", -r.score))
    for k, r in enumerate(rois, start=1):
        r.rank = k
    return rois
