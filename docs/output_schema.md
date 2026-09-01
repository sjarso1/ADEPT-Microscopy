# Output schema (v1.1.0)

Source of truth: `src/adept/schema/report.py`. This page is the plain-language
contract for the hardware and validation workstreams.

## `AdequacyReport`

| Field | Type | Meaning |
|-------|------|---------|
| `schema_version` | str | `"1.1.0"` |
| `slide_id` | str | Physical slide id |
| `instrument` | str | Instrument id (see `adept.INSTRUMENTS` or a registered new id) |
| `calibration_id` | str? | Which calibration JSON produced the canonical mapping |
| `adequacy` | `"B1"` \| `"B2+"` | The decision |
| `probability_inadequate` | float [0,1] | Model probability of B1 |
| `decision_threshold` | float | Threshold applied (chosen on validation data for the target sensitivity) |
| `reasons` | list of reason codes | Why B1 (empty for B2+) |
| `criteria` | list of `CriterionResult` | The five measurements |
| `rois` | list of `RegionOfInterest` | Ranked actions for the stage controller |
| `rescan_requested` | bool | True if any ROI action is `rescan`/`refocus` |
| `model_id` | str? | Identifier of the classifier version |

Reason codes: `no_tissue`, `insufficient_tissue`, `low_cellularity`,
`poor_staining`, `out_of_focus`, `artifact_fold`, `artifact_bubble`,
`artifact_crush`, `artifact_desiccation`.

## `CriterionResult`

`name`, `value` (+ `unit`), `score` ∈ [0,1] (1 = fully adequate), `passed`,
`threshold`, `details` (per-criterion diagnostics such as the focus tile map).

## `RegionOfInterest`

Pixel coordinates in overview-scan space (`x, y, width, height`); the hardware
workstream converts to stage coordinates with the scan's affine metadata.

| `action` | Meaning for the stage controller |
|----------|----------------------------------|
| `refocus` | Re-acquire this tile after autofocus |
| `rescan` | Re-acquire this tile (illumination / motion issue) |
| `capture_hr` | Acquire a high-resolution tile here, in `rank` order |

`score` is the ranking priority; `adequacy_score` and `suspicion_score`
(Semester 3, optional) expose its components.

## `SuspicionReport` (Semester 3)

Slide-level `suspicion_probability`, `suspicion_flag`, `scope`
(`binary` | `three_way`), `top_tiles`, and the constant
`claim_ceiling = "triage_not_diagnosis"` which consumers must display.

## Versioning

Additive changes (new optional fields) bump the minor version; any removal or
rename bumps the major version and requires an ADR plus coordinated updates in
the hardware and validation workstreams.
