# configs/instruments

One YAML per acquisition system describing the *hardware* (not the derived
calibration, which lives in `artifacts/calibrations/<id>.json`).

Fields: `id`, `site`, `class` (openflexure | printed3d | wsi | scanner),
`objective`, `camera`, `illumination`, `overview_microns_per_pixel`,
`hr_microns_per_pixel`, `kit_id`, `notes`.
