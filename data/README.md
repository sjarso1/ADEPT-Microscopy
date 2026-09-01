# data/

Local data roots. Everything here except this file and the `.gitkeep`
placeholders is git-ignored.

| Directory | Contents |
|-----------|----------|
| `calibration_kits/<instrument>/` | Kit images: `pure_h_*.png`, `pure_e_*.png`, `blank_*.png`, `target_*.png` |
| `raw/<instrument>/` | Supplied acquisitions (overview scans, HR tiles) as delivered by the hardware workstream |
| `canonical/<instrument>/` | Canonical-space intermediates (regenerable) |
| `splits/` | Locked split JSON files. **Never edit a split that has been reported on.** |
| `labels/` | Pathologist reference labels and (Semester 3) structured diagnostic labels — access-controlled |

See `docs/data_governance.md` for the ethics and access rules.
