# Semester 1 kanban board

Project board: <https://github.com/users/sjarso1/projects/5> (linked to this repository).

The 14-week Semester 1 work plan (course agreement) is organised as **18 tickets**, each with up to three sub-issues, across **seven fortnightly design reviews**. The working rule is **no more than three tickets in progress per design review**; the ticket count per review is 3 · 3 · 3 · 2 · 3 · 2 · 2, leaving slack for work that overruns. The generating source is [`semester1_board.yaml`](semester1_board.yaml); the issues were created from it and should be treated as the live copy — edit issues on GitHub, not the YAML.

## Design reviews (milestones)

| Review | Weeks | Tickets |
|---|---|---|
| DR1 | 1–2 | [T01](https://github.com/sjarso1/ADEPT-Microscopy/issues/1), [T02](https://github.com/sjarso1/ADEPT-Microscopy/issues/5), [T03](https://github.com/sjarso1/ADEPT-Microscopy/issues/10) |
| DR2 | 3–4 | [T04](https://github.com/sjarso1/ADEPT-Microscopy/issues/14), [T05](https://github.com/sjarso1/ADEPT-Microscopy/issues/18), [T06](https://github.com/sjarso1/ADEPT-Microscopy/issues/22) |
| DR3 | 5–6 | [T07](https://github.com/sjarso1/ADEPT-Microscopy/issues/26), [T08](https://github.com/sjarso1/ADEPT-Microscopy/issues/30), [T09](https://github.com/sjarso1/ADEPT-Microscopy/issues/34) |
| DR4 | 7–8 | [T10](https://github.com/sjarso1/ADEPT-Microscopy/issues/38), [T11](https://github.com/sjarso1/ADEPT-Microscopy/issues/42) |
| DR5 | 9–10 | [T12](https://github.com/sjarso1/ADEPT-Microscopy/issues/46), [T13](https://github.com/sjarso1/ADEPT-Microscopy/issues/50), [T14](https://github.com/sjarso1/ADEPT-Microscopy/issues/54) |
| DR6 | 11–12 | [T15](https://github.com/sjarso1/ADEPT-Microscopy/issues/58), [T16](https://github.com/sjarso1/ADEPT-Microscopy/issues/62) |
| DR7 | 13–14 | [T17](https://github.com/sjarso1/ADEPT-Microscopy/issues/66), [T18](https://github.com/sjarso1/ADEPT-Microscopy/issues/70) |

## Tickets

| Ticket | Issue | Review | Week | Labels | Title |
|---|---|---|---|---|---|
| T01 | [#1](https://github.com/sjarso1/ADEPT-Microscopy/issues/1) | DR1 | 1 | `setup`, `endpoint:minimum` | Set up the ML environment, training infrastructure and experiment tracking |
| T02 | [#5](https://github.com/sjarso1/ADEPT-Microscopy/issues/5) | DR1 | 2 | `setup`, `endpoint:minimum` | Literature review and annotated bibliography |
| T03 | [#10](https://github.com/sjarso1/ADEPT-Microscopy/issues/10) | DR1 | 2 | `setup`, `endpoint:minimum`, `depends:validation` | Reference-standard gate: confirm access to labels and inter-rater κ ≥ 0.75 |
| T04 | [#14](https://github.com/sjarso1/ADEPT-Microscopy/issues/14) | DR2 | 3 | `S1-ID-1`, `endpoint:minimum`, `depends:hardware` | Calibration-kit protocol v1 and kit image acquisition on every instrument |
| T05 | [#18](https://github.com/sjarso1/ADEPT-Microscopy/issues/18) | DR2 | 3 | `S1-ID-1`, `endpoint:minimum` | Derive per-instrument calibrations for the OpenFlexure and the reference scanner |
| T06 | [#22](https://github.com/sjarso1/ADEPT-Microscopy/issues/22) | DR2 | 4 | `S1-ID-1`, `endpoint:target` | 3D-printed microscope calibration and the canonical-space definition (ADR) |
| T07 | [#26](https://github.com/sjarso1/ADEPT-Microscopy/issues/26) | DR3 | 5 | `S1-ID-1`, `endpoint:target`, `endpoint:maximum` | Paired-slide residual analysis with error bars, re-runnable protocol, and calibration-stability check |
| T08 | [#30](https://github.com/sjarso1/ADEPT-Microscopy/issues/30) | DR3 | 5 | `S1-ID-1`, `endpoint:minimum` | S1-ID-1 report (calibrations and canonical space) and repository tag `s1-id1` |
| T09 | [#34](https://github.com/sjarso1/ADEPT-Microscopy/issues/34) | DR3 | 6 | `S1-ID-2`, `endpoint:minimum` | Criterion 1 (tissue/core detection) and Criterion 3 (staining quality) on real canonical-space images |
| T10 | [#38](https://github.com/sjarso1/ADEPT-Microscopy/issues/38) | DR4 | 7 | `S1-ID-2`, `endpoint:minimum`, `endpoint:target` | Criterion 4 (focus-quality map, per-instrument calibrated) and Criterion 2 (cellularity via pretrained nucleus model) |
| T11 | [#42](https://github.com/sjarso1/ADEPT-Microscopy/issues/42) | DR4 | 8 | `S1-ID-2`, `endpoint:minimum`, `endpoint:target`, `depends:validation` | Criterion 5: labelled artifact patch set and trained four-class artifact detector with mask output |
| T12 | [#46](https://github.com/sjarso1/ADEPT-Microscopy/issues/46) | DR5 | 9 | `S1-ID-2`, `endpoint:target`, `endpoint:maximum` | Instrument-invariance analysis per criterion and five-criterion assembly into the agreed schema |
| T13 | [#50](https://github.com/sjarso1/ADEPT-Microscopy/issues/50) | DR5 | 9 | `S1-ID-2`, `endpoint:minimum` | S1-ID-2 report (five criteria and invariance) and repository tag `s1-id2` |
| T14 | [#54](https://github.com/sjarso1/ADEPT-Microscopy/issues/54) | DR5 | 10 | `S1-ID-3`, `endpoint:minimum`, `endpoint:target` | Lock the site- and instrument-stratified split and train the integrated adequacy classifier (architecture experiment) |
| T15 | [#58](https://github.com/sjarso1/ADEPT-Microscopy/issues/58) | DR6 | 11 | `S1-ID-3`, `endpoint:minimum`, `endpoint:target`, `endpoint:maximum` | Held-out validation against the reference standard: sensitivity, specificity, AUROC, κ with CIs — overall and per instrument |
| T16 | [#62](https://github.com/sjarso1/ADEPT-Microscopy/issues/62) | DR6 | 12 | `S1-ID-3`, `endpoint:minimum`, `endpoint:target` | Per-criterion ablation, ROI detection and prioritisation, S1-ID-3 report and tag `s1-id3` |
| T17 | [#66](https://github.com/sjarso1/ADEPT-Microscopy/issues/66) | DR7 | 13 | `S1-final`, `endpoint:minimum` | Final report synthesis: one validation story from calibration to concordance, figures, TRIPOD-AI and STARD checklists |
| T18 | [#70](https://github.com/sjarso1/ADEPT-Microscopy/issues/70) | DR7 | 14 | `S1-final`, `endpoint:minimum` | Final report submission and Semester 1 archive: tag `s1-final`, reproducibility audit, final supervision meeting |

Each ticket's sub-issues are the three issues numbered immediately after it (for example T05 is #18 and its sub-issues are #19–#21). The one exception is T02 (#5), whose sub-issues are #7–#9 because #6 was created twice during setup and closed as a duplicate.

## Labels

| Label | Meaning |
|---|---|
| `S1-ID-1` | Instrument calibrations and the canonical image space |
| `S1-ID-2` | Five criterion modules and invariance verification |
| `S1-ID-3` | Integrated classifier, reference-standard validation, ROI |
| `S1-final` | Final report and Semester 1 archive |
| `setup` | Environment, reading, reference standard |
| `endpoint:minimum` | Required for the Minimum endpoint |
| `endpoint:target` | Required for the Target endpoint |
| `endpoint:maximum` | Required only for the Maximum endpoint |
| `depends:hardware` | Blocked on the hardware workstream (kit images, paired acquisitions) |
| `depends:validation` | Blocked on the validation workstream (labels, pathologist time) |

## Board fields

Besides GitHub's built-in Status, Milestone, Labels, Parent issue and Sub-issues progress, the project carries three custom fields: **Design Review** (DR1–DR7), **Week** (1–14) and **Kind** (Ticket or Sub-issue). Filter the board on `Kind:Ticket` to see the 18-card view; group by Design Review to see one column per review.

## Working the board

A ticket moves to *In Progress* when its first sub-issue is started and to *Done* when all sub-issues are closed and its evidence has been shown at the design review. Sub-issues close through pull requests that reference them (`Closes #NN`). Tickets that end a deliverable (T08, T13, T16, T18) also require the annotated tag named in the ticket.
