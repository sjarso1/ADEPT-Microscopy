# TRIPOD+AI checklist — ADEPT adequacy classifier

Fill in the "Where reported" column for each report (S1 final, S2 final, S3
final). Items follow TRIPOD+AI (Collins et al., BMJ 2024). Wording is
abbreviated; consult the published checklist for the full text.

| # | Section | Item (abbreviated) | Where reported |
|---|---------|--------------------|----------------|
| 1 | Title | Identify the study as developing/evaluating a prediction model, target population, outcome | |
| 2 | Abstract | Structured summary incl. AI-specific elements | |
| 3a | Introduction | Clinical context and rationale (LMIC telepathology; B1 waste) | |
| 3b | Introduction | Objectives: development, evaluation, or both; instruments and sites | |
| 4 | Methods — data | Data sources, sites, instruments, dates; paired-acquisition design | |
| 5 | Methods — participants | Eligibility of slides/cases; tissue-bank archive | |
| 6 | Methods — data prep | Calibration, canonical space, normalisation (where applied), tiling | |
| 7 | Methods — outcome | Reference standard (pathologist B1 vs B2+), κ ≥ 0.75, adjudication | |
| 8 | Methods — predictors | Five criteria; raw pixels for CNN arms; how measured | |
| 9 | Methods — sample size | Justification (number of slides, events per instrument) | |
| 10 | Methods — missing data | Handling of failed acquisitions | |
| 11 | Methods — analytical | Model type(s), training, hyper-parameters, class weighting | |
| 12 | Methods — analytical | Evaluation: locked slide-level split, LOIO, bootstrap CIs, calibration | |
| 13 | Methods — analytical | Threshold selection (validation set, target sensitivity) | |
| 14 | Methods — fairness | Performance by instrument, site (and any available demographic strata) | |
| 15 | Methods — open science | Code, weights, calibrations, splits, dataset (Zenodo DOI) | |
| 16 | Methods — ethics | Approvals, amendments (S3 diagnostic use) | |
| 17 | Results — participants | Flow of slides through splits; per-instrument counts | |
| 18 | Results — model | Full specification; the frozen model id | |
| 19 | Results — performance | Sens/spec/AUROC/κ with CIs, overall, per instrument, LOIO | |
| 20 | Results — updating | Adaptation / re-training (S2) | |
| 21 | Discussion | Interpretation, limitations, translatability claims and their limits | |
| 22 | Discussion | Usability: onboarding protocol, envelope, human oversight (pathologist diagnoses) | |
| 23 | Other | Registration, protocol, funding, conflicts | |
