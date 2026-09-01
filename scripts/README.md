# scripts/

One entry point per experiment; each reads a YAML from `configs/experiments/`
and writes `results/<name>/metrics.json` with the config hash embedded.

| Script | Deliverable | Status |
|--------|-------------|--------|
| `smoke_e2e.sh` | CI | runs |
| `s1_train_combiner.py` | S1-ID-3 (Minimum) | runs on synthetic data |
| `s1_train_cnn.py` | S1-ID-3 architecture experiment | stub — needs `[torch]` |
| `s2_three_arm.py` | S2-ID-1 | stub |
| `s2_envelope.py` | S2-ID-2 | stub (harness in `adept.eval.envelope`) |
| `s2_onboarding_report.py` | S2-ID-3 | stub |
| `s3_extract_labels.py` | S3-ID-1 | stub |
| `s3_train_mil.py` | S3-ID-2 | stub — needs `[torch]` |
| `s3_attention_concordance.py` | S3-ID-3 | stub |
