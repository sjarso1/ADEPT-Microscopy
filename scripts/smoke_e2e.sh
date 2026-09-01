#!/usr/bin/env bash
# End-to-end smoke test on synthetic data: dataset → calibration → invariance →
# assessment → split → package → combiner experiment. Used by CI.
set -euo pipefail
ROOT=$(mktemp -d)
echo "workdir $ROOT"
adept synth make-dataset --out "$ROOT/synth" --n-slides 40 --seed 3
for inst in openflexure printed3d wsi scanner; do
  adept calibrate derive --kit "$ROOT/synth/kits/$inst" --instrument $inst --out "$ROOT/cal/$inst.json" > /dev/null
done
adept calibrate validate --pairs "$ROOT/synth/pairs.csv" --calibrations "$ROOT/cal" --out "$ROOT/invariance.json" > /dev/null
python - "$ROOT/invariance.json" <<'PY'
import json, sys
t = json.load(open(sys.argv[1]))
for k, v in t.items():
    print(f"{k}: inter/intra ratio {v['ratio']:.2f} CI {v['ratio_ci95']}")
    assert v["ratio"] < 1.0, f"{k} is not instrument-invariant after calibration"
PY
adept assess run --image "$ROOT/synth/slides/S0001_openflexure.png" --calibration "$ROOT/cal/openflexure.json" \
  --criteria-config configs/models/criteria_synthetic.yaml --out "$ROOT/report.json"
adept package write --report "$ROOT/report.json" --image "$ROOT/synth/slides/S0001_openflexure.png" \
  --calibration "$ROOT/cal/openflexure.json" --out "$ROOT/packages"
adept splits make --registry "$ROOT/synth/registry.csv" --out "$ROOT/split.json" --seed 7
python scripts/s1_train_combiner.py --data "$ROOT/synth" --criteria configs/models/criteria_synthetic.yaml \
  --split "$ROOT/split.json" --out "$ROOT/results" --n-boot 200
echo "smoke OK"
