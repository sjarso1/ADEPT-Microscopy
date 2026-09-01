#!/usr/bin/env python
"""s3_train_mil — see scripts/README.md and docs/roadmap.md.

Stub: reads its experiment config and exits. Implement in the deliverable that
owns it, keeping the pattern of scripts/s1_train_combiner.py (config in,
metrics.json with bootstrap CIs out, nothing reported without an error bar).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    name = cfg.get("name", args.config.stem)
    raise SystemExit(f"{name}: not implemented yet — see docs/roadmap.md")


if __name__ == "__main__":
    main()
