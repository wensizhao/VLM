#!/usr/bin/env python3
"""Execute a reproducible experiment matrix; any failed run causes nonzero exit."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf
from train.experiments import run_matrix
from utils.config import load_experiment_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--sweep", default="configs/sweep.yaml")
    parser.add_argument("--output", default="results/stage2_sweep")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    records = run_matrix(load_experiment_config(args.config), OmegaConf.load(args.sweep),
                         Path(args.output).resolve(), args.dry_run)
    for record in records:
        print(f"{record['run_name']}: {record['status']}", flush=True)
    if any(r["status"] == "failed" for r in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
