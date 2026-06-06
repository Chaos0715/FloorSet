#!/usr/bin/env python3
"""Run full-validation ablations for the staged FloorSet soft repair pipeline."""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REFERENCE_SCORE = 7.1478597754195805
FULL_STAGES = "mib,boundary,cluster,gravity"

CONFIGS = {
    "baseline_soft_prior": {
        "FLOORSET_SOFT_REPAIR_STAGES": "none",
        "FLOORSET_GA_RESCUE": "0",
        "FLOORSET_QUALITY_GA_MIN_BLOCKS": "1000000",
    },
    "boundary_only": {
        "FLOORSET_SOFT_REPAIR_STAGES": "boundary",
        "FLOORSET_GA_RESCUE": "0",
        "FLOORSET_QUALITY_GA_MIN_BLOCKS": "1000000",
    },
    "gravity_only": {
        "FLOORSET_SOFT_REPAIR_STAGES": "gravity",
        "FLOORSET_GA_RESCUE": "0",
        "FLOORSET_QUALITY_GA_MIN_BLOCKS": "1000000",
    },
    "boundary_gravity": {
        "FLOORSET_SOFT_REPAIR_STAGES": "boundary,gravity",
        "FLOORSET_GA_RESCUE": "0",
        "FLOORSET_QUALITY_GA_MIN_BLOCKS": "1000000",
    },
    "full_soft_repair": {
        "FLOORSET_SOFT_REPAIR_STAGES": FULL_STAGES,
        "FLOORSET_GA_RESCUE": "0",
        "FLOORSET_QUALITY_GA_MIN_BLOCKS": "1000000",
    },
    "full_soft_repair_ga_rescue": {
        "FLOORSET_SOFT_REPAIR_STAGES": FULL_STAGES,
        "FLOORSET_GA_RESCUE": "1",
        "FLOORSET_QUALITY_GA_MIN_BLOCKS": "1000000",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default=",".join(CONFIGS), help="Comma-separated config names")
    parser.add_argument("--out-dir", default="repair_ablation_results")
    parser.add_argument("--optimizer", default="iccad2026contest/my_optimizer.py")
    parser.add_argument("--evaluator", default="iccad2026contest/iccad2026_evaluate.py")
    parser.add_argument("--data-path", default="../")
    parser.add_argument("--reference-score", type=float, default=REFERENCE_SCORE)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def load_summary(path: Path):
    with path.open() as f:
        data = json.load(f)
    summary = data.get("summary", {})
    return {
        "total_score": data.get("total_score"),
        "num_tests": summary.get("num_tests"),
        "num_feasible": summary.get("num_feasible"),
        "avg_cost": summary.get("avg_cost"),
        "avg_runtime": summary.get("avg_runtime"),
    }


def main():
    args = parse_args()
    names = [name.strip() for name in args.configs.split(",") if name.strip()]
    unknown = [name for name in names if name not in CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown configs: {', '.join(unknown)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for name in names:
        env = os.environ.copy()
        env.update(CONFIGS[name])
        env.pop("FLOORSET_FLOW_FILTER", None)
        target = out_dir / f"{name}_results.json"
        cmd = [
            sys.executable,
            args.evaluator,
            "--evaluate",
            args.optimizer,
            "--data-path",
            args.data_path,
            "--output",
            str(target),
        ]
        print(f"\n=== Running repair ablation: {name} ===", flush=True)
        print(json.dumps(CONFIGS[name], indent=2), flush=True)
        start = time.time()
        proc = subprocess.run(cmd, env=env)
        elapsed = time.time() - start
        if proc.returncode != 0:
            row = {"config": name, "returncode": proc.returncode, "elapsed_wall": elapsed}
            rows.append(row)
            if not args.continue_on_error:
                raise SystemExit(proc.returncode)
            continue

        summary = load_summary(target)
        score = summary.get("total_score")
        delta = None if score is None else score - args.reference_score
        row = {
            "config": name,
            "returncode": proc.returncode,
            "elapsed_wall": elapsed,
            "score_delta_vs_reference": delta,
            **summary,
        }
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    csv_path = out_dir / "summary.csv"
    json_path = out_dir / "summary.json"
    fields = [
        "config",
        "returncode",
        "total_score",
        "score_delta_vs_reference",
        "num_tests",
        "num_feasible",
        "avg_cost",
        "avg_runtime",
        "elapsed_wall",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    with json_path.open("w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved {csv_path} and {json_path}", flush=True)


if __name__ == "__main__":
    main()
