#!/usr/bin/env python3
"""Run per-flow FloorSet optimizer ablations.

Each run sets FLOORSET_FLOW_FILTER to one flow name, invokes the official
evaluator, saves the generated my_optimizer_results.json, and writes a compact
CSV/JSON summary.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_FLOWS = [
    "shelf_area",
    "shelf_importance",
    "ml_greedy",
    "ml_cluster_greedy",
    "cluster_ga",
    "bstar_sa",
    "trained_prior_order",
    "trained_prior_cluster",
    "preferred",
    "learned_only",
    "no_bstar",
    "default",
    "all",
]

FLOW_ALIASES = {
    # Best single-flow prior plus the stronger constructive/clustering baselines.
    "preferred": "shelf_area,ml_cluster_greedy,cluster_ga,trained_prior_order,trained_prior_cluster",
    # Checks whether the learned-prior/ML cluster family can stand alone.
    "learned_only": "ml_greedy,ml_cluster_greedy,cluster_ga,trained_prior_order,trained_prior_cluster",
    # Full selector except the weakest B*-tree/SA-inspired baseline.
    "no_bstar": "shelf_area,shelf_importance,ml_greedy,ml_cluster_greedy,cluster_ga,trained_prior_order,trained_prior_cluster",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flows", default=",".join(DEFAULT_FLOWS), help="Comma-separated flow list")
    parser.add_argument("--out-dir", default="flow_ablation_results")
    parser.add_argument("--optimizer", default="iccad2026contest/my_optimizer.py")
    parser.add_argument("--evaluator", default="iccad2026contest/iccad2026_evaluate.py")
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
    flows = [x.strip() for x in args.flows.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for flow in flows:
        env = os.environ.copy()
        flow_filter = FLOW_ALIASES.get(flow, flow)
        if flow == "default":
            env.pop("FLOORSET_FLOW_FILTER", None)
        elif flow == "all":
            env["FLOORSET_FLOW_FILTER"] = "all"
        else:
            env["FLOORSET_FLOW_FILTER"] = flow_filter
        print(f"\n=== Running flow: {flow} ===", flush=True)
        start = time.time()
        cmd = [sys.executable, args.evaluator, "--evaluate", args.optimizer]
        proc = subprocess.run(cmd, env=env)
        elapsed = time.time() - start
        if proc.returncode != 0:
            if not args.continue_on_error:
                raise SystemExit(proc.returncode)
            rows.append({"flow": flow, "returncode": proc.returncode, "elapsed_wall": elapsed})
            continue

        result_path = Path("my_optimizer_results.json")
        if not result_path.exists():
            raise RuntimeError("Evaluator did not produce my_optimizer_results.json")
        target = out_dir / f"{flow}_results.json"
        shutil.copy2(result_path, target)
        summary = load_summary(target)
        row = {"flow": flow, "returncode": proc.returncode, "elapsed_wall": elapsed, **summary}
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    csv_path = out_dir / "summary.csv"
    json_path = out_dir / "summary.json"
    fields = ["flow", "returncode", "total_score", "num_tests", "num_feasible", "avg_cost", "avg_runtime", "elapsed_wall"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    with json_path.open("w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
