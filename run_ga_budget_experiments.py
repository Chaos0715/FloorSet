#!/usr/bin/env python3
"""Run GA budget/elite-ratio experiments for FloorSet my_optimizer.

This runner intentionally uses evaluator subprocesses so each testcase gets a
fresh optimizer instance and a controlled FLOORSET_RANDOM_SEED.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


def parse_list(text, cast=str):
    return [cast(x.strip()) for x in text.split(',') if x.strip()]


def load_result(path):
    with open(path) as f:
        d = json.load(f)
    return d['test_results'][0]


def stats(xs):
    if not xs:
        return 0.0, 0.0
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    s = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
    return m, s


def weighted(costs, blocks):
    if not costs:
        return 0.0
    mx = max(blocks)
    ws = [math.exp((n - mx) / 12.0) for n in blocks]
    return sum(c * w for c, w in zip(costs, ws)) / sum(ws)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', default='30x30:0.05,30x30:0.10,30x30:0.20')
    ap.add_argument('--seeds', default='101,202,303,404,505')
    ap.add_argument('--test-ids', default='0,1,2,3,4,5,6,7,8,9,90,91,92,93,94,95,96,97,98,99')
    ap.add_argument('--out-dir', default='ga_budget_experiments')
    ap.add_argument('--optimizer', default='iccad2026contest/my_optimizer.py')
    ap.add_argument('--evaluator', default='iccad2026contest/iccad2026_evaluate.py')
    ap.add_argument('--baseline', default='my_optimizer_default_no_bstar_results.json')
    ap.add_argument('--continue-on-error', action='store_true')
    args = ap.parse_args()

    seeds = parse_list(args.seeds, int)
    test_ids = parse_list(args.test_ids, int)
    configs = []
    for item in parse_list(args.configs, str):
        budget, elite = item.split(':')
        pop, gens = budget.lower().split('x')
        configs.append({'name': f'ga{pop}x{gens}_e{elite}', 'pop': int(pop), 'gens': int(gens), 'elite': float(elite)})

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    with open(args.baseline) as f:
        baseline = json.load(f)
    baseline_by = {r['test_id']: r for r in baseline['test_results']}

    all_summary = []
    for cfg in configs:
        cfg_dir = out_root / cfg['name']
        cfg_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {cfg['name']} pop={cfg['pop']} gens={cfg['gens']} elite={cfg['elite']} ===", flush=True)
        seed_scores = []
        seed_avg_costs = []
        seed_avg_runtimes = []
        per_case = {tid: [] for tid in test_ids}

        for seed in seeds:
            seed_costs = []
            seed_blocks = []
            seed_runtimes = []
            for tid in test_ids:
                out = cfg_dir / f'seed{seed}_t{tid}.json'
                if not out.exists():
                    env = os.environ.copy()
                    env.update({
                        'PYTHONUNBUFFERED': '1',
                        'FLOORSET_RANDOM_SEED': str(seed),
                        'FLOORSET_QUALITY_GA_MIN_BLOCKS': '0',
                        'FLOORSET_GA_POP': str(cfg['pop']),
                        'FLOORSET_GA_GENS': str(cfg['gens']),
                        'FLOORSET_GA_ELITE_RATIO': str(cfg['elite']),
                    })
                    cmd = [sys.executable, args.evaluator, '--evaluate', args.optimizer, '--test-id', str(tid), '--output', str(out)]
                    t0 = time.time()
                    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    wall = time.time() - t0
                    (cfg_dir / f'seed{seed}.log').open('a').write(proc.stdout + f"\n# wall {wall:.3f}s tid={tid}\n")
                    if proc.returncode != 0:
                        if args.continue_on_error:
                            print(f"FAIL cfg={cfg['name']} seed={seed} tid={tid} rc={proc.returncode}", flush=True)
                            continue
                        print(proc.stdout)
                        raise SystemExit(proc.returncode)
                r = load_result(out)
                seed_costs.append(r['cost'])
                seed_blocks.append(r['block_count'])
                seed_runtimes.append(r['runtime_seconds'])
                per_case[tid].append(r)
                print(f"{cfg['name']} seed={seed} tid={tid} cost={r['cost']:.4f} rt={r['runtime_seconds']:.2f}s", flush=True)

            seed_scores.append(weighted(seed_costs, seed_blocks))
            seed_avg_costs.append(sum(seed_costs) / len(seed_costs))
            seed_avg_runtimes.append(sum(seed_runtimes) / len(seed_runtimes))
            print(f"seed {seed}: weighted={seed_scores[-1]:.4f} avg={seed_avg_costs[-1]:.4f} rt={seed_avg_runtimes[-1]:.2f}s", flush=True)

        old_costs = [baseline_by[tid]['cost'] for tid in test_ids]
        blocks = [baseline_by[tid]['block_count'] for tid in test_ids]
        old_weighted = weighted(old_costs, blocks)
        wm, ws = stats(seed_scores)
        am, astd = stats(seed_avg_costs)
        rm, rstd = stats(seed_avg_runtimes)
        rows = []
        for tid in test_ids:
            costs = [r['cost'] for r in per_case[tid]]
            runtimes = [r['runtime_seconds'] for r in per_case[tid]]
            cm, cs = stats(costs)
            rt, rts = stats(runtimes)
            old = baseline_by[tid]['cost']
            rows.append({
                'test_id': tid,
                'block_count': baseline_by[tid]['block_count'],
                'old_cost': old,
                'cost_mean': cm,
                'cost_std': cs,
                'cost_best': min(costs) if costs else None,
                'improvement_mean': old - cm,
                'improvement_best': old - min(costs) if costs else None,
                'runtime_mean': rt,
                'runtime_std': rts,
            })
        summary = {
            'config': cfg,
            'seeds': seeds,
            'test_ids': test_ids,
            'old_weighted': old_weighted,
            'weighted_mean': wm,
            'weighted_std': ws,
            'weighted_best_seed': min(seed_scores) if seed_scores else None,
            'weighted_improvement_mean': old_weighted - wm,
            'weighted_improvement_best_seed': old_weighted - min(seed_scores) if seed_scores else None,
            'avg_cost_mean': am,
            'avg_cost_std': astd,
            'avg_runtime_mean': rm,
            'avg_runtime_std': rstd,
            'seed_weighted': dict(zip(map(str, seeds), seed_scores)),
            'seed_avg_cost': dict(zip(map(str, seeds), seed_avg_costs)),
            'seed_avg_runtime': dict(zip(map(str, seeds), seed_avg_runtimes)),
            'per_case': rows,
        }
        (cfg_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
        all_summary.append(summary)
        print(json.dumps({k: summary[k] for k in ['old_weighted','weighted_mean','weighted_std','weighted_best_seed','weighted_improvement_mean','weighted_improvement_best_seed','avg_runtime_mean']}, indent=2), flush=True)

    (out_root / 'summary.json').write_text(json.dumps(all_summary, indent=2))
    print(f"\nSaved {out_root / 'summary.json'}", flush=True)


if __name__ == '__main__':
    main()
