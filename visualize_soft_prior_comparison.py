#!/usr/bin/env python3
"""Build an HTML comparison for soft-prior, ground truth, and B*tree layouts."""

import argparse
import html
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch

from iccad2026contest.iccad2026_evaluate import (
    FloorplanDatasetLiteTest,
    calculate_bbox_area,
    calculate_hpwl_b2b,
    calculate_hpwl_p2b,
    compute_total_score,
    evaluate_solution,
)

Position = Tuple[float, float, float, float]


DEFAULT_CASES = [0, 25, 50, 75, 90, 95, 99]
DEFAULT_CHECKPOINT = "models/soft_prior_h384_mp6_snapshots/floorset_soft_prior_h384_mp6_step70900.pt"


def parse_cases(text: str) -> List[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def load_eval_results(path: Path) -> Dict:
    with path.open() as f:
        data = json.load(f)
    data["by_test_id"] = {int(r["test_id"]): r for r in data.get("test_results", [])}
    return data


def to_positions(raw: Iterable[Iterable[float]]) -> List[Position]:
    return [tuple(float(v) for v in row[:4]) for row in raw]


def extract_ground_truth(labels, block_count: int) -> List[Position]:
    polygons, _ = labels
    positions: List[Position] = []
    for i in range(block_count):
        block = polygons[i]
        valid = block[block[:, 0] != -1]
        if len(valid) > 0:
            x_min, y_min = valid.min(dim=0).values
            x_max, y_max = valid.max(dim=0).values
            positions.append(
                (
                    float(x_min),
                    float(y_min),
                    float(x_max - x_min),
                    float(y_max - y_min),
                )
            )
        else:
            positions.append((0.0, 0.0, 1.0, 1.0))
    return positions


def baseline_from_labels(labels, gt_positions, b2b_conn, p2b_conn, pins_pos) -> Dict[str, float]:
    _, metrics = labels
    hpwl_b2b = calculate_hpwl_b2b(gt_positions, b2b_conn)
    hpwl_p2b = calculate_hpwl_p2b(gt_positions, p2b_conn, pins_pos)
    area = calculate_bbox_area(gt_positions)

    if metrics is not None and len(metrics) >= 8:
        if metrics[0] > 0:
            area = float(metrics[0])
        if metrics[-2] > 0:
            hpwl_b2b = float(metrics[-2])
        if metrics[-1] >= 0:
            hpwl_p2b = float(metrics[-1])

    return {"hpwl_baseline": hpwl_b2b + hpwl_p2b, "area_baseline": area}


def metric_dict(metrics) -> Dict[str, float]:
    return {
        "is_feasible": bool(metrics.is_feasible),
        "cost": float(metrics.cost),
        "hpwl_gap": float(metrics.hpwl_gap),
        "area_gap": float(metrics.area_gap),
        "violations_relative": float(metrics.violations_relative),
        "boundary_violations": int(metrics.boundary_violations),
        "grouping_violations": int(metrics.grouping_violations),
        "mib_violations": int(metrics.mib_violations),
        "overlap_violations": int(metrics.overlap_violations),
        "area_violations": int(metrics.area_violations),
        "dimension_violations": int(metrics.dimension_violations),
        "bbox_area": float(metrics.bbox_area),
        "hpwl_total": float(metrics.hpwl_total),
    }


def evaluate_positions(
    positions: List[Position],
    baseline: Dict[str, float],
    inputs,
    gt_positions: List[Position],
) -> Dict[str, float]:
    area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
    metrics = evaluate_solution(
        {"positions": positions, "runtime": 1.0},
        baseline,
        constraints,
        b2b_conn,
        p2b_conn,
        pins_pos,
        area_target,
        gt_positions,
        median_runtime=1.0,
    )
    return metric_dict(metrics)


def compute_ground_truth_summary(dataset) -> Dict:
    costs = []
    blocks = []
    feasible = 0
    selected = {}
    for test_id in range(len(dataset)):
        sample = dataset[test_id]
        inputs, labels = sample["input"], sample["label"]
        area_target, b2b_conn, p2b_conn, pins_pos, _ = inputs
        block_count = int((area_target != -1).sum().item())
        gt_positions = extract_ground_truth(labels, block_count)
        baseline = baseline_from_labels(labels, gt_positions, b2b_conn, p2b_conn, pins_pos)
        metrics = evaluate_positions(gt_positions, baseline, inputs, gt_positions)
        costs.append(metrics["cost"])
        blocks.append(block_count)
        feasible += int(metrics["is_feasible"])
        selected[test_id] = metrics
    return {
        "total_score": compute_total_score(costs, blocks),
        "num_tests": len(costs),
        "num_feasible": feasible,
        "avg_cost": sum(costs) / len(costs),
        "by_test_id": selected,
    }


def result_summary(data: Dict) -> Dict[str, float]:
    results = data.get("test_results", [])
    return {
        "total_score": float(data.get("total_score", math.nan)),
        "num_tests": int(data.get("summary", {}).get("num_tests", len(results))),
        "num_feasible": int(data.get("summary", {}).get("num_feasible", sum(1 for r in results if r.get("is_feasible")))),
        "avg_cost": float(data.get("summary", {}).get("avg_cost", sum(r["cost"] for r in results) / max(1, len(results)))),
    }


def fmt(value, digits=4) -> str:
    if value is None:
        return "-"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(value) or math.isinf(value):
        return "-"
    return f"{value:.{digits}f}"


def layout_bounds(position_sets: Iterable[List[Position]]) -> Tuple[float, float, float, float]:
    xs = []
    ys = []
    for positions in position_sets:
        for x, y, w, h in positions:
            xs.extend([x, x + w])
            ys.extend([y, y + h])
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span = max(max_x - min_x, max_y - min_y, 1.0)
    pad = span * 0.04
    return min_x - pad, min_y - pad, max_x + pad, max_y + pad


def constraint_stroke(constraint_row) -> Tuple[str, float]:
    fixed = bool(constraint_row[0].item()) if len(constraint_row) > 0 else False
    preplaced = bool(constraint_row[1].item()) if len(constraint_row) > 1 else False
    mib = int(constraint_row[2].item()) if len(constraint_row) > 2 else 0
    cluster = int(constraint_row[3].item()) if len(constraint_row) > 3 else 0
    boundary = int(constraint_row[4].item()) if len(constraint_row) > 4 else 0
    if preplaced:
        return "#7c3aed", 2.4
    if fixed:
        return "#2563eb", 2.2
    if boundary:
        return "#dc2626", 2.0
    if cluster:
        return "#f97316", 1.8
    if mib:
        return "#16a34a", 1.8
    return "#111827", 0.9


def svg_layout(
    positions: List[Position],
    pins_pos: torch.Tensor,
    constraints: torch.Tensor,
    bounds: Tuple[float, float, float, float],
    block_count: int,
) -> str:
    min_x, min_y, max_x, max_y = bounds
    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)
    font_size = max(width, height) / 46.0
    show_labels = block_count <= 75
    parts = [
        f'<svg viewBox="0 0 {width:.6f} {height:.6f}" role="img" aria-label="floorplan layout">'
    ]
    parts.append(f'<rect x="0" y="0" width="{width:.6f}" height="{height:.6f}" fill="#ffffff"/>')
    for i, (x, y, w, h) in enumerate(positions):
        vx = x - min_x
        vy = max_y - (y + h)
        hue = (i * 137.508) % 360
        fill = f"hsl({hue:.1f}, 62%, 75%)"
        stroke, stroke_width = constraint_stroke(constraints[i]) if i < len(constraints) else ("#111827", 0.9)
        title = html.escape(f"block {i}: x={x:.3f}, y={y:.3f}, w={w:.3f}, h={h:.3f}")
        parts.append(
            f'<rect x="{vx:.6f}" y="{vy:.6f}" width="{w:.6f}" height="{h:.6f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.4f}" vector-effect="non-scaling-stroke">'
            f"<title>{title}</title></rect>"
        )
        if show_labels and w > 0 and h > 0:
            parts.append(
                f'<text x="{vx + w / 2:.6f}" y="{vy + h / 2:.6f}" '
                f'font-size="{font_size:.6f}" text-anchor="middle" dominant-baseline="middle" '
                f'fill="#111827">{i}</text>'
            )

    pin_radius = max(width, height) / 180.0
    for pin in pins_pos:
        px, py = float(pin[0]), float(pin[1])
        if px < min_x or px > max_x or py < min_y or py > max_y:
            continue
        parts.append(
            f'<circle cx="{px - min_x:.6f}" cy="{max_y - py:.6f}" r="{pin_radius:.6f}" '
            f'fill="#0f766e" opacity="0.8"><title>pin ({px:.3f}, {py:.3f})</title></circle>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def make_method_card(name: str, summary: Dict, note: str) -> str:
    return f"""
    <section class="score-card">
      <div class="score-label">{html.escape(name)}</div>
      <div class="score-value">{fmt(summary.get("total_score"))}</div>
      <div class="score-meta">{summary.get("num_feasible", 0)}/{summary.get("num_tests", 0)} feasible</div>
      <div class="score-meta">avg cost {fmt(summary.get("avg_cost"))}</div>
      <p>{html.escape(note)}</p>
    </section>
    """


def make_case_table(cases: List[Dict]) -> str:
    rows = []
    for case in cases:
        gt = case["methods"]["ground_truth"]["metrics"]
        bstar = case["methods"]["bstar"]["metrics"]
        soft = case["methods"]["soft_prior"]["metrics"]
        delta = soft["cost"] - bstar["cost"]
        rows.append(
            "<tr>"
            f"<td>{case['test_id']}</td>"
            f"<td>{case['block_count']}</td>"
            f"<td>{fmt(gt['cost'])}</td>"
            f"<td>{fmt(bstar['cost'])}</td>"
            f"<td>{fmt(soft['cost'])}</td>"
            f"<td class=\"{'good' if delta < 0 else 'bad'}\">{fmt(delta)}</td>"
            f"<td>{fmt(soft['violations_relative'])}</td>"
            f"<td>{fmt(soft['hpwl_gap'])}</td>"
            f"<td>{fmt(soft['area_gap'])}</td>"
            "</tr>"
        )
    return """
    <table>
      <thead>
        <tr>
          <th>case</th><th>blocks</th><th>GT cost</th><th>B*tree cost</th>
          <th>soft-prior cost</th><th>soft - B*tree</th>
          <th>soft V_rel</th><th>soft HPWL gap</th><th>soft area gap</th>
        </tr>
      </thead>
      <tbody>
    """ + "\n".join(rows) + """
      </tbody>
    </table>
    """


def method_panel(title: str, method: Dict, svg: str) -> str:
    metrics = method["metrics"]
    return f"""
    <article class="panel">
      <header>
        <h3>{html.escape(title)}</h3>
        <div>cost {fmt(metrics['cost'])} | V_rel {fmt(metrics['violations_relative'])}</div>
        <div>HPWL gap {fmt(metrics['hpwl_gap'])} | area gap {fmt(metrics['area_gap'])}</div>
      </header>
      {svg}
    </article>
    """


def make_html(summary: Dict, cases: List[Dict]) -> str:
    cards = "\n".join(
        [
            make_method_card(
                "Ground truth",
                summary["full_scores"]["ground_truth"],
                "Validation labels re-scored with neutral local runtime.",
            ),
            make_method_card(
                "B*tree baseline",
                summary["full_scores"]["bstar"],
                "optimizer_template.py, 100 validation cases.",
            ),
            make_method_card(
                "Soft-prior step70900",
                summary["full_scores"]["soft_prior"],
                "Integrated selector result from soft_prior_eval_results.json.",
            ),
        ]
    )
    case_sections = []
    for case in cases:
        panels = "\n".join(
            [
                method_panel("Ground truth", case["methods"]["ground_truth"], case["methods"]["ground_truth"]["svg"]),
                method_panel("B*tree baseline", case["methods"]["bstar"], case["methods"]["bstar"]["svg"]),
                method_panel("Soft-prior step70900", case["methods"]["soft_prior"], case["methods"]["soft_prior"]["svg"]),
            ]
        )
        case_sections.append(
            f"""
            <section class="case-block">
              <div class="case-title">
                <h2>Validation case {case['test_id']}</h2>
                <span>{case['block_count']} blocks</span>
              </div>
              <div class="panels">{panels}</div>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Soft-prior step70900 comparison</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172033;
      --muted: #64748b;
      --line: #d7dee8;
      --paper: #f7f9fc;
      --soft: #ffffff;
      --accent: #0f766e;
      --good: #047857;
      --bad: #b91c1c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--paper);
      color: var(--ink);
    }}
    main {{
      max-width: 1540px;
      margin: 0 auto;
      padding: 28px;
    }}
    .topline {{
      display: flex;
      justify-content: space-between;
      gap: 20px;
      align-items: end;
      margin-bottom: 18px;
    }}
    h1, h2, h3, p {{ margin: 0; }}
    h1 {{ font-size: 30px; font-weight: 750; letter-spacing: 0; }}
    .subtitle {{ color: var(--muted); margin-top: 6px; line-height: 1.45; }}
    .stamp {{ color: var(--muted); font-size: 13px; text-align: right; }}
    .score-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin: 18px 0;
    }}
    .score-card {{
      background: var(--soft);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }}
    .score-label {{ color: var(--muted); font-size: 13px; font-weight: 700; text-transform: uppercase; }}
    .score-value {{ font-size: 34px; font-weight: 780; margin-top: 4px; }}
    .score-meta {{ color: var(--ink); font-size: 14px; margin-top: 4px; }}
    .score-card p {{ color: var(--muted); font-size: 13px; margin-top: 10px; line-height: 1.4; }}
    .table-wrap {{
      overflow: auto;
      background: var(--soft);
      border: 1px solid var(--line);
      border-radius: 8px;
      margin: 18px 0 24px;
    }}
    table {{ width: 100%; border-collapse: collapse; min-width: 900px; }}
    th, td {{
      padding: 10px 12px;
      text-align: right;
      border-bottom: 1px solid var(--line);
      font-variant-numeric: tabular-nums;
      font-size: 13px;
    }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ color: var(--muted); background: #eef3f8; font-size: 12px; text-transform: uppercase; }}
    tr:last-child td {{ border-bottom: 0; }}
    .good {{ color: var(--good); font-weight: 760; }}
    .bad {{ color: var(--bad); font-weight: 760; }}
    .case-block {{
      margin-top: 18px;
      background: var(--soft);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .case-title {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 12px;
      padding-bottom: 12px;
    }}
    .case-title h2 {{ font-size: 18px; }}
    .case-title span {{ color: var(--muted); font-size: 13px; }}
    .panels {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }}
    .panel {{
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #ffffff;
      min-width: 0;
    }}
    .panel header {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfe;
    }}
    .panel h3 {{ font-size: 15px; margin-bottom: 4px; }}
    .panel header div {{ color: var(--muted); font-size: 12px; line-height: 1.35; }}
    .panel svg {{
      display: block;
      width: 100%;
      height: 280px;
      background: #ffffff;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 16px;
      color: var(--muted);
      font-size: 13px;
      margin: 12px 0 4px;
    }}
    .legend span::before {{
      content: "";
      display: inline-block;
      width: 12px;
      height: 12px;
      border-radius: 2px;
      margin-right: 6px;
      vertical-align: -2px;
      border: 2px solid var(--c);
    }}
    @media (max-width: 980px) {{
      main {{ padding: 18px; }}
      .topline {{ display: block; }}
      .stamp {{ text-align: left; margin-top: 10px; }}
      .score-grid, .panels {{ grid-template-columns: 1fr; }}
      .panel svg {{ height: 340px; }}
    }}
  </style>
</head>
<body>
  <main>
    <div class="topline">
      <div>
        <h1>Soft-prior step70900 integrated selector comparison</h1>
        <p class="subtitle">Score and layout comparison against validation ground truth and the B*tree baseline.</p>
      </div>
      <div class="stamp">
        Generated {html.escape(summary['generated_at'])}<br>
        Checkpoint: {html.escape(summary['sources']['soft_checkpoint'])}
      </div>
    </div>
    <div class="score-grid">{cards}</div>
    <div class="legend">
      <span style="--c:#7c3aed">preplaced</span>
      <span style="--c:#2563eb">fixed-shape</span>
      <span style="--c:#dc2626">boundary</span>
      <span style="--c:#f97316">cluster</span>
      <span style="--c:#16a34a">MIB</span>
    </div>
    <div class="table-wrap">{make_case_table(cases)}</div>
    {''.join(case_sections)}
  </main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=".", help="FloorSet root containing LiteTensorDataTest")
    parser.add_argument("--soft-results", default="soft_prior_eval_results.json")
    parser.add_argument("--bstar-results", default="visualize_bstar_baseline_full.json")
    parser.add_argument("--soft-checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cases", default=",".join(str(c) for c in DEFAULT_CASES))
    parser.add_argument("--html-output", default="visualizations/soft_prior_step70900_comparison.html")
    parser.add_argument("--json-output", default="visualizations/soft_prior_step70900_comparison.json")
    args = parser.parse_args()

    case_ids = parse_cases(args.cases)
    soft_data = load_eval_results(Path(args.soft_results))
    bstar_data = load_eval_results(Path(args.bstar_results))
    dataset = FloorplanDatasetLiteTest(args.data_path)
    gt_summary = compute_ground_truth_summary(dataset)

    cases = []
    for test_id in case_ids:
        sample = dataset[test_id]
        inputs, labels = sample["input"], sample["label"]
        area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
        block_count = int((area_target != -1).sum().item())
        gt_positions = extract_ground_truth(labels, block_count)
        baseline = baseline_from_labels(labels, gt_positions, b2b_conn, p2b_conn, pins_pos)

        soft_result = soft_data["by_test_id"][test_id]
        bstar_result = bstar_data["by_test_id"][test_id]
        soft_positions = to_positions(soft_result["positions"])
        bstar_positions = to_positions(bstar_result["positions"])

        bounds = layout_bounds([gt_positions, bstar_positions, soft_positions])
        methods = {
            "ground_truth": {
                "positions": gt_positions,
                "metrics": gt_summary["by_test_id"][test_id],
            },
            "bstar": {
                "positions": bstar_positions,
                "metrics": evaluate_positions(bstar_positions, baseline, inputs, gt_positions),
            },
            "soft_prior": {
                "positions": soft_positions,
                "metrics": evaluate_positions(soft_positions, baseline, inputs, gt_positions),
            },
        }
        for method in methods.values():
            method["svg"] = svg_layout(
                method["positions"],
                pins_pos,
                constraints[:block_count],
                bounds,
                block_count,
            )
        cases.append({"test_id": test_id, "block_count": block_count, "methods": methods})

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sources": {
            "soft_results": args.soft_results,
            "bstar_results": args.bstar_results,
            "soft_checkpoint": args.soft_checkpoint,
        },
        "full_scores": {
            "ground_truth": {
                "total_score": gt_summary["total_score"],
                "num_tests": gt_summary["num_tests"],
                "num_feasible": gt_summary["num_feasible"],
                "avg_cost": gt_summary["avg_cost"],
            },
            "bstar": result_summary(bstar_data),
            "soft_prior": result_summary(soft_data),
        },
        "case_ids": case_ids,
    }

    html_output = Path(args.html_output)
    json_output = Path(args.json_output)
    html_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)

    json_ready_cases = []
    for case in cases:
        clean_methods = {}
        for name, method in case["methods"].items():
            clean_methods[name] = {
                "positions": method["positions"],
                "metrics": method["metrics"],
            }
        json_ready_cases.append(
            {
                "test_id": case["test_id"],
                "block_count": case["block_count"],
                "methods": clean_methods,
            }
        )

    json_output.write_text(
        json.dumps({"summary": summary, "cases": json_ready_cases}, indent=2),
        encoding="utf-8",
    )
    html_output.write_text(make_html(summary, cases), encoding="utf-8")

    print(f"Wrote {html_output}")
    print(f"Wrote {json_output}")
    print("Full validation scores:")
    print(f"  Ground truth: {summary['full_scores']['ground_truth']['total_score']:.6f}")
    print(f"  B*tree:       {summary['full_scores']['bstar']['total_score']:.6f}")
    print(f"  Soft-prior:   {summary['full_scores']['soft_prior']['total_score']:.6f}")


if __name__ == "__main__":
    main()
