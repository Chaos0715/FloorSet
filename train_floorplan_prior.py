#!/usr/bin/env python3
"""Train an RTX-5090-only graph prior for FloorSet placement.

This script is intentionally strict: real training must run on an RTX 5090.
It exits immediately if PyTorch cannot initialize CUDA or if the selected GPU
is not an RTX 5090-class device.

Examples:
  CUDA_VISIBLE_DEVICES=0 .venv-rtx5090/bin/python train_floorplan_prior.py --smoke --num-samples 32
  CUDA_VISIBLE_DEVICES=0 .venv-rtx5090/bin/python train_floorplan_prior.py --num-samples 20000 --epochs 5
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "iccad2026contest"))

from iccad2026contest.iccad2026_evaluate import compute_training_loss_differentiable  # noqa: E402
from lite_dataset import FloorplanDatasetLite, floorplan_collate  # noqa: E402


RATIO_CANDIDATES = torch.tensor([1.0, 2.0, 0.5, 1.5, 2.0 / 3.0], dtype=torch.float32)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=".", help="FloorSet root containing floorset_lite/ after download")
    parser.add_argument("--checkpoint", default="models/floorset_prior.pt")
    parser.add_argument("--log", default="models/floorset_prior_train_log.jsonl")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=5000, help="Number of samples to use; <=0 means all local samples")
    parser.add_argument("--sample-mode", choices=["first", "spread", "random", "stratified"], default="stratified", help="How to choose --num-samples from the dataset")
    parser.add_argument("--seed", type=int, default=1337, help="Sampling/shuffle seed")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--message-passes", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--snapshot-dir", default=None, help="Optional directory for per-save checkpoints such as prior_step0500.pt")
    parser.add_argument("--smoke", action="store_true", help="Run one forward/backward step and save a smoke checkpoint")
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after this many optimizer steps and save a checkpoint")
    parser.add_argument("--time-budget-hours", type=float, default=None, help="Stop after this wall-clock budget and save a checkpoint")
    parser.add_argument("--resume", action="store_true", help="Resume model/optimizer state from --checkpoint if it exists")
    return parser.parse_args()


def require_rtx5090(device_arg: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Refusing CPU training. "
            f"torch={torch.__version__}, torch_cuda={torch.version.cuda}. "
            "Install a PyTorch CUDA build compatible with the RTX 5090 driver first."
        )

    visible = []
    for idx in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(idx)
        props = torch.cuda.get_device_properties(idx)
        visible.append((idx, name, props.total_memory / (1024 ** 3)))

    rtx5090 = [item for item in visible if "RTX 5090" in item[1]]
    if not rtx5090:
        raise RuntimeError(f"No RTX 5090 visible to PyTorch. Visible GPUs: {visible}")

    device = torch.device(device_arg)
    if device.type != "cuda":
        raise RuntimeError(f"Device must be CUDA/RTX 5090, got {device_arg!r}")

    idx = device.index if device.index is not None else torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    props = torch.cuda.get_device_properties(idx)
    total_gib = props.total_memory / (1024 ** 3)
    if "RTX 5090" not in name:
        raise RuntimeError(f"Selected GPU cuda:{idx} is not RTX 5090: {name}")
    if total_gib < 29.0:
        raise RuntimeError(f"Selected RTX 5090 reports too little memory: {total_gib:.2f} GiB")

    print("GPU check passed")
    print(f"  torch={torch.__version__}, torch_cuda={torch.version.cuda}")
    print(f"  device=cuda:{idx}, name={name}, memory={total_gib:.2f} GiB")
    print(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    return device


class FloorSetGraphPrior(nn.Module):
    """Small dependency-free message-passing model for placement priors."""

    def __init__(self, feature_dim: int, hidden: int = 128, message_passes: int = 4):
        super().__init__()
        self.message_passes = message_passes
        self.node_in = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.self_update = nn.Linear(hidden, hidden)
        self.msg_update = nn.Linear(hidden, hidden)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(message_passes)])
        self.center_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2))
        self.ratio_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, len(RATIO_CANDIDATES)))
        self.order_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 1))

    def forward(self, features: torch.Tensor, edges: torch.Tensor, weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.node_in(features)
        n = h.shape[0]
        for layer in range(self.message_passes):
            agg = torch.zeros_like(h)
            if edges.numel() > 0:
                src = edges[:, 0].long()
                dst = edges[:, 1].long()
                w = weights.view(-1, 1).to(h.dtype)
                agg.index_add_(0, src, h[dst] * w)
                agg.index_add_(0, dst, h[src] * w)
            h = h + F.silu(self.self_update(h) + self.msg_update(agg / math.sqrt(max(n, 1))))
            h = self.norms[layer](h)
        return {
            "center": self.center_head(h),
            "ratio_logits": self.ratio_head(h),
            "order": self.order_head(h).squeeze(-1),
        }


def valid_rows(t: torch.Tensor, cols: int) -> torch.Tensor:
    if t is None or t.numel() == 0:
        return torch.empty((0, cols), dtype=torch.float32)
    t = t.float()
    if t.dim() == 1:
        t = t.view(1, -1)
    return t[t[:, 0] >= 0]


def build_sample(
    area_target: torch.Tensor,
    b2b_conn: torch.Tensor,
    p2b_conn: torch.Tensor,
    pins_pos: torch.Tensor,
    constraints: torch.Tensor,
    tree_sol: torch.Tensor,
    fp_sol: torch.Tensor,
    metrics: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    area_target = area_target.squeeze(0) if area_target.dim() > 1 else area_target
    b2b_conn = b2b_conn.squeeze(0) if b2b_conn.dim() > 2 else b2b_conn
    p2b_conn = p2b_conn.squeeze(0) if p2b_conn.dim() > 2 else p2b_conn
    pins_pos = pins_pos.squeeze(0) if pins_pos.dim() > 2 else pins_pos
    constraints = constraints.squeeze(0) if constraints.dim() > 2 else constraints
    fp_sol = fp_sol.squeeze(0) if fp_sol.dim() > 2 else fp_sol
    metrics = metrics.squeeze(0) if metrics.dim() > 1 else metrics

    n = int((area_target != -1).sum().item())
    area = area_target[:n].float().clamp_min(1.0)
    constraints = constraints[:n].float()
    fp_sol = fp_sol[:n].float()

    b2b_valid = valid_rows(b2b_conn, 3)
    edge_mask = (b2b_valid[:, 0] < n) & (b2b_valid[:, 1] < n) if b2b_valid.numel() else torch.zeros(0, dtype=torch.bool)
    b2b_valid = b2b_valid[edge_mask]

    p2b_valid = valid_rows(p2b_conn, 3)
    pin_mask = (p2b_valid[:, 1] < n) & (p2b_valid[:, 0] < pins_pos.shape[0]) if p2b_valid.numel() else torch.zeros(0, dtype=torch.bool)
    p2b_valid = p2b_valid[pin_mask]

    b_degree = torch.zeros(n)
    if b2b_valid.numel():
        for edge in b2b_valid:
            i, j, w = int(edge[0]), int(edge[1]), float(edge[2])
            if w > 0:
                b_degree[i] += w
                b_degree[j] += w

    p_degree = torch.zeros(n)
    pin_cx = torch.zeros(n)
    pin_cy = torch.zeros(n)
    if p2b_valid.numel():
        for edge in p2b_valid:
            pin, block, w = int(edge[0]), int(edge[1]), float(edge[2])
            if w > 0 and pins_pos[pin, 0] != -1 and pins_pos[pin, 1] != -1:
                p_degree[block] += w
                pin_cx[block] += pins_pos[pin, 0].float() * w
                pin_cy[block] += pins_pos[pin, 1].float() * w
    has_pin = p_degree > 0
    pin_cx[has_pin] /= p_degree[has_pin]
    pin_cy[has_pin] /= p_degree[has_pin]

    scale = torch.sqrt(area.sum()).clamp_min(1.0)
    target_w = fp_sol[:, 0].clamp_min(1e-3)
    target_h = fp_sol[:, 1].clamp_min(1e-3)
    target_x = fp_sol[:, 2]
    target_y = fp_sol[:, 3]
    target_center = torch.stack([(target_x + target_w / 2) / scale, (target_y + target_h / 2) / scale], dim=1)
    target_ratio = target_w / target_h
    ratio_idx = torch.argmin(torch.abs(target_ratio[:, None] - RATIO_CANDIDATES[None, :]), dim=1)
    placement_key = target_center[:, 0] + target_center[:, 1]
    target_order = torch.zeros(n, dtype=torch.float32)
    if n > 1:
        sorted_idx = torch.argsort(placement_key, descending=False)
        target_order[sorted_idx] = torch.linspace(1.0, 0.0, steps=n)
    else:
        target_order.fill_(1.0)

    feat = torch.stack(
        [
            area / area.max().clamp_min(1.0),
            torch.sqrt(area) / torch.sqrt(area.max()).clamp_min(1.0),
            b_degree / b_degree.max().clamp_min(1.0),
            p_degree / p_degree.max().clamp_min(1.0),
            has_pin.float(),
            pin_cx / scale,
            pin_cy / scale,
            (constraints[:, 0] != 0).float() if constraints.shape[1] > 0 else torch.zeros(n),
            (constraints[:, 1] != 0).float() if constraints.shape[1] > 1 else torch.zeros(n),
            (constraints[:, 2] > 0).float() if constraints.shape[1] > 2 else torch.zeros(n),
            (constraints[:, 3] > 0).float() if constraints.shape[1] > 3 else torch.zeros(n),
            (constraints[:, 4] > 0).float() if constraints.shape[1] > 4 else torch.zeros(n),
        ],
        dim=1,
    )

    edges = b2b_valid[:, :2].long() if b2b_valid.numel() else torch.empty((0, 2), dtype=torch.long)
    weights = b2b_valid[:, 2].float() if b2b_valid.numel() else torch.empty(0)
    if weights.numel() > 0:
        weights = weights / weights.max().clamp_min(1.0)

    return (
        feat.to(device),
        edges.to(device),
        weights.to(device),
        area.to(device),
        target_center.to(device),
        ratio_idx.to(device),
        target_order.to(device),
        b2b_conn.to(device),
        p2b_conn.to(device),
        pins_pos.to(device),
        metrics.to(device),
    )


def predictions_to_positions(pred: Dict[str, torch.Tensor], area: torch.Tensor) -> torch.Tensor:
    center = pred["center"]
    ratio_probs = F.softmax(pred["ratio_logits"], dim=-1)
    ratios = (ratio_probs * RATIO_CANDIDATES.to(area.device)[None, :]).sum(dim=1).clamp_min(1e-4)
    w = torch.sqrt(area * ratios)
    h = torch.sqrt(area / ratios)
    scale = torch.sqrt(area.sum()).clamp_min(1.0)
    cx = center[:, 0] * scale
    cy = center[:, 1] * scale
    return torch.stack([cx - w / 2, cy - h / 2, w, h], dim=1)


def compute_sample_loss(model, sample, device):
    (
        feat,
        edges,
        weights,
        area,
        target_center,
        ratio_idx,
        target_order,
        b2b_conn,
        p2b_conn,
        pins_pos,
        metrics,
    ) = build_sample(*sample, device=device)

    pred = model(feat, edges, weights)
    positions = predictions_to_positions(pred, area)
    center_loss = F.smooth_l1_loss(pred["center"], target_center)
    ratio_loss = F.cross_entropy(pred["ratio_logits"], ratio_idx)
    order_prob = torch.sigmoid(pred["order"])
    order_loss = F.smooth_l1_loss(order_prob, target_order)
    if area.numel() > 1:
        order_delta = pred["order"][:, None] - pred["order"][None, :]
        target_delta = target_order[:, None] - target_order[None, :]
        pair_mask = target_delta.abs() > 1e-6
        if pair_mask.any():
            pair_order_loss = F.softplus(-torch.sign(target_delta[pair_mask]) * order_delta[pair_mask]).mean()
        else:
            pair_order_loss = order_loss.new_tensor(0.0)
    else:
        pair_order_loss = order_loss.new_tensor(0.0)
    proxy_cost = compute_training_loss_differentiable(positions, b2b_conn, p2b_conn, pins_pos, area, metrics)
    loss = center_loss + 0.2 * ratio_loss + 0.35 * order_loss + 0.05 * pair_order_loss + 0.05 * proxy_cost

    stats = {
        "loss": float(loss.detach().cpu()),
        "center_loss": float(center_loss.detach().cpu()),
        "ratio_loss": float(ratio_loss.detach().cpu()),
        "order_loss": float(order_loss.detach().cpu()),
        "pair_order_loss": float(pair_order_loss.detach().cpu()),
        "proxy_cost": float(proxy_cost.detach().cpu()),
        "blocks": int(area.numel()),
    }
    return loss, stats


def train_step(model, batch, device, optimizer=None):
    batch_size = int(batch[0].shape[0]) if batch and batch[0].dim() > 0 else 1
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    totals: Dict[str, float] = {}
    total_blocks = 0
    sample_count = 0
    for idx in range(batch_size):
        sample = tuple(t[idx] for t in batch)
        loss, stats = compute_sample_loss(model, sample, device)
        if optimizer is not None:
            (loss / max(batch_size, 1)).backward()
        for key, value in stats.items():
            if key == "blocks":
                continue
            totals[key] = totals.get(key, 0.0) + float(value)
        total_blocks += int(stats["blocks"])
        sample_count += 1

    if optimizer is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    denom = max(sample_count, 1)
    averaged = {key: value / denom for key, value in totals.items()}
    averaged["blocks"] = float(total_blocks) / denom
    averaged["samples"] = sample_count
    averaged["total_blocks"] = total_blocks
    return averaged


def save_checkpoint(path: Path, model, optimizer, args, step: int, epoch: int, stats: Dict[str, float]):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "args": vars(args),
            "ratio_candidates": RATIO_CANDIDATES.tolist(),
            "feature_dim": 12,
            "hidden": args.hidden,
            "message_passes": args.message_passes,
            "step": step,
            "epoch": epoch,
            "stats": stats,
            "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device()),
        },
        path,
    )


def save_snapshot(args, checkpoint: Path, model, optimizer, step: int, epoch: int, stats: Dict[str, float]):
    if not args.snapshot_dir:
        return
    snapshot_dir = Path(args.snapshot_dir)
    stem = checkpoint.stem
    snapshot_path = snapshot_dir / f"{stem}_step{step:05d}.pt"
    save_checkpoint(snapshot_path, model, optimizer, args, step, epoch, stats)


def choose_indices(total: int, count: int, mode: str, seed: int, layouts_per_file: int = 112):
    count = min(max(int(count), 0), total)
    if count >= total:
        return list(range(total))
    if mode == "first":
        return list(range(count))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    if mode == "random":
        return torch.randperm(total, generator=generator)[:count].tolist()
    if mode == "stratified":
        files = max(1, total // max(int(layouts_per_file), 1))
        per_file = count // files
        remainder = count % files
        indices = []
        file_order = torch.randperm(files, generator=generator).tolist()
        for rank, file_idx in enumerate(file_order):
            take = per_file + (1 if rank < remainder else 0)
            if take <= 0:
                continue
            offsets = torch.randperm(layouts_per_file, generator=generator)[:min(take, layouts_per_file)].tolist()
            base = file_idx * layouts_per_file
            indices.extend(min(total - 1, base + int(offset)) for offset in offsets)
        if len(indices) < count:
            seen = set(indices)
            for idx in torch.randperm(total, generator=generator).tolist():
                if idx not in seen:
                    indices.append(idx)
                    seen.add(idx)
                    if len(indices) >= count:
                        break
        return sorted(indices[:count])
    # Spread samples across the full pool; useful for deterministic smoke tests,
    # but random/stratified are better for real training.
    if count <= 1:
        return [0] if count == 1 else []
    step = (total - 1) / float(count - 1)
    return sorted({min(total - 1, int(round(i * step))) for i in range(count)})


def main():
    args = parse_args()
    device = require_rtx5090(args.device)

    dataset = FloorplanDatasetLite(args.data_path)
    full_sample_count = len(dataset)
    selected_indices = None
    if args.num_samples is not None and args.num_samples > 0:
        selected_indices = choose_indices(full_sample_count, args.num_samples, args.sample_mode, args.seed, getattr(dataset, "layouts_per_file", 112))
        dataset = Subset(dataset, selected_indices)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=not args.smoke, collate_fn=floorplan_collate)

    model = FloorSetGraphPrior(feature_dim=12, hidden=args.hidden, message_passes=args.message_passes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    checkpoint = Path(args.checkpoint)
    resume_step = 0
    resume_epoch = 0
    if args.resume and checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved.get("model_state", {}), strict=False)
        if saved.get("optimizer_state") is not None:
            try:
                optimizer.load_state_dict(saved["optimizer_state"])
            except Exception:
                pass
        resume_step = int(saved.get("step", 0) or 0)
        resume_epoch = int(saved.get("epoch", 0) or 0)
        print(f"Resumed checkpoint {checkpoint} at step={resume_step}, epoch={resume_epoch}")

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Training samples: {len(dataset)} / {full_sample_count}")
    print(f"Sample mode: {args.sample_mode}")
    print(f"Batch size: {args.batch_size}")
    print(f"Checkpoint: {checkpoint}")

    step = resume_step
    start = time.time()
    gpu_idx = device.index if device.index is not None else torch.cuda.current_device()
    gpu_props = torch.cuda.get_device_properties(gpu_idx)
    startup_event = {
        "event": "startup",
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "device": f"cuda:{gpu_idx}",
        "gpu_name": torch.cuda.get_device_name(gpu_idx),
        "gpu_memory_gib": gpu_props.total_memory / (1024 ** 3),
        "batch_size": args.batch_size,
        "checkpoint": str(checkpoint),
        "num_samples": len(dataset),
        "full_sample_count": full_sample_count,
        "sample_mode": args.sample_mode,
        "seed": args.seed,
        "time_budget_hours": args.time_budget_hours,
        "resume": args.resume,
        "resume_step": resume_step,
    }
    with log_path.open("a") as log_file:
        log_file.write(json.dumps(startup_event) + "\n")
        log_file.flush()
        for epoch in range(args.epochs):
            for batch in loader:
                stats = train_step(model, batch, device, optimizer=optimizer)
                step += 1
                stats.update({"step": step, "epoch": epoch, "elapsed": time.time() - start})
                log_file.write(json.dumps(stats) + "\n")
                log_file.flush()
                if step == 1 or step % 50 == 0:
                    print(
                        f"step={step} epoch={epoch} loss={stats['loss']:.4f} "
                        f"proxy={stats['proxy_cost']:.4f} samples={stats.get('samples', 1)} "
                        f"avg_blocks={stats['blocks']:.1f}"
                    )
                if args.time_budget_hours is not None and (time.time() - start) >= args.time_budget_hours * 3600.0:
                    save_checkpoint(checkpoint, model, optimizer, args, step, epoch, stats)
                    save_snapshot(args, checkpoint, model, optimizer, step, epoch, stats)
                    torch.cuda.synchronize(device)
                    print(f"Reached --time-budget-hours={args.time_budget_hours}; saved checkpoint: {checkpoint}", flush=True)
                    return
                if args.smoke:
                    save_checkpoint(checkpoint, model, optimizer, args, step, epoch, stats)
                    torch.cuda.synchronize(device)
                    print("Smoke forward/backward passed on RTX 5090")
                    print(torch.cuda.memory_summary(device=device, abbreviated=True))
                    return
                if args.max_steps is not None and step >= args.max_steps:
                    save_checkpoint(checkpoint, model, optimizer, args, step, epoch, stats)
                    save_snapshot(args, checkpoint, model, optimizer, step, epoch, stats)
                    torch.cuda.synchronize(device)
                    print(f"Reached --max-steps={args.max_steps}; saved checkpoint: {checkpoint}", flush=True)
                    return
                if step % args.save_every == 0:
                    save_checkpoint(checkpoint, model, optimizer, args, step, epoch, stats)
                    save_snapshot(args, checkpoint, model, optimizer, step, epoch, stats)

    final_stats = stats if step else {}
    save_checkpoint(checkpoint, model, optimizer, args, step, args.epochs, final_stats)
    save_snapshot(args, checkpoint, model, optimizer, step, args.epochs, final_stats)
    print(f"Saved checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
