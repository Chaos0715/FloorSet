#!/usr/bin/env python3
"""Train a constraint-aware GNN policy prior for FloorSet.

This is a separate experiment entrypoint. It deliberately does not modify the
existing prior trainer or optimizer code.

Example:
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 venv/bin/python train_floorplan_soft_prior.py \
    --num-samples 100000 \
    --sample-mode stratified \
    --epochs 5 \
    --batch-size 100 \
    --hidden 384 \
    --message-passes 6 \
    --time-budget-hours 12 \
    --save-every 100 \
    --snapshot-dir models/soft_prior_h384_mp6_snapshots \
    --checkpoint models/floorset_soft_prior_h384_mp6.pt \
    --log models/floorset_soft_prior_h384_mp6.jsonl
"""

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lite_dataset import FloorplanDatasetLite, floorplan_collate  # noqa: E402


RATIO_CANDIDATES = torch.tensor([1.0, 2.0, 0.5, 1.5, 2.0 / 3.0], dtype=torch.float32)
BOUNDARY_BITS = (1, 2, 4, 8)
FEATURE_DIM = 16


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=".", help="FloorSet root containing floorset_lite/")
    parser.add_argument("--checkpoint", default="models/floorset_soft_prior_h384_mp6.pt")
    parser.add_argument("--log", default="models/floorset_soft_prior_h384_mp6.jsonl")
    parser.add_argument("--snapshot-dir", default="models/soft_prior_h384_mp6_snapshots")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=100000, help="<=0 means all local samples")
    parser.add_argument(
        "--sample-mode",
        choices=["first", "spread", "random", "stratified"],
        default="stratified",
        help="How to choose --num-samples from the dataset",
    )
    parser.add_argument("--stratified-pool-multiplier", type=float, default=2.0)
    parser.add_argument("--stratified-pool-max", type=int, default=250000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--message-passes", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--time-budget-hours", type=float, default=12.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="One training step, then save and exit")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle selected samples in the DataLoader")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-edge-pairs", type=int, default=256, help="Pair-touch samples per graph")
    parser.add_argument("--max-order-pairs", type=int, default=512, help="Order-ranking pairs per graph")
    return parser.parse_args()


def require_rtx5090(device_arg: str) -> torch.device:
    assert torch.cuda.is_available(), "CUDA is not available; refusing CPU training"
    visible = []
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        visible.append((idx, torch.cuda.get_device_name(idx), props.total_memory))

    device = torch.device(device_arg)
    assert device.type == "cuda", f"Device must be CUDA, got {device_arg!r}"
    idx = device.index if device.index is not None else torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    props = torch.cuda.get_device_properties(idx)
    assert "RTX 5090" in name, f"Selected GPU cuda:{idx} is not RTX 5090: {name}; visible={visible}"
    assert props.total_memory > 30 * 1024**3, (
        f"Selected RTX 5090 reports {props.total_memory / 1024**3:.2f} GiB; "
        "expected more than 30 GiB"
    )
    print("GPU check passed")
    print(f"  torch={torch.__version__}, torch_cuda={torch.version.cuda}")
    print(f"  device=cuda:{idx}, name={name}, memory={props.total_memory / 1024**3:.2f} GiB")
    print(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    return device


def valid_rows(tensor: torch.Tensor, cols: int) -> torch.Tensor:
    if tensor is None or tensor.numel() == 0:
        return torch.empty((0, cols), dtype=torch.float32)
    tensor = tensor.float()
    if tensor.dim() == 1:
        tensor = tensor.view(1, -1)
    valid = tensor[:, 0] >= 0
    if tensor.shape[1] < cols:
        pad = torch.full((tensor.shape[0], cols - tensor.shape[1]), -1.0, dtype=tensor.dtype)
        tensor = torch.cat([tensor, pad], dim=1)
    return tensor[valid, :cols]


def boundary_bits_tensor(codes: torch.Tensor) -> torch.Tensor:
    code_long = codes.long().view(-1, 1)
    bits = torch.tensor(BOUNDARY_BITS, dtype=torch.long).view(1, 4)
    return ((code_long & bits) != 0).float()


def rect_edge_touch_pairs(rects: torch.Tensor, pairs: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    if pairs.numel() == 0:
        return torch.empty(0, dtype=torch.float32)
    a = rects[pairs[:, 0]]
    b = rects[pairs[:, 1]]
    ax, ay, aw, ah = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx, by, bw, bh = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    y_overlap = torch.minimum(ay + ah, by + bh) - torch.maximum(ay, by)
    x_overlap = torch.minimum(ax + aw, bx + bw) - torch.maximum(ax, bx)
    vertical_edge = ((ax + aw - bx).abs() <= eps) | ((bx + bw - ax).abs() <= eps)
    horizontal_edge = ((ay + ah - by).abs() <= eps) | ((by + bh - ay).abs() <= eps)
    return ((vertical_edge & (y_overlap > eps)) | (horizontal_edge & (x_overlap > eps))).float()


def group_pairs(ids: torch.Tensor) -> torch.Tensor:
    pairs = []
    for group_id in sorted(int(x) for x in ids.unique().tolist() if int(x) > 0):
        members = torch.where(ids.long() == group_id)[0]
        if members.numel() > 1:
            local = torch.combinations(members, r=2)
            pairs.append(local)
    if not pairs:
        return torch.empty((0, 2), dtype=torch.long)
    return torch.cat(pairs, dim=0)


def unique_pair_rows(pairs: torch.Tensor) -> torch.Tensor:
    if pairs.numel() == 0:
        return pairs.reshape(0, 2).long()
    pairs = pairs.long()
    lo = torch.minimum(pairs[:, 0], pairs[:, 1])
    hi = torch.maximum(pairs[:, 0], pairs[:, 1])
    pairs = torch.stack([lo, hi], dim=1)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if pairs.numel() == 0:
        return pairs.reshape(0, 2)
    return torch.unique(pairs, dim=0)


def sample_pairs(
    n: int,
    b2b_edges: torch.Tensor,
    cluster_ids: torch.Tensor,
    rects: torch.Tensor,
    max_pairs: int,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if n <= 1 or max_pairs <= 0:
        empty_pairs = torch.empty((0, 2), dtype=torch.long)
        empty = torch.empty(0, dtype=torch.float32)
        return empty_pairs, empty, empty.bool()

    all_pairs = torch.combinations(torch.arange(n, dtype=torch.long), r=2)
    touch_labels = rect_edge_touch_pairs(rects, all_pairs)
    positive_pairs = all_pairs[touch_labels > 0.5]
    cluster_pair_candidates = group_pairs(cluster_ids)
    b2b_pair_candidates = unique_pair_rows(b2b_edges[:, :2]) if b2b_edges.numel() else torch.empty((0, 2), dtype=torch.long)

    chosen = []
    for candidate in (cluster_pair_candidates, b2b_pair_candidates, positive_pairs):
        if candidate.numel():
            chosen.append(candidate)

    priority = unique_pair_rows(torch.cat(chosen, dim=0)) if chosen else torch.empty((0, 2), dtype=torch.long)
    priority_count = min(priority.shape[0], max_pairs)
    if priority.shape[0] > priority_count:
        perm = torch.randperm(priority.shape[0], generator=generator)[:priority_count]
        priority = priority[perm]

    remaining = max_pairs - priority.shape[0]
    if remaining > 0:
        priority_keys = set((int(i), int(j)) for i, j in priority.tolist())
        available = [idx for idx, pair in enumerate(all_pairs.tolist()) if (pair[0], pair[1]) not in priority_keys]
        if available:
            take = min(remaining, len(available))
            perm = torch.randperm(len(available), generator=generator)[:take].tolist()
            random_pairs = all_pairs[[available[i] for i in perm]]
            pairs = unique_pair_rows(torch.cat([priority, random_pairs], dim=0))
        else:
            pairs = priority
    else:
        pairs = priority

    if pairs.shape[0] > max_pairs:
        perm = torch.randperm(pairs.shape[0], generator=generator)[:max_pairs]
        pairs = pairs[perm]

    labels = rect_edge_touch_pairs(rects, pairs)
    cluster_mask = torch.zeros(pairs.shape[0], dtype=torch.bool)
    if cluster_ids.numel():
        ids_i = cluster_ids[pairs[:, 0]].long()
        ids_j = cluster_ids[pairs[:, 1]].long()
        cluster_mask = (ids_i > 0) & (ids_i == ids_j)
    return pairs, labels, cluster_mask


def sample_order_pairs(
    ranks: torch.Tensor,
    max_pairs: int,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = ranks.numel()
    if n <= 1 or max_pairs <= 0:
        return torch.empty((0, 2), dtype=torch.long), torch.empty(0, dtype=torch.float32)
    pairs = torch.combinations(torch.arange(n, dtype=torch.long), r=2)
    if pairs.shape[0] > max_pairs:
        perm = torch.randperm(pairs.shape[0], generator=generator)[:max_pairs]
        pairs = pairs[perm]
    rank_i = ranks[pairs[:, 0]]
    rank_j = ranks[pairs[:, 1]]
    signs = torch.where(rank_i < rank_j, 1.0, -1.0)
    return pairs, signs.float()


@dataclass
class GraphBatch:
    features: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    edge_type: torch.Tensor
    center_target: torch.Tensor
    ratio_target: torch.Tensor
    boundary_target: torch.Tensor
    boundary_mask: torch.Tensor
    pair_index: torch.Tensor
    pair_label: torch.Tensor
    pair_cluster_mask: torch.Tensor
    order_pair_index: torch.Tensor
    order_pair_sign: torch.Tensor
    mib_groups: List[torch.Tensor]
    mib_ratio_target: torch.Tensor
    total_blocks: int
    graph_count: int


def build_graph_batch(
    batch,
    device: torch.device,
    max_edge_pairs: int,
    max_order_pairs: int,
    seed: int,
    step: int,
) -> GraphBatch:
    (
        area_batch,
        b2b_batch,
        p2b_batch,
        pins_batch,
        constraints_batch,
        _tree_batch,
        fp_batch,
        _metrics_batch,
    ) = batch

    batch_size = int(area_batch.shape[0])
    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(step) * 1009)

    features = []
    edge_indices = []
    edge_weights = []
    edge_types = []
    center_targets = []
    ratio_targets = []
    boundary_targets = []
    boundary_masks = []
    pair_indices = []
    pair_labels = []
    pair_cluster_masks = []
    order_pair_indices = []
    order_pair_signs = []
    mib_groups = []
    mib_ratio_targets = []

    node_offset = 0
    graph_count = 0
    for sample_idx in range(batch_size):
        area_target = area_batch[sample_idx]
        b2b_conn = b2b_batch[sample_idx]
        p2b_conn = p2b_batch[sample_idx]
        pins_pos = pins_batch[sample_idx]
        constraints = constraints_batch[sample_idx]
        fp_sol = fp_batch[sample_idx]

        n = int((area_target != -1).sum().item())
        if n <= 0:
            continue
        graph_count += 1

        area = area_target[:n].float().clamp_min(1e-6)
        constraints = constraints[:n].float()
        fp_sol = fp_sol[:n].float()
        widths = fp_sol[:, 0].clamp_min(1e-6)
        heights = fp_sol[:, 1].clamp_min(1e-6)
        xs = fp_sol[:, 2]
        ys = fp_sol[:, 3]
        rects = torch.stack([xs, ys, widths, heights], dim=1)

        b2b_valid = valid_rows(b2b_conn, 3)
        if b2b_valid.numel():
            mask = (b2b_valid[:, 0] < n) & (b2b_valid[:, 1] < n) & (b2b_valid[:, 2] > 0)
            b2b_valid = b2b_valid[mask]

        p2b_valid = valid_rows(p2b_conn, 3)
        if p2b_valid.numel():
            mask = (
                (p2b_valid[:, 0] >= 0)
                & (p2b_valid[:, 0] < pins_pos.shape[0])
                & (p2b_valid[:, 1] >= 0)
                & (p2b_valid[:, 1] < n)
                & (p2b_valid[:, 2] > 0)
            )
            p2b_valid = p2b_valid[mask]

        b_degree = torch.zeros(n, dtype=torch.float32)
        if b2b_valid.numel():
            src = b2b_valid[:, 0].long()
            dst = b2b_valid[:, 1].long()
            weights = b2b_valid[:, 2].float()
            b_degree.index_add_(0, src, weights)
            b_degree.index_add_(0, dst, weights)

        p_degree = torch.zeros(n, dtype=torch.float32)
        pin_cx = torch.zeros(n, dtype=torch.float32)
        pin_cy = torch.zeros(n, dtype=torch.float32)
        if p2b_valid.numel():
            pin_idx = p2b_valid[:, 0].long()
            block_idx = p2b_valid[:, 1].long()
            weights = p2b_valid[:, 2].float()
            valid_pin_pos = (pins_pos[pin_idx, 0] != -1) & (pins_pos[pin_idx, 1] != -1)
            pin_idx = pin_idx[valid_pin_pos]
            block_idx = block_idx[valid_pin_pos]
            weights = weights[valid_pin_pos]
            if weights.numel():
                p_degree.index_add_(0, block_idx, weights)
                pin_cx.index_add_(0, block_idx, pins_pos[pin_idx, 0].float() * weights)
                pin_cy.index_add_(0, block_idx, pins_pos[pin_idx, 1].float() * weights)
        has_pin = p_degree > 0
        pin_cx[has_pin] /= p_degree[has_pin]
        pin_cy[has_pin] /= p_degree[has_pin]

        total_area = area.sum().clamp_min(1.0)
        scale = torch.sqrt(total_area).clamp_min(1.0)
        fixed_flag = (constraints[:, 0] != 0).float() if constraints.shape[1] > 0 else torch.zeros(n)
        preplaced_flag = (constraints[:, 1] != 0).float() if constraints.shape[1] > 1 else torch.zeros(n)
        mib_ids = constraints[:, 2].long() if constraints.shape[1] > 2 else torch.zeros(n, dtype=torch.long)
        cluster_ids = constraints[:, 3].long() if constraints.shape[1] > 3 else torch.zeros(n, dtype=torch.long)
        boundary_codes = constraints[:, 4].long() if constraints.shape[1] > 4 else torch.zeros(n, dtype=torch.long)
        mib_flag = (mib_ids > 0).float()
        cluster_flag = (cluster_ids > 0).float()
        boundary_flag = (boundary_codes > 0).float()
        boundary_bits = boundary_bits_tensor(boundary_codes)

        sample_features = torch.cat(
            [
                (area / total_area).view(-1, 1),
                (torch.sqrt(area) / scale).view(-1, 1),
                (b_degree / b_degree.max().clamp_min(1.0)).view(-1, 1),
                (p_degree / p_degree.max().clamp_min(1.0)).view(-1, 1),
                (pin_cx / scale).view(-1, 1),
                (pin_cy / scale).view(-1, 1),
                has_pin.float().view(-1, 1),
                fixed_flag.view(-1, 1),
                preplaced_flag.view(-1, 1),
                mib_flag.view(-1, 1),
                cluster_flag.view(-1, 1),
                boundary_flag.view(-1, 1),
                boundary_bits,
            ],
            dim=1,
        )
        features.append(sample_features)

        if b2b_valid.numel():
            b_edges = b2b_valid[:, :2].long() + node_offset
            b_weights = b2b_valid[:, 2].float()
            b_weights = b_weights / b_weights.max().clamp_min(1e-12)
            edge_indices.append(b_edges)
            edge_weights.append(b_weights)
            edge_types.append(torch.zeros(b_edges.shape[0], dtype=torch.long))

        cluster_edges = group_pairs(cluster_ids)
        if cluster_edges.numel():
            edge_indices.append(cluster_edges + node_offset)
            edge_weights.append(torch.ones(cluster_edges.shape[0], dtype=torch.float32))
            edge_types.append(torch.ones(cluster_edges.shape[0], dtype=torch.long))

        mib_edges = group_pairs(mib_ids)
        if mib_edges.numel():
            edge_indices.append(mib_edges + node_offset)
            edge_weights.append(torch.ones(mib_edges.shape[0], dtype=torch.float32))
            edge_types.append(torch.full((mib_edges.shape[0],), 2, dtype=torch.long))

        cx = (xs + widths / 2.0) / scale
        cy = (ys + heights / 2.0) / scale
        center_targets.append(torch.stack([cx, cy], dim=1))
        ratios = widths / heights
        ratio_targets.append(torch.argmin((ratios[:, None] - RATIO_CANDIDATES[None, :]).abs(), dim=1))

        x_min = xs.min()
        y_min = ys.min()
        x_max = (xs + widths).max()
        y_max = (ys + heights).max()
        gt_boundary = torch.stack(
            [
                (xs - x_min).abs() <= 1e-4,
                (xs + widths - x_max).abs() <= 1e-4,
                (ys + heights - y_max).abs() <= 1e-4,
                (ys - y_min).abs() <= 1e-4,
            ],
            dim=1,
        ).float()
        boundary_targets.append(gt_boundary)
        boundary_masks.append((boundary_codes > 0).float().view(-1, 1).expand(-1, 4))

        pair_index, pair_label, pair_cluster_mask = sample_pairs(
            n=n,
            b2b_edges=b2b_valid[:, :2].long() if b2b_valid.numel() else torch.empty((0, 2), dtype=torch.long),
            cluster_ids=cluster_ids,
            rects=rects,
            max_pairs=max_edge_pairs,
            generator=generator,
        )
        if pair_index.numel():
            pair_indices.append(pair_index + node_offset)
            pair_labels.append(pair_label)
            pair_cluster_masks.append(pair_cluster_mask)

        placement_key = xs + ys
        sorted_idx = torch.argsort(placement_key)
        ranks = torch.empty(n, dtype=torch.long)
        ranks[sorted_idx] = torch.arange(n, dtype=torch.long)
        order_pairs, order_signs = sample_order_pairs(ranks, max_order_pairs, generator)
        if order_pairs.numel():
            order_pair_indices.append(order_pairs + node_offset)
            order_pair_signs.append(order_signs)

        sample_ratio_target = ratio_targets[-1]
        for group_id in sorted(int(x) for x in mib_ids.unique().tolist() if int(x) > 0):
            members = torch.where(mib_ids == group_id)[0]
            if members.numel() <= 1:
                continue
            global_members = members + node_offset
            mib_groups.append(global_members)
            values, counts = torch.unique(sample_ratio_target[members], return_counts=True)
            group_ratio = values[torch.argmax(counts)]
            mib_ratio_targets.extend([(int(global_idx), int(group_ratio)) for global_idx in global_members.tolist()])

        node_offset += n

    def cat_or_empty(items, shape, dtype):
        if not items:
            return torch.empty(shape, dtype=dtype)
        return torch.cat(items, dim=0)

    total_blocks = node_offset
    if mib_ratio_targets:
        mib_ratio_target = torch.full((total_blocks,), -1, dtype=torch.long)
        for node_idx, ratio_idx in mib_ratio_targets:
            mib_ratio_target[node_idx] = ratio_idx
    else:
        mib_ratio_target = torch.full((total_blocks,), -1, dtype=torch.long)

    graph_batch = GraphBatch(
        features=cat_or_empty(features, (0, FEATURE_DIM), torch.float32).to(device),
        edge_index=cat_or_empty(edge_indices, (0, 2), torch.long).to(device),
        edge_weight=cat_or_empty(edge_weights, (0,), torch.float32).to(device),
        edge_type=cat_or_empty(edge_types, (0,), torch.long).to(device),
        center_target=cat_or_empty(center_targets, (0, 2), torch.float32).to(device),
        ratio_target=cat_or_empty(ratio_targets, (0,), torch.long).to(device),
        boundary_target=cat_or_empty(boundary_targets, (0, 4), torch.float32).to(device),
        boundary_mask=cat_or_empty(boundary_masks, (0, 4), torch.float32).to(device),
        pair_index=cat_or_empty(pair_indices, (0, 2), torch.long).to(device),
        pair_label=cat_or_empty(pair_labels, (0,), torch.float32).to(device),
        pair_cluster_mask=cat_or_empty(pair_cluster_masks, (0,), torch.bool).to(device),
        order_pair_index=cat_or_empty(order_pair_indices, (0, 2), torch.long).to(device),
        order_pair_sign=cat_or_empty(order_pair_signs, (0,), torch.float32).to(device),
        mib_groups=[group.to(device) for group in mib_groups],
        mib_ratio_target=mib_ratio_target.to(device),
        total_blocks=total_blocks,
        graph_count=graph_count,
    )
    return graph_batch


class ConstraintAwareGNNPrior(nn.Module):
    def __init__(self, feature_dim: int = FEATURE_DIM, hidden: int = 384, message_passes: int = 6):
        super().__init__()
        self.message_passes = int(message_passes)
        self.node_in = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.edge_type_embed = nn.Embedding(3, hidden)
        self.msg_layers = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(self.message_passes))
        self.self_layers = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(self.message_passes))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(self.message_passes))

        self.center_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2))
        self.ratio_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, len(RATIO_CANDIDATES)))
        self.order_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 1))
        self.boundary_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 4))
        self.mib_ratio_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, len(RATIO_CANDIDATES)))
        self.edge_touch_head = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, graph: GraphBatch) -> Dict[str, torch.Tensor]:
        h = self.node_in(graph.features)
        n = h.shape[0]
        if n == 0:
            raise ValueError("empty graph batch")
        for layer in range(self.message_passes):
            agg = torch.zeros_like(h)
            if graph.edge_index.numel():
                src = graph.edge_index[:, 0]
                dst = graph.edge_index[:, 1]
                weight = graph.edge_weight.to(h.dtype).view(-1, 1)
                edge_bias = self.edge_type_embed(graph.edge_type)
                src_msg = self.msg_layers[layer](h[src] + edge_bias) * weight
                dst_msg = self.msg_layers[layer](h[dst] + edge_bias) * weight
                agg.index_add_(0, dst, src_msg)
                agg.index_add_(0, src, dst_msg)
            update = self.self_layers[layer](h) + agg / math.sqrt(max(n, 1))
            h = self.norms[layer](h + F.silu(update))
        return {
            "embedding": h,
            "center": self.center_head(h),
            "ratio_logits": self.ratio_head(h),
            "order": self.order_head(h).squeeze(-1),
            "boundary_logits": self.boundary_head(h),
            "mib_ratio_logits": self.mib_ratio_head(h),
        }

    def edge_touch_logits(self, embeddings: torch.Tensor, pair_index: torch.Tensor) -> torch.Tensor:
        if pair_index.numel() == 0:
            return embeddings.new_empty(0)
        a = embeddings[pair_index[:, 0]]
        b = embeddings[pair_index[:, 1]]
        pair_features = torch.cat([a + b, (a - b).abs(), a * b], dim=1)
        return self.edge_touch_head(pair_features).squeeze(-1)


def masked_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    active = mask > 0
    if not active.any():
        return logits.new_tensor(0.0)
    return F.binary_cross_entropy_with_logits(logits[active], target[active])


def balanced_bce_with_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.numel() == 0:
        return logits.new_tensor(0.0)
    positives = target.sum()
    negatives = target.numel() - positives
    pos_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 20.0)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)


def compute_losses(model: ConstraintAwareGNNPrior, graph: GraphBatch) -> Tuple[torch.Tensor, Dict[str, float]]:
    pred = model(graph)
    center_loss = F.smooth_l1_loss(pred["center"], graph.center_target)
    ratio_loss = F.cross_entropy(pred["ratio_logits"], graph.ratio_target)

    if graph.order_pair_index.numel():
        order_delta = pred["order"][graph.order_pair_index[:, 0]] - pred["order"][graph.order_pair_index[:, 1]]
        order_loss = F.softplus(-graph.order_pair_sign * order_delta).mean()
    else:
        order_loss = pred["order"].new_tensor(0.0)

    boundary_loss = masked_bce_with_logits(pred["boundary_logits"], graph.boundary_target, graph.boundary_mask)

    edge_logits = model.edge_touch_logits(pred["embedding"], graph.pair_index)
    if graph.pair_cluster_mask.any():
        edge_touch_loss = balanced_bce_with_logits(edge_logits[graph.pair_cluster_mask], graph.pair_label[graph.pair_cluster_mask])
    else:
        edge_touch_loss = balanced_bce_with_logits(edge_logits, graph.pair_label)

    mib_active = graph.mib_ratio_target >= 0
    if mib_active.any():
        mib_ce = F.cross_entropy(pred["mib_ratio_logits"][mib_active], graph.mib_ratio_target[mib_active])
    else:
        mib_ce = pred["mib_ratio_logits"].new_tensor(0.0)

    consistency_terms = []
    mib_probs = F.softmax(pred["mib_ratio_logits"], dim=-1)
    for group in graph.mib_groups:
        if group.numel() <= 1:
            continue
        probs = mib_probs[group]
        group_mean = probs.mean(dim=0, keepdim=True)
        consistency_terms.append(F.mse_loss(probs, group_mean.expand_as(probs)))
    if consistency_terms:
        mib_consistency = torch.stack(consistency_terms).mean()
    else:
        mib_consistency = pred["mib_ratio_logits"].new_tensor(0.0)
    mib_shape_loss = mib_ce + mib_consistency

    loss = (
        1.0 * center_loss
        + 0.5 * ratio_loss
        + 0.4 * order_loss
        + 0.8 * boundary_loss
        + 1.2 * edge_touch_loss
        + 0.8 * mib_shape_loss
    )

    with torch.no_grad():
        stats = {
            "loss": float(loss.detach().cpu()),
            "center_loss": float(center_loss.detach().cpu()),
            "ratio_loss": float(ratio_loss.detach().cpu()),
            "order_loss": float(order_loss.detach().cpu()),
            "boundary_loss": float(boundary_loss.detach().cpu()),
            "edge_touch_loss": float(edge_touch_loss.detach().cpu()),
            "mib_shape_loss": float(mib_shape_loss.detach().cpu()),
            "mib_ce_loss": float(mib_ce.detach().cpu()),
            "mib_consistency_loss": float(mib_consistency.detach().cpu()),
            "graphs": int(graph.graph_count),
            "blocks": int(graph.total_blocks),
            "edge_pairs": int(graph.pair_label.numel()),
            "cluster_edge_pairs": int(graph.pair_cluster_mask.sum().detach().cpu()),
            "order_pairs": int(graph.order_pair_sign.numel()),
            "boundary_masked_bits": int(graph.boundary_mask.sum().detach().cpu()),
            "mib_groups": int(len(graph.mib_groups)),
        }
    return loss, stats


def sample_metadata(sample) -> Tuple[int, int, int, int, int, int]:
    area_target, b2b_conn, p2b_conn, _pins_pos, constraints = sample["input"]
    n = int((area_target != -1).sum().item())
    constraints = constraints[:n]
    mib = int((constraints[:, 2] > 0).sum().item()) if constraints.shape[1] > 2 else 0
    cluster = int((constraints[:, 3] > 0).sum().item()) if constraints.shape[1] > 3 else 0
    boundary = int((constraints[:, 4] > 0).sum().item()) if constraints.shape[1] > 4 else 0
    p2b_valid = valid_rows(p2b_conn, 3)
    b2b_valid = valid_rows(b2b_conn, 3)
    high_pin = int(p2b_valid.shape[0] >= max(12, n))
    block_bin = 0 if n <= 40 else 1 if n <= 70 else 2 if n <= 95 else 3
    soft_density = (mib + cluster + boundary) / max(n, 1)
    soft_bin = 0 if soft_density == 0 else 1 if soft_density < 0.25 else 2 if soft_density < 0.5 else 3
    return (
        block_bin,
        soft_bin,
        int(mib >= max(2, n // 8)),
        int(cluster >= max(2, n // 8)),
        int(boundary >= max(2, n // 10)),
        high_pin or int(b2b_valid.shape[0] >= max(20, n * 2)),
    )


def spread_indices(total: int, count: int) -> List[int]:
    if count <= 0:
        return []
    if count >= total:
        return list(range(total))
    if count == 1:
        return [0]
    step = (total - 1) / float(count - 1)
    return sorted({min(total - 1, int(round(i * step))) for i in range(count)})


def choose_indices(dataset, count: int, mode: str, seed: int, pool_multiplier: float, pool_max: int) -> List[int]:
    total = len(dataset)
    count = min(max(int(count), 0), total)
    if count >= total:
        return list(range(total))
    if mode == "first":
        return list(range(count))
    if mode == "spread":
        return spread_indices(total, count)

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    if mode == "random":
        return sorted(torch.randperm(total, generator=generator)[:count].tolist())

    pool_count = min(total, max(count, min(int(math.ceil(count * pool_multiplier)), int(pool_max))))
    pool = torch.randperm(total, generator=generator)[:pool_count].tolist()
    pool.sort()

    strata: Dict[Tuple[int, int, int, int, int, int], List[int]] = {}
    for idx in pool:
        key = sample_metadata(dataset[idx])
        strata.setdefault(key, []).append(idx)

    keys = list(strata)
    random.Random(seed).shuffle(keys)
    selected = []
    key_cursor = 0
    while len(selected) < count and keys:
        key = keys[key_cursor % len(keys)]
        bucket = strata[key]
        if bucket:
            selected.append(bucket.pop())
        else:
            keys.remove(key)
            key_cursor -= 1
        key_cursor += 1

    if len(selected) < count:
        seen = set(selected)
        for idx in pool:
            if idx not in seen:
                selected.append(idx)
                seen.add(idx)
                if len(selected) >= count:
                    break
    return sorted(selected[:count])


def save_checkpoint(path: Path, model, optimizer, args, step: int, epoch: int, stats: Dict[str, float]):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "args": vars(args),
            "ratio_candidates": RATIO_CANDIDATES.tolist(),
            "feature_dim": FEATURE_DIM,
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
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / f"{checkpoint.stem}_step{step:05d}.pt"
    save_checkpoint(snapshot_path, model, optimizer, args, step, epoch, stats)


def append_log(log_file, event: Dict):
    log_file.write(json.dumps(event, separators=(",", ":")) + "\n")
    log_file.flush()


def main():
    args = parse_args()
    if args.smoke:
        args.num_samples = min(args.num_samples, max(args.batch_size, 8))
        args.time_budget_hours = None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = require_rtx5090(args.device)

    dataset = FloorplanDatasetLite(args.data_path)
    dataset.all_files = sorted(dataset.all_files)
    full_sample_count = len(dataset)
    if args.num_samples is not None and args.num_samples > 0:
        start_select = time.time()
        selected_indices = choose_indices(
            dataset,
            args.num_samples,
            args.sample_mode,
            args.seed,
            args.stratified_pool_multiplier,
            args.stratified_pool_max,
        )
        dataset = Subset(dataset, selected_indices)
        print(f"Selected {len(selected_indices)} stratified samples in {time.time() - start_select:.1f}s")
    else:
        selected_indices = None

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle and not args.smoke,
        collate_fn=floorplan_collate,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = ConstraintAwareGNNPrior(feature_dim=FEATURE_DIM, hidden=args.hidden, message_passes=args.message_passes).to(device)
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
            except Exception as exc:
                print(f"Warning: could not restore optimizer state: {exc}")
        resume_step = int(saved.get("step", 0) or 0)
        resume_epoch = int(saved.get("epoch", 0) or 0)
        print(f"Resumed checkpoint {checkpoint} at step={resume_step}, epoch={resume_epoch}")

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    gpu_idx = device.index if device.index is not None else torch.cuda.current_device()
    gpu_props = torch.cuda.get_device_properties(gpu_idx)

    print(f"Training samples: {len(dataset)} / {full_sample_count}")
    print(f"Sample mode: {args.sample_mode}")
    print(f"Batch size: {args.batch_size}")
    print(f"Hidden/message passes: {args.hidden}/{args.message_passes}")
    print(f"Checkpoint: {checkpoint}")

    step = resume_step
    start = time.time()
    stats: Dict[str, float] = {}
    with log_path.open("a") as log_file:
        append_log(
            log_file,
            {
                "event": "startup",
                "torch_version": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
                "device": f"cuda:{gpu_idx}",
                "gpu_name": torch.cuda.get_device_name(gpu_idx),
                "gpu_memory_gib": gpu_props.total_memory / (1024**3),
                "checkpoint": str(checkpoint),
                "num_samples": len(dataset),
                "full_sample_count": full_sample_count,
                "sample_mode": args.sample_mode,
                "seed": args.seed,
                "batch_size": args.batch_size,
                "hidden": args.hidden,
                "message_passes": args.message_passes,
                "time_budget_hours": args.time_budget_hours,
                "max_edge_pairs": args.max_edge_pairs,
                "max_order_pairs": args.max_order_pairs,
                "shuffle": args.shuffle,
                "resume": args.resume,
                "resume_step": resume_step,
            },
        )
        for epoch in range(resume_epoch, args.epochs):
            for batch in loader:
                graph = build_graph_batch(batch, device, args.max_edge_pairs, args.max_order_pairs, args.seed, step)
                optimizer.zero_grad(set_to_none=True)
                loss, stats = compute_losses(model, graph)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                step += 1
                stats.update({"step": step, "epoch": epoch, "elapsed": time.time() - start})
                append_log(log_file, stats)

                if step == 1 or step % 10 == 0:
                    print(
                        f"step={step} epoch={epoch} loss={stats['loss']:.4f} "
                        f"center={stats['center_loss']:.4f} ratio={stats['ratio_loss']:.4f} "
                        f"edge={stats['edge_touch_loss']:.4f} graphs={stats['graphs']} blocks={stats['blocks']}",
                        flush=True,
                    )

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

                if args.time_budget_hours is not None and (time.time() - start) >= args.time_budget_hours * 3600.0:
                    save_checkpoint(checkpoint, model, optimizer, args, step, epoch, stats)
                    save_snapshot(args, checkpoint, model, optimizer, step, epoch, stats)
                    torch.cuda.synchronize(device)
                    print(f"Reached --time-budget-hours={args.time_budget_hours}; saved checkpoint: {checkpoint}", flush=True)
                    return

                if args.save_every > 0 and step % args.save_every == 0:
                    save_checkpoint(checkpoint, model, optimizer, args, step, epoch, stats)
                    save_snapshot(args, checkpoint, model, optimizer, step, epoch, stats)

    save_checkpoint(checkpoint, model, optimizer, args, step, args.epochs, stats)
    save_snapshot(args, checkpoint, model, optimizer, step, args.epochs, stats)
    print(f"Saved checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
