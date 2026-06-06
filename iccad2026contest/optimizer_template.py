#!/usr/bin/env python3
"""
ICCAD 2026 FloorSet Challenge - Optimizer Template

USAGE:
  1. Copy: cp optimizer_template.py my_optimizer.py
  2. Replace the B*-tree code with your algorithm
  3. Test: python iccad2026_evaluate.py --evaluate my_optimizer.py

BASELINE: B*-tree Simulated Annealing
  - GUARANTEES: Overlap-free, area constraints satisfied, fixed/preplaced hard constraints respected
  - NOT HANDLED: MIB, cluster, boundary soft constraints

Your solve() receives:
  - block_count: int
  - area_targets: [n] target area per block
  - b2b_connectivity: [edges, 3] (block_i, block_j, weight)
  - p2b_connectivity: [edges, 3] (pin_idx, block_idx, weight)
  - pins_pos: [n_pins, 2] pin (x, y)
  - constraints: [n, 5] (fixed, preplaced, MIB, cluster, boundary)
  - target_positions: [n, 4] target (x, y, w, h) per block.
      All -1 by default (free). For fixed-shape blocks, w and h are set.
      For preplaced blocks, all four (x, y, w, h) are set.

Your solve() must return:
  - List of (x, y, width, height), exactly block_count tuples
  - Floating-point coordinates allowed
  - Any aspect ratio (w/h) allowed

HARD CONSTRAINTS (violation = Cost 10.0):
  - NO OVERLAPS between blocks
  - AREA: w*h within 1% of area_targets[i] (soft blocks only)
  - DIMENSION IMMUTABILITY: Fixed-shape blocks must use exact (w, h) from
    target_positions; preplaced blocks must use exact (x, y, w, h)

RELAXED CONSTRAINTS:
  - Aspect ratio: Any w/h ratio is valid
  - Fixed outline: Removed (implicitly optimized via p2b HPWL and bbox area)
  - Coordinates: Floating-point allowed
"""

import math
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent))

from iccad2026_evaluate import (
    FloorplanOptimizer,
    calculate_hpwl_b2b,
    calculate_hpwl_p2b,
    calculate_bbox_area,
    check_overlap,
)


# =============================================================================
# B*-TREE DATA STRUCTURE
# Replace this entire class if using a different representation
# (Sequence Pair, O-tree, Corner Block List, etc.)
# =============================================================================

class BStarTree:
    """
    B*-tree for overlap-free floorplanning.
    
    Left child: placed to the RIGHT of parent
    Right child: placed ABOVE parent (same x)
    """
    
    def __init__(self, n_blocks: int, widths: List[float], heights: List[float]):
        self.n = n_blocks
        self.widths = list(widths)
        self.heights = list(heights)
        self.parent = [-1] * n_blocks
        self.left = [-1] * n_blocks
        self.right = [-1] * n_blocks
        self.root = 0
        self._build_random_tree()
    
    def _build_random_tree(self):
        if self.n == 0:
            return
        self.parent = [-1] * self.n
        self.left = [-1] * self.n
        self.right = [-1] * self.n
        
        order = list(range(self.n))
        random.shuffle(order)
        self.root = order[0]
        
        for i in range(1, self.n):
            block = order[i]
            existing = order[random.randint(0, i - 1)]
            if random.random() < 0.5:
                if self.left[existing] == -1:
                    self.left[existing] = block
                    self.parent[block] = existing
                elif self.right[existing] == -1:
                    self.right[existing] = block
                    self.parent[block] = existing
                else:
                    self._insert_at_leaf(block, existing)
            else:
                if self.right[existing] == -1:
                    self.right[existing] = block
                    self.parent[block] = existing
                elif self.left[existing] == -1:
                    self.left[existing] = block
                    self.parent[block] = existing
                else:
                    self._insert_at_leaf(block, existing)
    
    def _insert_at_leaf(self, block: int, start: int):
        current = start
        while True:
            if random.random() < 0.5:
                if self.left[current] == -1:
                    self.left[current] = block
                    self.parent[block] = current
                    return
                current = self.left[current]
            else:
                if self.right[current] == -1:
                    self.right[current] = block
                    self.parent[block] = current
                    return
                current = self.right[current]
    
    def pack(self) -> List[Tuple[float, float, float, float]]:
        """
        Compute (x, y, w, h) from tree structure.
        
        Uses proper contour tracking to ensure overlap-free placement.
        B*-tree rules:
        - Left child: placed to the RIGHT of parent
        - Right child: placed ABOVE parent (same x as parent)
        """
        positions = [(0.0, 0.0, self.widths[i], self.heights[i]) for i in range(self.n)]
        if self.n == 0:
            return positions
        
        # Contour: sorted list of (x_end, y_top) representing skyline
        # At any x, the contour height is the y_top of the rightmost segment with x_end > x
        contour = [(0.0, 0.0)]  # Start with ground level
        
        def get_contour_y(x_start: float, x_end: float) -> float:
            """Find max y in contour for range [x_start, x_end]."""
            max_y = 0.0
            for i, (cx_end, cy_top) in enumerate(contour):
                # Get x_start of this segment
                cx_start = contour[i-1][0] if i > 0 else 0.0
                # Check if segments overlap
                if x_start < cx_end and x_end > cx_start:
                    max_y = max(max_y, cy_top)
            return max_y
        
        def update_contour(x_start: float, x_end: float, y_top: float):
            """Add a new block to the contour."""
            nonlocal contour
            new_contour = []
            
            for i, (cx_end, cy_top) in enumerate(contour):
                cx_start = contour[i-1][0] if i > 0 else 0.0
                
                # Before the new block
                if cx_end <= x_start:
                    new_contour.append((cx_end, cy_top))
                # After the new block
                elif cx_start >= x_end:
                    new_contour.append((cx_end, cy_top))
                # Overlapping - need to split
                else:
                    # Part before new block
                    if cx_start < x_start:
                        new_contour.append((x_start, cy_top))
                    # Part after new block
                    if cx_end > x_end:
                        new_contour.append((cx_end, cy_top))
            
            # Add the new block segment
            # Find where to insert
            insert_pos = 0
            for i, (cx_end, _) in enumerate(new_contour):
                if cx_end <= x_start:
                    insert_pos = i + 1
            new_contour.insert(insert_pos, (x_end, y_top))
            
            # Sort by x_end and merge adjacent segments with same y
            new_contour.sort(key=lambda x: x[0])
            
            # Merge adjacent segments with same height
            merged = []
            for x_end, y_top in new_contour:
                if merged and merged[-1][1] == y_top:
                    merged[-1] = (x_end, y_top)  # Extend previous
                else:
                    merged.append((x_end, y_top))
            
            contour = merged if merged else [(x_end, 0.0)]
        
        # DFS traversal to place blocks
        def dfs(node: int, parent_right_edge: float):
            if node == -1:
                return
            
            w, h = self.widths[node], self.heights[node]
            
            if node == self.root:
                x = 0.0
                y = 0.0
            else:
                x = parent_right_edge
                y = get_contour_y(x, x + w)
            
            positions[node] = (x, y, w, h)
            update_contour(x, x + w, y + h)
            
            # Left child: to the RIGHT of this node
            dfs(self.left[node], x + w)
            # Right child: ABOVE this node (same x, will stack due to contour)
            dfs(self.right[node], x)
        
        dfs(self.root, 0.0)
        
        # Verify no overlaps (should never happen with correct contour)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                x1, y1, w1, h1 = positions[i]
                x2, y2, w2, h2 = positions[j]
                overlap_x = min(x1 + w1, x2 + w2) - max(x1, x2)
                overlap_y = min(y1 + h1, y2 + h2) - max(y1, y2)
                if overlap_x > 1e-6 and overlap_y > 1e-6:
                    # Fix by pushing j up
                    positions[j] = (x2, max(y1 + h1, y2), w2, h2)
        
        return positions
    
    def copy(self) -> 'BStarTree':
        new = BStarTree.__new__(BStarTree)
        new.n = self.n
        new.widths = self.widths.copy()
        new.heights = self.heights.copy()
        new.parent = self.parent.copy()
        new.left = self.left.copy()
        new.right = self.right.copy()
        new.root = self.root
        return new
    
    # SA moves
    def move_rotate(self, block: int):
        """Swap width/height (90° rotation, preserves area)."""
        self.widths[block], self.heights[block] = self.heights[block], self.widths[block]
    
    def move_swap(self, b1: int, b2: int):
        """Swap two blocks' dimensions."""
        self.widths[b1], self.widths[b2] = self.widths[b2], self.widths[b1]
        self.heights[b1], self.heights[b2] = self.heights[b2], self.heights[b1]
    
    def move_delete_insert(self, block: int):
        """Delete and reinsert block at random position."""
        if self.n <= 1:
            return
        w, h = self.widths[block], self.heights[block]
        self._delete_node(block)
        target = random.randint(0, self.n - 1)
        while target == block:
            target = random.randint(0, self.n - 1)
        self._insert_node(block, target, random.choice([True, False]))
        self.widths[block], self.heights[block] = w, h
    
    def _delete_node(self, node: int):
        parent = self.parent[node]
        left_child = self.left[node]
        right_child = self.right[node]
        
        if left_child == -1 and right_child == -1:
            replacement = -1
        elif left_child == -1:
            replacement = right_child
        elif right_child == -1:
            replacement = left_child
        else:
            replacement = left_child
            rightmost = left_child
            while self.right[rightmost] != -1:
                rightmost = self.right[rightmost]
            self.right[rightmost] = right_child
            self.parent[right_child] = rightmost
        
        if parent == -1:
            self.root = replacement
        elif self.left[parent] == node:
            self.left[parent] = replacement
        else:
            self.right[parent] = replacement
        
        if replacement != -1:
            self.parent[replacement] = parent
        
        self.parent[node] = -1
        self.left[node] = -1
        self.right[node] = -1
    
    def _insert_node(self, node: int, target: int, as_left: bool):
        if as_left:
            old_child = self.left[target]
            self.left[target] = node
        else:
            old_child = self.right[target]
            self.right[target] = node
        self.parent[node] = target
        if old_child != -1:
            self.left[node] = old_child
            self.parent[old_child] = node


# =============================================================================
# OPTIMIZER CLASS - Replace this with your algorithm
# =============================================================================

class MyOptimizer(FloorplanOptimizer):
    """
    B*-tree Simulated Annealing baseline.
    
    REPLACE THIS CLASS WITH YOUR ALGORITHM.
    Keep the solve() signature the same.
    """
    
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.initial_temp = 100.0
        self.final_temp = 1.0
        self.cooling_rate = 0.9
        self.moves_per_temp = 20
    
    def solve(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: torch.Tensor = None
    ) -> List[Tuple[float, float, float, float]]:
        """
        B*-tree SA optimization.
        
        REPLACE THIS METHOD with your algorithm.
        Must return List[(x, y, w, h)] with exactly block_count entries.
        """
        if block_count == 0:
            return []
        if target_positions is None:
            target_positions = torch.full((block_count, 4), -1.0)

        fixed_dim, fixed_xy = self._hard_constraint_flags(block_count, constraints, target_positions)
        rotatable_blocks = [i for i in range(block_count) if not fixed_dim[i]]
        relocatable_blocks = [i for i in range(block_count) if not fixed_xy[i]]

        # Initialize dimensions: use exact target dimensions for fixed/preplaced
        # blocks, otherwise start with a square matching the area target.
        widths, heights = [], []
        for i in range(block_count):
            w, h = self._block_dims(i, area_targets, target_positions, fixed_dim)
            widths.append(w)
            heights.append(h)

        # Build B*-tree. Raw B*-tree packing is cheap and overlap-free for
        # annealing. The selected draft is hard-legalized before returning,
        # because preplaced blocks have absolute coordinates that ordinary
        # B*-tree packing cannot represent directly.
        tree = BStarTree(block_count, widths, heights)
        current_draft = tree.pack()
        current_cost = self._cost(current_draft, b2b_connectivity, p2b_connectivity, pins_pos)

        best_draft = current_draft
        best_cost = current_cost

        # Simulated Annealing. Keep the template baseline responsive on
        # larger public cases; the goal here is a legal baseline, not an
        # exhaustive B*-tree search.
        moves_per_temp = max(3, min(self.moves_per_temp, 360 // max(block_count, 1)))
        temp = self.initial_temp
        while temp > self.final_temp:
            for _ in range(moves_per_temp):
                old_tree = tree.copy()

                # Rotate only soft blocks; fixed-shape and preplaced dimensions
                # are immutable hard constraints. Reinsert only blocks whose
                # absolute location is not fixed.
                if rotatable_blocks and (not relocatable_blocks or random.random() < 0.5):
                    tree.move_rotate(random.choice(rotatable_blocks))
                elif relocatable_blocks:
                    tree.move_delete_insert(random.choice(relocatable_blocks))
                else:
                    continue

                new_draft = tree.pack()
                new_cost = self._cost(new_draft, b2b_connectivity, p2b_connectivity, pins_pos)

                # Accept/reject
                delta = new_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / temp):
                    current_draft = new_draft
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_draft = new_draft
                else:
                    tree = old_tree

            temp *= self.cooling_rate

        repair_strategy = os.environ.get("BSTAR_REPAIR_STRATEGY", "best").strip().lower()
        best_positions = self._repair_bstar_draft(
            best_draft, block_count, area_targets, constraints, target_positions,
            b2b_connectivity, p2b_connectivity, pins_pos, repair_strategy)
        if not self._hard_feasible(best_positions, area_targets, constraints, target_positions):
            best_positions = self._shelf_fallback(block_count, area_targets, constraints, target_positions)
        return best_positions

    def _tensor_value(self, tensor, row: int, col: int = None, default: float = -1.0) -> float:
        try:
            if tensor is None:
                return default
            if col is None:
                if row >= len(tensor):
                    return default
                return float(tensor[row])
            if row >= tensor.shape[0] or col >= tensor.shape[1]:
                return default
            return float(tensor[row, col])
        except Exception:
            return default

    def _hard_constraint_flags(self, block_count, constraints, target_positions):
        ncols = constraints.shape[1] if constraints is not None and constraints.dim() > 1 else 0
        fixed_dim = [False] * block_count
        fixed_xy = [False] * block_count
        for i in range(block_count):
            is_fixed = ncols > 0 and self._tensor_value(constraints, i, 0, 0.0) != 0
            is_preplaced = ncols > 1 and self._tensor_value(constraints, i, 1, 0.0) != 0
            tw = self._tensor_value(target_positions, i, 2)
            th = self._tensor_value(target_positions, i, 3)
            tx = self._tensor_value(target_positions, i, 0)
            ty = self._tensor_value(target_positions, i, 1)
            has_dim = math.isfinite(tw) and math.isfinite(th) and tw > 0 and th > 0
            fixed_dim[i] = (is_fixed or is_preplaced) and has_dim
            fixed_xy[i] = is_preplaced and has_dim and math.isfinite(tx) and math.isfinite(ty) and tx != -1 and ty != -1
        return fixed_dim, fixed_xy

    def _block_dims(self, block, area_targets, target_positions, fixed_dim, source=None):
        if fixed_dim[block]:
            return (
                self._tensor_value(target_positions, block, 2, default=1.0),
                self._tensor_value(target_positions, block, 3, default=1.0),
            )
        area = self._tensor_value(area_targets, block, default=1.0)
        if not math.isfinite(area) or area <= 0:
            area = 1.0
        if source is not None:
            try:
                w, h = float(source[2]), float(source[3])
                if math.isfinite(w) and math.isfinite(h) and w > 0 and h > 0:
                    scale = math.sqrt(area / max(w * h, 1e-12))
                    return w * scale, h * scale
            except Exception:
                pass
        side = math.sqrt(area)
        return side, side

    def _rects_overlap(self, a, b, eps=1e-7):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        return (min(ax + aw, bx + bw) - max(ax, bx) > eps and
                min(ay + ah, by + bh) - max(ay, by) > eps)

    def _overlaps_any(self, rect, positions, placed):
        return any(self._rects_overlap(rect, positions[j]) for j in placed if positions[j] is not None)

    def _normalize_bstar_draft(self, draft, block_count, area_targets, constraints, target_positions):
        fixed_dim, fixed_xy = self._hard_constraint_flags(block_count, constraints, target_positions)
        result = [None] * block_count
        for i in range(block_count):
            source = draft[i] if draft is not None and draft[i] is not None else None
            w, h = self._block_dims(i, area_targets, target_positions, fixed_dim, source=source)
            if fixed_xy[i]:
                result[i] = (
                    self._tensor_value(target_positions, i, 0),
                    self._tensor_value(target_positions, i, 1),
                    w,
                    h,
                )
            elif source is not None:
                result[i] = (float(source[0]), float(source[1]), w, h)
            else:
                result[i] = (0.0, 0.0, w, h)
        return result

    def _overlap_pairs(self, positions):
        pairs = []
        if positions is None:
            return pairs
        for i in range(len(positions)):
            if positions[i] is None:
                continue
            for j in range(i + 1, len(positions)):
                if positions[j] is not None and self._rects_overlap(positions[i], positions[j]):
                    pairs.append((i, j))
        return pairs

    def _spread_repair(self, draft, block_count, area_targets, constraints, target_positions):
        fixed_dim, fixed_xy = self._hard_constraint_flags(block_count, constraints, target_positions)
        base = self._normalize_bstar_draft(draft, block_count, area_targets, constraints, target_positions)
        if self._hard_feasible(base, area_targets, constraints, target_positions):
            return base

        anchors = [base[i] for i in range(block_count) if fixed_xy[i]] or base
        anchor_x = sum(x + w / 2.0 for x, y, w, h in anchors) / max(len(anchors), 1)
        anchor_y = sum(y + h / 2.0 for x, y, w, h in anchors) / max(len(anchors), 1)
        best = base
        for step in range(0, 81):
            factor = 1.0 + 0.04 * step
            candidate = list(base)
            for i in range(block_count):
                if fixed_xy[i]:
                    continue
                x, y, w, h = base[i]
                candidate[i] = (
                    anchor_x + factor * (x - anchor_x),
                    anchor_y + factor * (y - anchor_y),
                    w,
                    h,
                )
            best = candidate
            if self._hard_feasible(candidate, area_targets, constraints, target_positions):
                return candidate
        return best

    def _selective_repair(self, draft, block_count, area_targets, constraints, target_positions):
        fixed_dim, fixed_xy = self._hard_constraint_flags(block_count, constraints, target_positions)
        result = self._normalize_bstar_draft(draft, block_count, area_targets, constraints, target_positions)
        if self._hard_feasible(result, area_targets, constraints, target_positions):
            return result

        move_set = set()
        for _ in range(4):
            pairs = self._overlap_pairs(result)
            if not pairs:
                break
            changed = False
            for i, j in pairs:
                if fixed_xy[i] and fixed_xy[j]:
                    continue
                if fixed_xy[i]:
                    chosen = j
                elif fixed_xy[j]:
                    chosen = i
                else:
                    ai = result[i][2] * result[i][3]
                    aj = result[j][2] * result[j][3]
                    chosen = i if ai <= aj else j
                if not fixed_xy[chosen] and chosen not in move_set:
                    move_set.add(chosen)
                    result[chosen] = None
                    changed = True
            if not changed:
                break

        if not move_set:
            return result

        placed = [i for i in range(block_count) if result[i] is not None]
        def draft_key(i):
            if draft is None or draft[i] is None:
                return (0.0, 0.0, i)
            return (float(draft[i][1]), float(draft[i][0]), i)

        for block in sorted(move_set, key=draft_key):
            source = draft[block] if draft is not None and draft[block] is not None else None
            w, h = self._block_dims(block, area_targets, target_positions, fixed_dim, source=source)
            target_x = float(source[0]) if source is not None else 0.0
            target_y = float(source[1]) if source is not None else 0.0
            best_rect = None
            best_cost = float('inf')
            for x, y in self._candidate_points(block, w, h, draft, result, placed):
                rect = (x, y, w, h)
                if self._overlaps_any(rect, result, placed):
                    continue
                bbox = self._bbox_with_rect(result, placed, rect)
                cost = (x - target_x) ** 2 + (y - target_y) ** 2 + 0.001 * bbox
                if cost < best_cost:
                    best_cost = cost
                    best_rect = rect
            if best_rect is None:
                if placed:
                    max_x = max(result[j][0] + result[j][2] for j in placed if result[j] is not None)
                    min_y = min(result[j][1] for j in placed if result[j] is not None)
                    best_rect = (max_x, min_y, w, h)
                else:
                    best_rect = (0.0, 0.0, w, h)
            result[block] = best_rect
            placed.append(block)
        return result

    def _bbox_with_rect(self, positions, placed, rect):
        x, y, w, h = rect
        if not placed:
            return w * h
        min_x = min([positions[j][0] for j in placed if positions[j] is not None] + [x])
        min_y = min([positions[j][1] for j in placed if positions[j] is not None] + [y])
        max_x = max([positions[j][0] + positions[j][2] for j in placed if positions[j] is not None] + [x + w])
        max_y = max([positions[j][1] + positions[j][3] for j in placed if positions[j] is not None] + [y + h])
        return (max_x - min_x) * (max_y - min_y)

    def _repair_bstar_draft(self, draft, block_count, area_targets, constraints, target_positions,
                            b2b_conn, p2b_conn, pins_pos, strategy):
        candidates = {}
        if strategy in ("repack", "best", ""):
            candidates["repack"] = self._legalize_hard_constraints(draft, block_count, area_targets, constraints, target_positions)
        if strategy in ("spread", "best", ""):
            candidates["spread"] = self._spread_repair(draft, block_count, area_targets, constraints, target_positions)
        if strategy in ("selective", "best", ""):
            candidates["selective"] = self._selective_repair(draft, block_count, area_targets, constraints, target_positions)
        if strategy not in ("repack", "spread", "selective", "best", ""):
            candidates[strategy] = self._legalize_hard_constraints(draft, block_count, area_targets, constraints, target_positions)

        feasible = {
            name: pos for name, pos in candidates.items()
            if self._hard_feasible(pos, area_targets, constraints, target_positions)
        }
        if not feasible:
            return self._shelf_fallback(block_count, area_targets, constraints, target_positions)
        return min(feasible.values(), key=lambda pos: self._cost(pos, b2b_conn, p2b_conn, pins_pos))

    def _candidate_points(self, block, w, h, draft, result, placed):
        points = [(0.0, 0.0)]
        if draft is not None and draft[block] is not None:
            x, y = float(draft[block][0]), float(draft[block][1])
            if math.isfinite(x) and math.isfinite(y):
                points.append((x, y))

        if placed:
            dx = draft[block][0] if draft is not None and draft[block] is not None else 0.0
            dy = draft[block][1] if draft is not None and draft[block] is not None else 0.0
            nearby = sorted(
                [j for j in placed if result[j] is not None],
                key=lambda j: (result[j][0] - dx) ** 2 + (result[j][1] - dy) ** 2,
            )[:50]
            for j in nearby:
                x, y, jw, jh = result[j]
                points.extend([(x + jw, y), (x, y + jh), (x - w, y), (x, y - h), (x + jw, y + jh)])
            max_x = max(result[j][0] + result[j][2] for j in placed if result[j] is not None)
            min_y = min(result[j][1] for j in placed if result[j] is not None)
            points.append((max_x, min_y))

        unique = []
        seen = set()
        for x, y in points:
            if not math.isfinite(x) or not math.isfinite(y):
                continue
            key = (round(x, 6), round(y, 6))
            if key not in seen:
                seen.add(key)
                unique.append((x, y))
        return unique

    def _legalize_hard_constraints(self, draft, block_count, area_targets, constraints, target_positions):
        fixed_dim, fixed_xy = self._hard_constraint_flags(block_count, constraints, target_positions)
        result = [None] * block_count
        placed = []

        for i in range(block_count):
            if fixed_xy[i]:
                w, h = self._block_dims(i, area_targets, target_positions, fixed_dim)
                result[i] = (
                    self._tensor_value(target_positions, i, 0),
                    self._tensor_value(target_positions, i, 1),
                    w,
                    h,
                )
                placed.append(i)

        def draft_key(i):
            if draft is None or draft[i] is None:
                return (0.0, 0.0, i)
            return (float(draft[i][1]), float(draft[i][0]), i)

        order = [i for i in sorted(range(block_count), key=draft_key) if not fixed_xy[i]]
        for block in order:
            w, h = self._block_dims(
                block, area_targets, target_positions, fixed_dim,
                source=(draft[block] if draft is not None and draft[block] is not None else None),
            )
            best_rect = None
            best_cost = float('inf')
            target_x = draft[block][0] if draft is not None and draft[block] is not None else 0.0
            target_y = draft[block][1] if draft is not None and draft[block] is not None else 0.0

            for x, y in self._candidate_points(block, w, h, draft, result, placed):
                if not any(fixed_xy):
                    x = max(0.0, x)
                    y = max(0.0, y)
                rect = (x, y, w, h)
                if self._overlaps_any(rect, result, placed):
                    continue
                if placed:
                    min_x = min([result[j][0] for j in placed if result[j] is not None] + [x])
                    min_y = min([result[j][1] for j in placed if result[j] is not None] + [y])
                    max_x = max([result[j][0] + result[j][2] for j in placed if result[j] is not None] + [x + w])
                    max_y = max([result[j][1] + result[j][3] for j in placed if result[j] is not None] + [y + h])
                    bbox = (max_x - min_x) * (max_y - min_y)
                else:
                    bbox = w * h
                cost = (x - target_x) ** 2 + (y - target_y) ** 2 + 0.001 * bbox
                if cost < best_cost:
                    best_cost = cost
                    best_rect = rect

            if best_rect is None:
                if placed:
                    max_x = max(result[j][0] + result[j][2] for j in placed if result[j] is not None)
                    min_y = min(result[j][1] for j in placed if result[j] is not None)
                    best_rect = (max_x, min_y, w, h)
                else:
                    best_rect = (0.0, 0.0, w, h)

            result[block] = best_rect
            placed.append(block)

        return result

    def _hard_feasible(self, positions, area_targets, constraints, target_positions):
        if positions is None:
            return False
        block_count = len(positions)
        fixed_dim, fixed_xy = self._hard_constraint_flags(block_count, constraints, target_positions)
        for i, pos in enumerate(positions):
            if pos is None or len(pos) != 4:
                return False
            x, y, w, h = pos
            if not all(math.isfinite(v) for v in pos) or w <= 0 or h <= 0:
                return False
            if fixed_dim[i]:
                if (abs(w - self._tensor_value(target_positions, i, 2)) > 1e-4 or
                        abs(h - self._tensor_value(target_positions, i, 3)) > 1e-4):
                    return False
            else:
                area = self._tensor_value(area_targets, i, default=1.0)
                if area > 0 and abs(w * h - area) / max(area, 1e-9) > 0.01:
                    return False
            if fixed_xy[i]:
                if (abs(x - self._tensor_value(target_positions, i, 0)) > 1e-4 or
                        abs(y - self._tensor_value(target_positions, i, 1)) > 1e-4):
                    return False
        return check_overlap(positions) == 0

    def _shelf_fallback(self, block_count, area_targets, constraints, target_positions):
        fixed_dim, fixed_xy = self._hard_constraint_flags(block_count, constraints, target_positions)
        draft = [None] * block_count
        cursor_x = 0.0
        cursor_y = 0.0
        row_h = 0.0
        total_area = sum(max(self._tensor_value(area_targets, i, default=1.0), 1.0) for i in range(block_count))
        row_w = max(math.sqrt(total_area) * 1.4, 1.0)
        for i in range(block_count):
            w, h = self._block_dims(i, area_targets, target_positions, fixed_dim)
            if fixed_xy[i]:
                draft[i] = (self._tensor_value(target_positions, i, 0), self._tensor_value(target_positions, i, 1), w, h)
                continue
            if cursor_x > 0 and cursor_x + w > row_w:
                cursor_x = 0.0
                cursor_y += row_h
                row_h = 0.0
            draft[i] = (cursor_x, cursor_y, w, h)
            cursor_x += w
            row_h = max(row_h, h)
        return self._legalize_hard_constraints(draft, block_count, area_targets, constraints, target_positions)

    def _cost(self, positions, b2b_conn, p2b_conn, pins_pos) -> float:
        """Evaluate solution quality (lower is better)."""
        hpwl_b2b = calculate_hpwl_b2b(positions, b2b_conn)
        hpwl_p2b = calculate_hpwl_p2b(positions, p2b_conn, pins_pos)
        area = calculate_bbox_area(positions)
        return hpwl_b2b + hpwl_p2b + area * 0.01
