import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from battleship.domain.board import cell_index
from battleship.domain.config import BOARD_SIZE, EMPTY, HIT, MISS
from battleship.domain.worlds import compute_min_expected_worlds_after_one_shot, sample_worlds


@dataclass
class Posterior:
    worlds_union: List[int]
    cell_hit_counts: List[int]
    N: int


_NEIGHBORS4_CACHE: Dict[int, List[List[Tuple[int, int]]]] = {}


def _neighbors4(board_size: int) -> List[List[Tuple[int, int]]]:
    cached = _NEIGHBORS4_CACHE.get(board_size)
    if cached is not None:
        return cached
    total_cells = board_size * board_size
    neighbors: List[List[Tuple[int, int]]] = [[] for _ in range(total_cells)]
    for r in range(board_size):
        for c in range(board_size):
            idx = r * board_size + c
            if r > 0:
                neighbors[idx].append((r - 1, c))
            if r + 1 < board_size:
                neighbors[idx].append((r + 1, c))
            if c > 0:
                neighbors[idx].append((r, c - 1))
            if c + 1 < board_size:
                neighbors[idx].append((r, c + 1))
    _NEIGHBORS4_CACHE[board_size] = neighbors
    return neighbors


def _param_float(params: Dict[str, float], key: str, env_name: str, default: float) -> float:
    if key in params:
        try:
            return float(params[key])
        except (TypeError, ValueError):
            return default
    raw = os.getenv(env_name)
    if raw is None:
        return default
    try:
        return float(str(raw).strip())
    except ValueError:
        return default


def _param_int(params: Dict[str, float], key: str, env_name: str, default: int) -> int:
    if key in params:
        try:
            value = int(params[key])
            return value if value > 0 else default
        except (TypeError, ValueError):
            return default
    raw = os.getenv(env_name)
    if raw is None:
        return default
    try:
        value = int(str(raw).strip())
        return value if value > 0 else default
    except ValueError:
        return default


def _mask_to_cells(mask: int, board_size: int) -> List[Tuple[int, int]]:
    cells: List[Tuple[int, int]] = []
    m = mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        cells.append(divmod(idx, board_size))
        m ^= lsb
    return cells


def _mask_from_cells(cells: Sequence[Tuple[int, int]], board_size: int) -> int:
    mask = 0
    for r, c in cells:
        if 0 <= r < board_size and 0 <= c < board_size:
            mask |= 1 << cell_index(r, c, board_size)
    return mask


def _placement_len(p: object) -> int:
    cells = getattr(p, "cells", None)
    if cells:
        return len(cells)
    mask = getattr(p, "mask", 0)
    if isinstance(mask, int):
        return mask.bit_count()
    return 0


def _ship_lengths_from_placements(
    placements: Dict[str, List[object]],
    ship_ids: Sequence[str],
) -> Dict[str, int]:
    lengths: Dict[str, int] = {}
    for ship in ship_ids:
        max_len = 0
        for p in placements.get(ship, []):
            plen = _placement_len(p)
            if plen > max_len:
                max_len = plen
        lengths[ship] = max_len
    return lengths


def two_ply_selection(
    board: List[List[str]],
    world_masks: List[int],
    cell_hit_counts: List[int],
    board_size: int = BOARD_SIZE,
) -> Tuple[List[float], List[int], float]:
    """
    Shared 2-ply heuristic used by the Attack tab and the model simulator.

    Returns:
        info_gain_values: list[float] per cell, normalized 0..1 (same as Attack overlay).
        best_indices: list[int] of cell indices (0..board_size*board_size-1) tied for best 2-ply score.
        best_p_hit: hit probability of those best cells.
    """
    N = len(world_masks)
    total_cells = board_size * board_size

    if N == 0:
        return [0.0] * total_cells, [], 0.0

    # Build known mask from current board
    known_mask = 0
    for r in range(board_size):
        for c in range(board_size):
            if board[r][c] != EMPTY:
                known_mask |= 1 << cell_index(r, c, board_size)

    best_score1: Optional[float] = None
    scores1: List[Optional[float]] = [None] * total_cells
    info_values: List[float] = [0.0] * total_cells
    candidates: List[int] = []

    # 1-ply split score (same as in Attack tab): smaller is better
    for idx in range(total_cells):
        if (known_mask >> idx) & 1:
            continue
        n_hit = cell_hit_counts[idx]
        n_miss = N - n_hit
        score = (n_hit * n_hit + n_miss * n_miss) / N
        scores1[idx] = score
        candidates.append(idx)
        if best_score1 is None or score < best_score1:
            best_score1 = score

    if best_score1 is None:
        return [0.0] * total_cells, [], 0.0

    # Info gain normalization (same overlay as Attack tab)
    max_gain = 0.0
    for idx in candidates:
        gain = N - scores1[idx]
        info_values[idx] = gain
        if gain > max_gain:
            max_gain = gain
    if max_gain > 0:
        for idx in candidates:
            info_values[idx] /= max_gain

    # 2-ply expected log-worlds criterion
    candidates.sort(key=lambda i: scores1[i])  # best 1-ply splits first
    TOP_K = min(24, len(candidates))
    top_candidates = candidates[:TOP_K]

    best_two_ply: Optional[float] = None
    best_indices: List[int] = []
    best_p = 0.0

    for idx in top_candidates:
        bit = 1 << idx
        worlds_hit = [wm for wm in world_masks if wm & bit]
        worlds_miss = [wm for wm in world_masks if not (wm & bit)]
        Nh = len(worlds_hit)
        Nm = len(worlds_miss)
        if Nh + Nm == 0:
            continue
        p_hit = Nh / (Nh + Nm)

        known_after = known_mask | bit
        Eh = compute_min_expected_worlds_after_one_shot(worlds_hit, known_after, board_size) if Nh > 0 else 0.0
        Em = compute_min_expected_worlds_after_one_shot(worlds_miss, known_after, board_size) if Nm > 0 else 0.0
        two_ply = p_hit * Eh + (1.0 - p_hit) * Em

        if best_two_ply is None or two_ply < best_two_ply - 1e-9:
            best_two_ply = two_ply
            best_indices = [idx]
            best_p = p_hit
        elif abs(two_ply - best_two_ply) <= 1e-9:
            if p_hit > best_p + 1e-9:
                best_indices = [idx]
                best_p = p_hit
            elif abs(p_hit - best_p) <= 1e-9:
                best_indices.append(idx)

    return info_values, best_indices, best_p


def _choose_next_shot_for_strategy(
    strategy: str,
    board: List[List[str]],
    placements: Dict[str, List[object]],
    rng: random.Random,
    ship_ids: Sequence[str],
    board_size: int = BOARD_SIZE,
    known_sunk: Optional[Set[str]] = None,
    known_assigned: Optional[Dict[str, Set[Tuple[int, int]]]] = None,
    params: Optional[Dict[str, float]] = None,
    posterior: Optional[Posterior] = None,
    profiler: Optional[object] = None,
    unknown_cells: Optional[Sequence[Tuple[int, int]]] = None,
    has_any_hit: Optional[bool] = None,
    hit_mask: Optional[int] = None,
    miss_mask: Optional[int] = None,
    opponent_prior: Optional[List[float]] = None,
) -> Tuple[int, int]:
    """
    Choose the next shot for the given strategy using the current board state.

    Notes:
    - `known_sunk` and `known_assigned` are carried across turns by the simulator.
    - `params` is optional and is used for tunable strategies (e.g., softmax temperature).
    """
    if params is None:
        params = {}

    total_cells = board_size * board_size
    hit_cells: Optional[List[Tuple[int, int]]] = None
    if unknown_cells is None or hit_mask is None or miss_mask is None or has_any_hit is None:
        calc_unknown: List[Tuple[int, int]] = []
        calc_hit_mask = 0
        calc_miss_mask = 0
        calc_hit_cells: List[Tuple[int, int]] = []
        for r in range(board_size):
            for c in range(board_size):
                state = board[r][c]
                idx = cell_index(r, c, board_size)
                if state == EMPTY:
                    calc_unknown.append((r, c))
                elif state == HIT:
                    calc_hit_mask |= 1 << idx
                    calc_hit_cells.append((r, c))
                else:
                    calc_miss_mask |= 1 << idx
        unknown_cells = calc_unknown
        if hit_mask is None:
            hit_mask = calc_hit_mask
        if miss_mask is None:
            miss_mask = calc_miss_mask
        if has_any_hit is None:
            has_any_hit = bool(calc_hit_mask)
        hit_cells = calc_hit_cells
    else:
        unknown_cells = list(unknown_cells)
        if has_any_hit is None:
            has_any_hit = bool(hit_mask)

    if not unknown_cells:
        return 0, 0

    if strategy == "random":
        return rng.choice(unknown_cells)

    if has_any_hit is None:
        has_any_hit = bool(hit_mask)

    confirmed_sunk = known_sunk if known_sunk is not None else set()
    assigned_hits = known_assigned if known_assigned is not None else {s: set() for s in ship_ids}
    hit_mask = hit_mask or 0
    miss_mask = miss_mask or 0
    unknown_mask = ((1 << total_cells) - 1) & ~(hit_mask | miss_mask)
    if hit_cells is None and hit_mask:
        hit_cells = _mask_to_cells(hit_mask, board_size)
    neighbors = _neighbors4(board_size)

    def _adjacent_hits(cell: Tuple[int, int], hit_set: Set[Tuple[int, int]]) -> int:
        idx = cell_index(cell[0], cell[1], board_size)
        count = 0
        for nr, nc in neighbors[idx]:
            if (nr, nc) in hit_set:
                count += 1
        return count

    def _choose_assigned_target_marginal() -> Optional[Tuple[int, int]]:
        active_ships = [s for s in ship_ids if s not in confirmed_sunk]
        if not active_ships:
            return None
        live_hits = set(hit_cells or [])
        if confirmed_sunk and assigned_hits:
            for ship in confirmed_sunk:
                for cell in assigned_hits.get(ship, set()):
                    live_hits.discard(cell)
        any_assigned = any(assigned_hits.get(s) for s in active_ships)
        if not live_hits and not any_assigned:
            return None

        best_cell = None
        best_prob = -1.0
        best_tiebreak = -1

        for ship in active_ships:
            hits = assigned_hits.get(ship, set())
            if not hits:
                continue
            req_mask = _mask_from_cells(hits, board_size)
            counts = [0] * total_cells
            total = 0
            for p in placements.get(ship, []):
                pmask = getattr(p, "mask", 0)
                if pmask & miss_mask:
                    continue
                if req_mask and (req_mask & ~pmask) != 0:
                    continue
                total += 1
                m = pmask & unknown_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    counts[idx] += 1
                    m ^= lsb
            if total == 0:
                return None

            max_count = max(counts) if counts else 0
            if max_count <= 0:
                continue

            candidates_idx = [i for i, v in enumerate(counts) if v == max_count]
            hits_list = list(hits)
            endpoint_idxs: Set[int] = set()
            if len(hits_list) >= 2:
                rows = {r for r, _ in hits_list}
                cols = {c for _, c in hits_list}
                if len(rows) == 1:
                    r0 = next(iter(rows))
                    min_c = min(cols)
                    max_c = max(cols)
                    for cand_c in (min_c - 1, max_c + 1):
                        if 0 <= cand_c < board_size and board[r0][cand_c] == EMPTY:
                            endpoint_idxs.add(cell_index(r0, cand_c, board_size))
                elif len(cols) == 1:
                    c0 = next(iter(cols))
                    min_r = min(rows)
                    max_r = max(rows)
                    for cand_r in (min_r - 1, max_r + 1):
                        if 0 <= cand_r < board_size and board[cand_r][c0] == EMPTY:
                            endpoint_idxs.add(cell_index(cand_r, c0, board_size))

            if endpoint_idxs:
                endpoint_candidates = [idx for idx in candidates_idx if idx in endpoint_idxs]
            else:
                endpoint_candidates = []
            if endpoint_candidates:
                chosen_idx = endpoint_candidates[0]
            else:
                hit_set = set(hits_list) if hits_list else live_hits
                best_adj = -1
                chosen_idx = candidates_idx[0]
                for idx in candidates_idx:
                    cell = divmod(idx, board_size)
                    adj = _adjacent_hits(cell, hit_set)
                    if adj > best_adj:
                        best_adj = adj
                        chosen_idx = idx
                best_adj = max(best_adj, 0)

            prob = max_count / max(1, total)
            tiebreak = max_count
            if prob > best_prob or (abs(prob - best_prob) <= 1e-9 and tiebreak > best_tiebreak):
                best_prob = prob
                best_tiebreak = tiebreak
                best_cell = divmod(chosen_idx, board_size)

        return best_cell

    def _choose_placement_factorized() -> Optional[Tuple[int, int]]:
        active_ships = [s for s in ship_ids if s not in confirmed_sunk]
        if not active_ships:
            return None
        counts = [0] * total_cells
        for ship in active_ships:
            req_mask = _mask_from_cells(assigned_hits.get(ship, set()), board_size)
            for p in placements.get(ship, []):
                pmask = getattr(p, "mask", 0)
                if pmask & miss_mask:
                    continue
                if req_mask and (req_mask & ~pmask) != 0:
                    continue
                m = pmask & unknown_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    counts[idx] += 1
                    m ^= lsb

        best_cell = None
        best_score = -1
        best_endpoint = -1
        best_adj = -1
        best_center = float("inf")
        hit_cells_local = set(hit_cells or [])
        endpoint_idxs: Set[int] = set()
        if hit_cells_local:
            visited: Set[Tuple[int, int]] = set()
            for start in hit_cells_local:
                if start in visited:
                    continue
                stack = [start]
                visited.add(start)
                comp: List[Tuple[int, int]] = []
                while stack:
                    cr, cc = stack.pop()
                    comp.append((cr, cc))
                    idx = cell_index(cr, cc, board_size)
                    for nr, nc in neighbors[idx]:
                        if (nr, nc) in hit_cells_local and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            stack.append((nr, nc))
                if len(comp) < 2:
                    continue
                rows = {r for r, _ in comp}
                cols = {c for _, c in comp}
                if len(rows) == 1:
                    r0 = next(iter(rows))
                    cols_list = [c for _, c in comp]
                    for cand_c in (min(cols_list) - 1, max(cols_list) + 1):
                        if 0 <= cand_c < board_size and board[r0][cand_c] == EMPTY:
                            endpoint_idxs.add(cell_index(r0, cand_c, board_size))
                elif len(cols) == 1:
                    c0 = next(iter(cols))
                    rows_list = [r for r, _ in comp]
                    for cand_r in (min(rows_list) - 1, max(rows_list) + 1):
                        if 0 <= cand_r < board_size and board[cand_r][c0] == EMPTY:
                            endpoint_idxs.add(cell_index(cand_r, c0, board_size))
        center = (board_size - 1) / 2.0
        max_count = max(counts) if counts else 0
        for r, c in unknown_cells:
            idx = cell_index(r, c, board_size)
            score = counts[idx]
            if combined_probs is not None and max_count > 0 and not is_target_mode:
                norm_count = score / max_count
                mix = 0.35
                score = (1.0 - mix) * norm_count + mix * combined_probs[idx]
            if score < 0:
                continue
            is_endpoint = 1 if idx in endpoint_idxs else 0
            adj = _adjacent_hits((r, c), hit_cells_local) if hit_cells_local else 0
            dist2 = (r - center) ** 2 + (c - center) ** 2
            if score > best_score:
                best_score = score
                best_endpoint = is_endpoint
                best_adj = adj
                best_center = dist2
                best_cell = (r, c)
            elif score == best_score:
                if hit_cells_local:
                    if is_endpoint > best_endpoint:
                        best_endpoint = is_endpoint
                        best_adj = adj
                        best_cell = (r, c)
                    elif is_endpoint == best_endpoint and adj > best_adj:
                        best_adj = adj
                        best_cell = (r, c)
                else:
                    if dist2 < best_center:
                        best_center = dist2
                        best_cell = (r, c)

        if best_cell is None:
            return None
        return best_cell

    if strategy == "assigned_target_marginal":
        choice = _choose_assigned_target_marginal()
        if choice is not None:
            return choice
        strategy = "greedy"

    if strategy == "placement_factorized":
        choice = _choose_placement_factorized()
        if choice is not None:
            return choice
        return rng.choice(unknown_cells)
    if not has_any_hit and strategy in {"random_checkerboard", "systematic_checkerboard", "diagonal_stripe"}:
        if strategy == "random_checkerboard":
            whites = [(r, c) for (r, c) in unknown_cells if (r + c) % 2 == 0]
            return rng.choice(whites) if whites else rng.choice(unknown_cells)
        if strategy == "systematic_checkerboard":
            sorted_cells = sorted(unknown_cells, key=lambda p: (p[0], p[1]))
            whites = [p for p in sorted_cells if (p[0] + p[1]) % 2 == 0]
            return whites[0] if whites else sorted_cells[0]
        diagonals = [p for p in unknown_cells if (p[0] - p[1]) % 4 == 0]
        if diagonals:
            return rng.choice(diagonals)
        secondary = [p for p in unknown_cells if (p[0] - p[1]) % 2 == 0]
        if secondary:
            return rng.choice(secondary)
        return rng.choice(unknown_cells)

    # 1) Build World Model
    if posterior is None:
        sample_start = time.perf_counter()
        worlds_union, cell_hit_counts, _, N = sample_worlds(
            board,
            placements,
            ship_ids,
            confirmed_sunk,
            assigned_hits,
            rng_seed=rng.randint(0, 2 ** 31 - 1),
            board_size=board_size,
        )
        sample_time = time.perf_counter() - sample_start
        if profiler is not None:
            try:
                profiler.record_sample(sample_time, N, topup=False)
            except Exception:
                pass
    else:
        worlds_union = posterior.worlds_union
        cell_hit_counts = posterior.cell_hit_counts
        N = posterior.N

    if N <= 0:
        return rng.choice(unknown_cells)

    cell_probs = [cnt / N for cnt in cell_hit_counts]

    # --- TARGET MODE LOGIC (Refined) ---
    if hit_cells is None:
        hit_cells = _mask_to_cells(hit_mask, board_size)

    live_hit_set = set(hit_cells)
    if confirmed_sunk and assigned_hits:
        for ship in confirmed_sunk:
            for cell in assigned_hits.get(ship, set()):
                live_hit_set.discard(cell)

    frontier_mass = 0.0
    frontier_max = 0.0
    max_component = 0
    if live_hit_set:
        unknown_set = set(unknown_cells)
        frontier: Set[Tuple[int, int]] = set()
        visited: Set[Tuple[int, int]] = set()
        for start in live_hit_set:
            if start in visited:
                continue
            stack = [start]
            visited.add(start)
            comp_size = 0
            while stack:
                cr, cc = stack.pop()
                comp_size += 1
                idx = cell_index(cr, cc, board_size)
                for nr, nc in neighbors[idx]:
                    if (nr, nc) in live_hit_set and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        stack.append((nr, nc))
                    elif (nr, nc) in unknown_set:
                        frontier.add((nr, nc))
            if comp_size > max_component:
                max_component = comp_size
        if frontier:
            frontier_mass = sum(cell_probs[cell_index(r, c, board_size)] for r, c in frontier)
            frontier_max = max(cell_probs[cell_index(r, c, board_size)] for r, c in frontier)

    min_hits = _param_int(params, "target_min_hits", "SIM_TARGET_MIN_HITS", 1)
    mass_thresh = _param_float(params, "target_frontier_mass", "SIM_TARGET_FRONTIER_MASS", 0.20)
    max_thresh = _param_float(params, "target_frontier_max", "SIM_TARGET_FRONTIER_MAX", 0.15)
    comp_thresh = _param_int(params, "target_component_min", "SIM_TARGET_COMPONENT_MIN", 2)

    live_hits = len(live_hit_set)
    is_target_mode = False
    if live_hits >= min_hits and live_hits > 0:
        if frontier_mass >= mass_thresh or frontier_max >= max_thresh or max_component >= comp_thresh:
            is_target_mode = True

    if is_target_mode and strategy == "entropy1":
        strategy = "greedy"

    combined_probs: Optional[List[float]] = None
    if (
        opponent_prior is not None
        and isinstance(opponent_prior, list)
        and len(opponent_prior) == total_cells
        and not is_target_mode
    ):
        alpha = max(0.2, min(0.95, N / (N + 1200.0)))
        combined_probs = [
            alpha * cell_probs[i] + (1.0 - alpha) * float(opponent_prior[i])
            for i in range(total_cells)
        ]

    def _entropy_score_idx(idx: int) -> float:
        n_hit = cell_hit_counts[idx]
        if n_hit <= 0 or n_hit >= N:
            return 0.0
        p_hit = n_hit / N
        p_miss = 1.0 - p_hit
        return -(p_hit * math.log2(p_hit) + p_miss * math.log2(p_miss))

    def _choose_minlen_parity_entropy() -> Optional[Tuple[int, int]]:
        active_ships = [s for s in ship_ids if s not in confirmed_sunk]
        ship_lengths = _ship_lengths_from_placements(placements, active_ships)
        lengths = [l for l in ship_lengths.values() if l > 0]
        if not lengths:
            return None
        min_len = min(lengths)
        if min_len <= 1:
            return None
        if is_target_mode:
            choice = _choose_assigned_target_marginal()
            return choice
        buckets: Dict[int, float] = {}
        for r, c in unknown_cells:
            idx = cell_index(r, c, board_size)
            key = (r + c) % min_len
            prob = combined_probs[idx] if combined_probs is not None else cell_probs[idx]
            buckets[key] = buckets.get(key, 0.0) + prob
        if not buckets:
            return None
        best_bucket = max(buckets.items(), key=lambda kv: kv[1])[0]
        best_cell = None
        best_score = -1.0
        for r, c in unknown_cells:
            if (r + c) % min_len != best_bucket:
                continue
            idx = cell_index(r, c, board_size)
            score = _entropy_score_idx(idx)
            if score > best_score:
                best_score = score
                best_cell = (r, c)
        return best_cell

    if strategy == "minlen_parity_entropy":
        choice = _choose_minlen_parity_entropy()
        if choice is not None:
            return choice
        strategy = "greedy" if is_target_mode else "entropy1"

    if strategy == "endgame_exact_combo":
        active_ships = [s for s in ship_ids if s not in confirmed_sunk]
        max_ships = _param_int(params, "endgame_max_ships", "SIM_ENDGAME_MAX_SHIPS", 2)
        combo_limit = _param_int(params, "endgame_combo_limit", "SIM_ENDGAME_COMBO_LIMIT", 200000)
        candidates: List[Tuple[str, List[int]]] = []
        product = 1
        for ship in active_ships:
            req_mask = _mask_from_cells(assigned_hits.get(ship, set()), board_size)
            ship_masks: List[int] = []
            for p in placements.get(ship, []):
                pmask = getattr(p, "mask", 0)
                if pmask & miss_mask:
                    continue
                if req_mask and (req_mask & ~pmask) != 0:
                    continue
                ship_masks.append(pmask)
            if not ship_masks:
                candidates = []
                break
            product *= len(ship_masks)
            candidates.append((ship, ship_masks))
        if candidates and (len(active_ships) <= max_ships or product <= combo_limit):
            candidates.sort(key=lambda item: len(item[1]))
            cell_counts = [0] * total_cells
            total_worlds = 0

            def backtrack(i: int, used_mask: int) -> None:
                nonlocal total_worlds
                if i >= len(candidates):
                    total_worlds += 1
                    m = used_mask & unknown_mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cell_counts[idx] += 1
                        m ^= lsb
                    return
                for pmask in candidates[i][1]:
                    if pmask & used_mask:
                        continue
                    backtrack(i + 1, used_mask | pmask)

            backtrack(0, 0)
            if total_worlds > 0:
                best_cell = None
                best_prob = -1.0
                for r, c in unknown_cells:
                    idx = cell_index(r, c, board_size)
                    prob = cell_counts[idx] / total_worlds
                    if prob > best_prob:
                        best_prob = prob
                        best_cell = (r, c)
                if best_cell is not None:
                    return best_cell
        strategy = "greedy" if is_target_mode else "entropy1"

    if strategy == "ewa1_pruned":
        min_worlds = _param_int(params, "ewa_min_worlds", "SIM_EWA_MIN_WORLDS", 200)
        top_k = _param_int(params, "ewa_top_k", "SIM_EWA_TOPK", 32)
        if not worlds_union or N < min_worlds:
            strategy = "entropy1"
        else:
            known_mask = hit_mask | miss_mask
            scored: List[Tuple[float, int]] = []
            for r, c in unknown_cells:
                idx = cell_index(r, c, board_size)
                n_hit = cell_hit_counts[idx]
                n_miss = N - n_hit
                score = N - (n_hit * n_hit + n_miss * n_miss) / N
                scored.append((score, idx))
            scored.sort(key=lambda t: t[0], reverse=True)
            top_candidates = [idx for _score, idx in scored[: max(1, min(top_k, len(scored)))]]
            best_score = None
            best_idx = None
            for idx in top_candidates:
                bit = 1 << idx
                worlds_hit = [wm for wm in worlds_union if wm & bit]
                worlds_miss = [wm for wm in worlds_union if not (wm & bit)]
                Nh = len(worlds_hit)
                Nm = len(worlds_miss)
                if Nh + Nm == 0:
                    continue
                p_hit = Nh / (Nh + Nm)
                known_after = known_mask | bit
                Eh = compute_min_expected_worlds_after_one_shot(worlds_hit, known_after, board_size) if Nh > 0 else 0.0
                Em = compute_min_expected_worlds_after_one_shot(worlds_miss, known_after, board_size) if Nm > 0 else 0.0
                val = p_hit * Eh + (1.0 - p_hit) * Em
                if best_score is None or val < best_score:
                    best_score = val
                    best_idx = idx
            if best_idx is not None:
                return divmod(best_idx, board_size)
            strategy = "entropy1"

    if strategy == "meta_ucb_hybrid":
        arms_default = ["entropy1", "greedy", "minlen_parity_entropy", "placement_factorized"]
        arms_raw = os.getenv("SIM_META_UCB_ARMS")
        if arms_raw:
            arms = [a.strip() for a in arms_raw.split(",") if a.strip()]
        else:
            arms = arms_default
        if not arms:
            strategy = "entropy1"
        else:
            state = params.get("_meta_ucb_state")
            if not isinstance(state, dict) or state.get("board_id") != id(board):
                state = {
                    "board_id": id(board),
                    "counts": {a: 0 for a in arms},
                    "rewards": {a: 0.0 for a in arms},
                    "t": 0,
                }
                params["_meta_ucb_state"] = state

            def pick_arm_cell(arm: str) -> Optional[Tuple[int, int]]:
                if arm == "entropy1":
                    best = None
                    best_score = -1.0
                    for r, c in unknown_cells:
                        idx = cell_index(r, c, board_size)
                        score = _entropy_score_idx(idx)
                        if score > best_score:
                            best_score = score
                            best = (r, c)
                    return best
                if arm == "greedy":
                    best = None
                    best_score = -1.0
                    for r, c in unknown_cells:
                        idx = cell_index(r, c, board_size)
                        score = cell_probs[idx]
                        if score > best_score:
                            best_score = score
                            best = (r, c)
                    return best
                if arm == "minlen_parity_entropy":
                    return _choose_minlen_parity_entropy()
                if arm == "placement_factorized":
                    return _choose_placement_factorized()
                return None

            counts = state.get("counts", {})
            rewards = state.get("rewards", {})
            total = int(state.get("t", 0))
            c_bonus = _param_float(params, "meta_ucb_c", "SIM_META_UCB_C", 1.25)
            best_ucb = None
            best_arm = None
            best_cell = None
            for arm in arms:
                cell = pick_arm_cell(arm)
                if cell is None:
                    continue
                n = int(counts.get(arm, 0))
                avg = float(rewards.get(arm, 0.0)) / n if n > 0 else 0.0
                bonus = c_bonus * math.sqrt(math.log(total + 1.0) / (n + 1.0))
                ucb = avg + bonus
                if best_ucb is None or ucb > best_ucb:
                    best_ucb = ucb
                    best_arm = arm
                    best_cell = cell

            if best_cell is not None and best_arm is not None:
                idx = cell_index(best_cell[0], best_cell[1], board_size)
                n_hit = cell_hit_counts[idx]
                n_miss = N - n_hit
                info_gain = (N - (n_hit * n_hit + n_miss * n_miss) / N) / max(1.0, N)
                counts[best_arm] = int(counts.get(best_arm, 0)) + 1
                rewards[best_arm] = float(rewards.get(best_arm, 0.0)) + info_gain
                state["counts"] = counts
                state["rewards"] = rewards
                state["t"] = total + 1
                return best_cell
            strategy = "entropy1"

    if strategy == "two_ply":
        unknown_ratio = len(unknown_cells) / max(1, total_cells)
        min_worlds = _param_int(params, "two_ply_min_worlds", "SIM_TWO_PLY_MIN_WORLDS", 2000)
        min_unknown = _param_float(params, "two_ply_min_unknown_ratio", "SIM_TWO_PLY_MIN_UNKNOWN_RATIO", 0.60)
        if is_target_mode or N < min_worlds or unknown_ratio < min_unknown:
            strategy = "greedy"

    # --- ADVANCED STRATEGIES ---

    if strategy == "endpoint_phase":
        # Only use geometry in true target mode; otherwise hunt with entropy.
        if not is_target_mode:
            strategy = "entropy1"
        else:
            hits_all = list(live_hit_set) if live_hit_set else []
            hit_set = set(hits_all)
            neighbors = _neighbors4(board_size)

            # Connected components of HITs
            components: List[List[Tuple[int, int]]] = []
            visited: Set[Tuple[int, int]] = set()
            for start in hits_all:
                if start in visited:
                    continue
                stack = [start]
                visited.add(start)
                comp: List[Tuple[int, int]] = []
                while stack:
                    cur = stack.pop()
                    comp.append(cur)
                    idx = cell_index(cur[0], cur[1], board_size)
                    for nr, nc in neighbors[idx]:
                        if (nr, nc) in hit_set and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            stack.append((nr, nc))
                components.append(comp)

            best_local_score = -float("inf")
            best_local_cells: List[Tuple[int, int]] = []

            # Tunable weights
            w_prob = float(params.get("w_prob", 1.0))
            w_neighbor = float(params.get("w_neighbor", 0.2))
            w_endpoint = float(params.get("w_endpoint", 0.4))

            for comp in components:
                # Frontier = unknown neighbors of the component
                frontier: Set[Tuple[int, int]] = set()
                for hr, hc in comp:
                    idx = cell_index(hr, hc, board_size)
                    for nr, nc in neighbors[idx]:
                        if board[nr][nc] == EMPTY:
                            frontier.add((nr, nc))
                if not frontier:
                    continue

                # Alignment detection: if hits form a line, favor endpoints.
                aligned_row = (len(comp) >= 2) and all(r == comp[0][0] for r, _ in comp)
                aligned_col = (len(comp) >= 2) and all(c == comp[0][1] for _, c in comp)

                endpoint_cells: Set[Tuple[int, int]] = set()
                if aligned_row:
                    r0 = comp[0][0]
                    cols = [c for _, c in comp]
                    for cand_c in (min(cols) - 1, max(cols) + 1):
                        if 0 <= cand_c < board_size and board[r0][cand_c] == EMPTY:
                            endpoint_cells.add((r0, cand_c))
                elif aligned_col:
                    c0 = comp[0][1]
                    rows = [r for r, _ in comp]
                    for cand_r in (min(rows) - 1, max(rows) + 1):
                        if 0 <= cand_r < board_size and board[cand_r][c0] == EMPTY:
                            endpoint_cells.add((cand_r, c0))

                for r, c in frontier:
                    idx = cell_index(r, c, board_size)
                    p = cell_probs[idx]

                    # Neighbor bonus: prefer cells adjacent to multiple hits (clusters)
                    hit_neighbors = 0
                    idx2 = cell_index(r, c, board_size)
                    for nr, nc in neighbors[idx2]:
                        if board[nr][nc] == HIT:
                            hit_neighbors += 1

                    is_endpoint = 1.0 if (r, c) in endpoint_cells else 0.0
                    score = (w_prob * p) + (w_neighbor * hit_neighbors) + (w_endpoint * is_endpoint)

                    if score > best_local_score + 1e-12:
                        best_local_score = score
                        best_local_cells = [(r, c)]
                    elif abs(score - best_local_score) <= 1e-12:
                        best_local_cells.append((r, c))

            if best_local_cells:
                return rng.choice(best_local_cells)

            # If geometry does not find anything useful, fall back to greedy.
            strategy = "greedy"

    if strategy == "thompson_world":
        if is_target_mode:
            strategy = "greedy"
        else:
            if worlds_union:
                chosen_mask = rng.choice(worlds_union)
                targets = [
                    (r, c)
                    for (r, c) in unknown_cells
                    if (chosen_mask >> cell_index(r, c, board_size)) & 1
                ]
                if targets:
                    return rng.choice(targets)
            return rng.choice(unknown_cells)

    if strategy == "dynamic_parity":
        if is_target_mode:
            strategy = "greedy"
        else:
            alive = [s for s in ship_ids if s not in confirmed_sunk]
            step = 2
            # If the only ship left is size-3, a 3-color parity is sufficient.
            if len(alive) == 1 and alive[0] == "line3":
                step = 3

            best_cells: List[Tuple[int, int]] = []
            best_mass = -1.0
            for color in range(step):
                cells = [(r, c) for (r, c) in unknown_cells if (r + c) % step == color]
                if not cells:
                    continue
                mass = sum(cell_probs[cell_index(r, c, board_size)] for r, c in cells)
                if mass > best_mass:
                    best_mass = mass
                    best_cells = cells

        if strategy == "dynamic_parity":
            if best_cells:
                return max(best_cells, key=lambda rc: cell_probs[cell_index(rc[0], rc[1], board_size)])
            return rng.choice(unknown_cells)

    if strategy == "rollout_mcts":
        rollouts = max(2, int(params.get("rollouts", 6)))
        top_k = max(2, int(params.get("top_k", 5)))
        max_shots = int(params.get("max_shots", board_size * board_size))

        def info_gain(idx: int) -> float:
            n_hit = cell_hit_counts[idx]
            n_miss = N - n_hit
            return N - (n_hit * n_hit + n_miss * n_miss) / N

        if is_target_mode:
            ranked = sorted(
                unknown_cells,
                key=lambda rc: cell_probs[cell_index(rc[0], rc[1], board_size)],
                reverse=True,
            )
        else:
            ranked = sorted(
                unknown_cells,
                key=lambda rc: info_gain(cell_index(rc[0], rc[1], board_size)),
                reverse=True,
            )
        eval_cells = ranked[: min(top_k, len(ranked))]

        if not eval_cells or not worlds_union:
            strategy = "greedy"
        else:
            world_samples = worlds_union if len(worlds_union) <= rollouts else rng.sample(worlds_union, rollouts)
            neighbors = _neighbors4(board_size)
            base_hits = set(_mask_to_cells(hit_mask, board_size)) if hit_mask else set()
            base_misses = set(_mask_to_cells(miss_mask, board_size)) if miss_mask else set()

            def rollout_policy(sim_hits: Set[Tuple[int, int]], sim_misses: Set[Tuple[int, int]]) -> Tuple[int, int]:
                frontier = []
                for rr, cc in sim_hits:
                    idx = cell_index(rr, cc, board_size)
                    for nr, nc in neighbors[idx]:
                        if (nr, nc) not in sim_hits and (nr, nc) not in sim_misses:
                            frontier.append((nr, nc))
                if frontier:
                    return rng.choice(frontier)
                parity = [
                    (rr, cc)
                    for rr, cc in unknown_cells
                    if (rr, cc) not in sim_hits
                    and (rr, cc) not in sim_misses
                    and (rr + cc) % 2 == 0
                ]
                if parity:
                    return rng.choice(parity)
                remaining = [(rr, cc) for rr, cc in unknown_cells if (rr, cc) not in sim_hits and (rr, cc) not in sim_misses]
                return rng.choice(remaining) if remaining else (0, 0)

            def simulate_rollout(first_cell: Tuple[int, int], world_mask: int) -> int:
                sim_hits = set(base_hits)
                sim_misses = set(base_misses)
                total_targets = int(world_mask.bit_count())
                known_hits = int((world_mask & hit_mask).bit_count()) if hit_mask else 0
                remaining = max(0, total_targets - known_hits)

                shots = 0
                fr, fc = first_cell
                if (fr, fc) not in sim_hits and (fr, fc) not in sim_misses:
                    idx = cell_index(fr, fc, board_size)
                    is_hit = (world_mask >> idx) & 1
                    if is_hit:
                        sim_hits.add((fr, fc))
                        remaining -= 1
                    else:
                        sim_misses.add((fr, fc))
                    shots += 1

                while remaining > 0 and shots < max_shots:
                    r2, c2 = rollout_policy(sim_hits, sim_misses)
                    if (r2, c2) in sim_hits or (r2, c2) in sim_misses:
                        break
                    idx2 = cell_index(r2, c2, board_size)
                    is_hit = (world_mask >> idx2) & 1
                    if is_hit:
                        sim_hits.add((r2, c2))
                        remaining -= 1
                    else:
                        sim_misses.add((r2, c2))
                    shots += 1
                if remaining > 0 and shots >= max_shots:
                    shots += remaining
                return shots

            best_cell = None
            best_avg = None
            for cell in eval_cells:
                total = 0.0
                for wmask in world_samples:
                    total += simulate_rollout(cell, wmask)
                avg = total / len(world_samples)
                if best_avg is None or avg < best_avg:
                    best_avg = avg
                    best_cell = cell
            if best_cell is not None:
                return best_cell
            strategy = "greedy"

    if strategy == "softmax_greedy":
        if is_target_mode:
            strategy = "greedy"
        else:
            T = float(params.get("temperature", 0.10))
            # Filter based on non-zero counts for robustness.
            candidates = [(r, c) for (r, c) in unknown_cells if cell_hit_counts[cell_index(r, c, board_size)] > 0]
            if not candidates:
                return rng.choice(unknown_cells)

            ps = [cell_probs[cell_index(r, c, board_size)] for r, c in candidates]
            pmax = max(ps)
            denom = max(1e-6, T)
            weights = [math.exp((p - pmax) / denom) for p in ps]
            return rng.choices(candidates, weights=weights, k=1)[0]

    # --- STANDARD STRATEGIES ---

    if strategy == "hybrid_phase":
        strategy = "greedy" if is_target_mode else "entropy1"

    if strategy == "weighted_sample":
        candidates = []
        weights = []
        for r, c in unknown_cells:
            p = cell_probs[cell_index(r, c, board_size)]
            if p > 0:
                candidates.append((r, c))
                weights.append(p)
        if not candidates:
            return rng.choice(unknown_cells)
        return rng.choices(candidates, weights=weights, k=1)[0]

    if strategy == "random_checkerboard":
        if is_target_mode:
            strategy = "greedy"
        else:
            whites = [(r, c) for (r, c) in unknown_cells if (r + c) % 2 == 0]
            return rng.choice(whites) if whites else rng.choice(unknown_cells)

    if strategy == "systematic_checkerboard":
        if is_target_mode:
            strategy = "greedy"
        else:
            sorted_cells = sorted(unknown_cells, key=lambda p: (p[0], p[1]))
            whites = [p for p in sorted_cells if (p[0] + p[1]) % 2 == 0]
            return whites[0] if whites else sorted_cells[0]

    if strategy == "diagonal_stripe":
        if is_target_mode:
            strategy = "greedy"
        else:
            diagonals = [p for p in unknown_cells if (p[0] - p[1]) % 4 == 0]
            if diagonals:
                return rng.choice(diagonals)
            secondary = [p for p in unknown_cells if (p[0] - p[1]) % 2 == 0]
            if secondary:
                return rng.choice(secondary)
            return rng.choice(unknown_cells)

    if strategy == "two_ply":
        _, best_indices, _ = two_ply_selection(board, worlds_union, cell_hit_counts)
        if best_indices:
            idx = rng.choice(best_indices)
            return divmod(idx, board_size)
        strategy = "greedy"

    # --- SCORING FUNCTIONS ---

    def score_greedy(r: int, c: int) -> float:
        return cell_probs[cell_index(r, c, board_size)]

    def score_entropy(r: int, c: int) -> float:
        idx = cell_index(r, c, board_size)
        return _entropy_score_idx(idx)

    def score_adaptive_skew(r: int, c: int) -> float:
        base_score = cell_probs[cell_index(r, c, board_size)]
        center = (board_size - 1) / 2.0
        dist = math.sqrt((r - center) ** 2 + (c - center) ** 2)
        norm_dist = dist / math.sqrt(2 * center ** 2)
        unknown_ratio = len(unknown_cells) / (board_size * board_size)
        penalty = 0.0
        if unknown_ratio > 0.5:
            penalty = 0.20 * norm_dist
        return base_score - penalty

    def score_center_weighted(r: int, c: int) -> float:
        center = (board_size - 1) / 2.0
        dist2 = (r - center) ** 2 + (c - center) ** 2
        return cell_probs[cell_index(r, c, board_size)] / (1.0 + 0.25 * dist2)

    def score_ucb_explore(r: int, c: int) -> float:
        idx = cell_index(r, c, board_size)
        p = cell_probs[idx]
        c_bonus = float(params.get("ucb_c", 0.35))
        bonus = c_bonus * math.sqrt(max(0.0, p * (1.0 - p)))
        return p + bonus

    prefer_even = False
    if strategy == "parity_greedy":
        evens = [(rr, cc) for (rr, cc) in unknown_cells if (rr + cc) % 2 == 0]
        odds = [(rr, cc) for (rr, cc) in unknown_cells if (rr + cc) % 2 == 1]
        even_mass = sum(cell_probs[cell_index(rr, cc, board_size)] for rr, cc in evens)
        odd_mass = sum(cell_probs[cell_index(rr, cc, board_size)] for rr, cc in odds)
        prefer_even = even_mass >= odd_mass

    def score_parity_greedy(r: int, c: int) -> float:
        is_even = (r + c) % 2 == 0
        if prefer_even != is_even:
            return -1.0e9
        return cell_probs[cell_index(r, c, board_size)]

    # --- EXECUTE ---
    scorers = {
        "greedy": score_greedy,
        "entropy1": score_entropy,
        "parity_greedy": score_parity_greedy,
        "center_weighted": score_center_weighted,
        "adaptive_skew": score_adaptive_skew,
        "ucb_explore": score_ucb_explore,
    }
    scorer = scorers.get(strategy, score_greedy)

    best_score = -float("inf")
    best_candidates: List[Tuple[int, int]] = []
    for r, c in unknown_cells:
        s = scorer(r, c)
        if s > best_score + 1e-12:
            best_score = s
            best_candidates = [(r, c)]
        elif abs(s - best_score) <= 1e-12:
            best_candidates.append((r, c))

    return rng.choice(best_candidates)
