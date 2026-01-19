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
    confirmed_sunk = known_sunk if known_sunk is not None else set()
    assigned_hits = known_assigned if known_assigned is not None else {s: set() for s in ship_ids}

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
    hit_mask = hit_mask or 0
    miss_mask = miss_mask or 0
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
        neighbors = _neighbors4(board_size)
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
                    for nr, nc in neighbors4(r, c):
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
        n_hit = cell_hit_counts[idx]
        if n_hit <= 0 or n_hit >= N:
            return 0.0
        p_hit = n_hit / N
        p_miss = 1.0 - p_hit
        return -(
            p_hit * math.log2(p_hit)
            + p_miss * math.log2(p_miss)
        )

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
