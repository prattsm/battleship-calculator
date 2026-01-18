import math
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
) -> Tuple[int, int]:
    """
    Choose the next shot for the given strategy using the current board state.

    Notes:
    - `known_sunk` and `known_assigned` are carried across turns by the simulator.
    - `params` is optional and is used for tunable strategies (e.g., softmax temperature).
    """
    if params is None:
        params = {}

    unknown_cells = [
        (r, c)
        for r in range(board_size)
        for c in range(board_size)
        if board[r][c] == EMPTY
    ]
    if not unknown_cells:
        return 0, 0

    if strategy == "random":
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
    # Only enter target mode if we have *any* hit on the board and the posterior is confident.
    has_any_hit = any(board[r][c] == HIT for r in range(board_size) for c in range(board_size))
    max_p = max(cell_probs[cell_index(r, c, board_size)] for r, c in unknown_cells) if unknown_cells else 0.0
    is_target_mode = has_any_hit and (max_p > 0.30)

    # --- ADVANCED STRATEGIES ---

    if strategy == "endpoint_phase":
        # Only use geometry in true target mode; otherwise hunt with entropy.
        if not is_target_mode:
            strategy = "entropy1"
        else:
            hits_all = [(r, c) for r in range(board_size) for c in range(board_size) if board[r][c] == HIT]
            hit_set = set(hits_all)

            def neighbors4(rr: int, cc: int):
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < board_size and 0 <= nc < board_size:
                        yield nr, nc

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
                    for nr, nc in neighbors4(cur[0], cur[1]):
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
                    for nr, nc in neighbors4(hr, hc):
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

            def neighbours4(rr: int, cc: int):
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < board_size and 0 <= nc < board_size:
                        yield nr, nc

            def rollout_policy(sim_board: List[List[str]]) -> Tuple[int, int]:
                frontier = []
                for rr in range(board_size):
                    for cc in range(board_size):
                        if sim_board[rr][cc] != HIT:
                            continue
                        for nr, nc in neighbours4(rr, cc):
                            if sim_board[nr][nc] == EMPTY:
                                frontier.append((nr, nc))
                if frontier:
                    return rng.choice(frontier)
                parity = [(rr, cc) for rr in range(board_size) for cc in range(board_size)
                          if sim_board[rr][cc] == EMPTY and (rr + cc) % 2 == 0]
                if parity:
                    return rng.choice(parity)
                remaining = [(rr, cc) for rr in range(board_size) for cc in range(board_size)
                             if sim_board[rr][cc] == EMPTY]
                return rng.choice(remaining) if remaining else (0, 0)

            def simulate_rollout(first_cell: Tuple[int, int], world_mask: int) -> int:
                sim_board = [row[:] for row in board]
                total_targets = int(bin(world_mask).count("1"))
                hits = 0
                for rr in range(board_size):
                    for cc in range(board_size):
                        if sim_board[rr][cc] == HIT:
                            idx = cell_index(rr, cc, board_size)
                            if (world_mask >> idx) & 1:
                                hits += 1
                remaining = max(0, total_targets - hits)

                shots = 0
                fr, fc = first_cell
                if sim_board[fr][fc] == EMPTY:
                    idx = cell_index(fr, fc, board_size)
                    is_hit = (world_mask >> idx) & 1
                    sim_board[fr][fc] = HIT if is_hit else MISS
                    shots += 1
                    if is_hit:
                        remaining -= 1

                while remaining > 0 and shots < max_shots:
                    r2, c2 = rollout_policy(sim_board)
                    if sim_board[r2][c2] != EMPTY:
                        break
                    idx2 = cell_index(r2, c2, board_size)
                    is_hit = (world_mask >> idx2) & 1
                    sim_board[r2][c2] = HIT if is_hit else MISS
                    shots += 1
                    if is_hit:
                        remaining -= 1
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
        n_miss = N - n_hit
        if n_hit == 0 or n_miss == 0:
            return -1.0e9
        p_hit = n_hit / N
        p_miss = 1.0 - p_hit
        # Keep the existing "count-log" proxy but guard edge cases.
        E = p_hit * math.log(max(1, n_hit)) + p_miss * math.log(max(1, n_miss))
        return -E

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
        even_mass = sum(cell_probs[cell_index(rr, cc)] for rr, cc in evens)
        odd_mass = sum(cell_probs[cell_index(rr, cc)] for rr, cc in odds)
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
