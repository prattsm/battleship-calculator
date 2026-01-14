import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

from battleship.domain.config import BOARD_SIZE, DISP_RADIUS, NO_SHOT


def build_base_heat(
    hit_counts,
    miss_counts,
    board_size: int = BOARD_SIZE,
    prior: float = 1.0,
    blend_k: float = 120.0,
):
    """Build a smoothed heatmap of opponent shot preferences.

    Uses shot counts (hit + miss) with a Dirichlet-like prior and blends
    toward uniform when data are sparse.
    """
    heat = [[0.0 for _ in range(board_size)] for _ in range(board_size)]
    shot_total = 0.0
    for r in range(board_size):
        for c in range(board_size):
            shot_total += float(hit_counts[r][c]) + float(miss_counts[r][c])

    total = 0.0
    for r in range(board_size):
        for c in range(board_size):
            v = float(hit_counts[r][c]) + float(miss_counts[r][c]) + float(prior)
            heat[r][c] = v
            total += v

    if total <= 0:
        uniform = 1.0 / (board_size * board_size)
        for r in range(board_size):
            for c in range(board_size):
                heat[r][c] = uniform
        return heat

    for r in range(board_size):
        for c in range(board_size):
            heat[r][c] /= total

    uniform = 1.0 / (board_size * board_size)
    confidence = shot_total / (shot_total + float(blend_k)) if blend_k > 0 else 1.0
    confidence = max(0.0, min(1.0, confidence))
    if confidence < 1.0:
        for r in range(board_size):
            for c in range(board_size):
                heat[r][c] = confidence * heat[r][c] + (1.0 - confidence) * uniform

    return heat


def weighted_random_choice(indices, weights, rng: random.Random) -> int:
    assert len(indices) == len(weights)
    if not indices:
        raise ValueError("no indices")
    total = sum(weights)
    if total <= 1e-12:
        return rng.choice(indices)
    x = rng.random() * total
    acc = 0.0
    for idx, w in zip(indices, weights):
        acc += w
        if x <= acc:
            return idx
    return indices[-1]


def compute_clusters_mask(layout_mask: int, board_size: int = BOARD_SIZE) -> Tuple[List[int], List[int]]:
    visited = [False] * (board_size * board_size)
    owner = [-1] * (board_size * board_size)
    clusters: List[int] = []
    for idx in range(board_size * board_size):
        bit = 1 << idx
        if not (layout_mask & bit) or visited[idx]:
            continue
        stack = [idx]
        visited[idx] = True
        cluster_mask = 0
        while stack:
            cur = stack.pop()
            cluster_mask |= 1 << cur
            owner[cur] = len(clusters)
            r = cur // board_size
            c = cur % board_size
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < board_size and 0 <= cc < board_size:
                    ni = rr * board_size + cc
                    if (layout_mask & (1 << ni)) and not visited[ni]:
                        visited[ni] = True
                        stack.append(ni)
        clusters.append(cluster_mask)
    return clusters, owner


def simulate_enemy_game_phase(
    layout_mask: int,
    base_heat_phase,
    disp_counts,
    model_type: str,
    rng: random.Random,
    board_size: int = BOARD_SIZE,
    max_shots: Optional[int] = None,
    initial_shot_board: Optional[List[List[int]]] = None,
) -> Tuple[int, int]:
    if max_shots is None:
        max_shots = board_size * board_size

    shot = [False] * (board_size * board_size)
    remaining_mask = layout_mask
    hits = 0
    shots_taken = 0

    if initial_shot_board:
        for r in range(board_size):
            for c in range(board_size):
                s = initial_shot_board[r][c]
                if s != NO_SHOT:
                    idx = r * board_size + c
                    shot[idx] = True
                    shots_taken += 1
                    if (layout_mask >> idx) & 1:
                        hits += 1
                        remaining_mask &= ~(1 << idx)
    frontier: Set[int] = set()

    cluster_masks, owner = compute_clusters_mask(layout_mask, board_size)
    cluster_remaining = list(cluster_masks)
    sunk_count = 0

    last_idx: Optional[int] = None
    last_was_hit: bool = False

    disp_sums = [
        [
            sum(
                disp_counts[phase][hm][dr][dc]
                for dr in range(2 * DISP_RADIUS + 1)
                for dc in range(2 * DISP_RADIUS + 1)
            )
            for hm in range(2)
        ]
        for phase in range(4)
    ]

    def neighbours(idx: int):
        r = idx // board_size
        c = idx % board_size
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < board_size and 0 <= cc < board_size:
                yield rr * board_size + cc

    def seq_weight(base_w: float, phase_idx: int, last_idx_local, last_was_hit_local, candidate_idx: int) -> float:
        if model_type != "seq":
            return base_w
        if last_idx_local is None:
            return base_w
        lr = last_idx_local // board_size
        lc = last_idx_local % board_size
        r = candidate_idx // board_size
        c = candidate_idx % board_size
        dr = r - lr
        dc = c - lc
        if abs(dr) > DISP_RADIUS or abs(dc) > DISP_RADIUS:
            return base_w
        hm_idx = 1 if last_was_hit_local else 0
        count = disp_counts[phase_idx][hm_idx][dr + DISP_RADIUS][dc + DISP_RADIUS]
        total = disp_sums[phase_idx][hm_idx]
        if total <= 0:
            return base_w
        alpha = 3.0
        confidence = total / (total + 40.0)
        factor = 1.0 + alpha * confidence * (count / total)
        return base_w * factor

    while shots_taken < max_shots and remaining_mask != 0:
        phase_idx = sunk_count
        if phase_idx > 3:
            phase_idx = 3
        heat = base_heat_phase[phase_idx]

        if frontier:
            choices = [idx for idx in frontier if not shot[idx]]
            if not choices:
                frontier.clear()
        if frontier:
            weights = []
            for idx in choices:
                r = idx // board_size
                c = idx % board_size
                base_w = heat[r][c]
                w = seq_weight(base_w, phase_idx, last_idx, last_was_hit, idx)
                weights.append(w)
            chosen = weighted_random_choice(choices, weights, rng)
            frontier.discard(chosen)
        else:
            choices = [idx for idx in range(board_size * board_size) if not shot[idx]]
            if not choices:
                break
            weights = []
            for idx in choices:
                r = idx // board_size
                c = idx % board_size
                base_w = heat[r][c]
                w = seq_weight(base_w, phase_idx, last_idx, last_was_hit, idx)
                weights.append(w)
            chosen = weighted_random_choice(choices, weights, rng)

        shot[chosen] = True
        shots_taken += 1
        bit = 1 << chosen
        if layout_mask & bit:
            hits += 1
            remaining_mask &= ~bit
            last_was_hit = True
            cid = owner[chosen]
            if cid != -1:
                cluster_remaining[cid] &= ~bit
                if cluster_remaining[cid] == 0:
                    sunk_count += 1
            for nb in neighbours(chosen):
                if not shot[nb]:
                    frontier.add(nb)
        else:
            last_was_hit = False
        last_idx = chosen

    return shots_taken, hits


def recommend_layout_phase(
    hit_counts_phase,
    miss_counts_phase,
    disp_counts,
    placements: Dict[str, List[object]],
    ship_ids: Optional[Sequence[str]] = None,
    board_size: int = BOARD_SIZE,
    n_iter: int = 250,
    sim_games_per_layout: int = 3,
    rng_seed: Optional[int] = None,
):
    rng = random.Random(rng_seed)

    base_heat_phase = []
    for p in range(4):
        base_heat_phase.append(build_base_heat(hit_counts_phase[p], miss_counts_phase[p], board_size))

    scored_placements: Dict[str, List[Tuple[object, float]]] = {}
    for ship, plist in placements.items():
        tmp = []
        for p in plist:
            s = sum(base_heat_phase[0][r][c] for (r, c) in p.cells)
            tmp.append((p, s))
        tmp.sort(key=lambda ps: ps[1])
        scored_placements[ship] = tmp

    best_layout = None
    best_mask = 0
    best_robust = -1.0
    best_heat = 0.0
    best_seq = 0.0

    ship_order = list(ship_ids) if ship_ids is not None else sorted(placements.keys())

    for _ in range(n_iter):
        used_mask = 0
        layout = {}
        rng.shuffle(ship_order)
        for ship in ship_order:
            candidates = [(p, s) for (p, s) in scored_placements[ship] if not (p.mask & used_mask)]
            if not candidates:
                layout = None
                break
            top_k = min(10, len(candidates))
            p, s = rng.choice(candidates[:top_k])
            layout[ship] = p
            used_mask |= p.mask
        if layout is None:
            continue
        union_mask = 0
        for p in layout.values():
            union_mask |= p.mask
        total_heat = 0.0
        total_seq = 0.0
        for _ in range(sim_games_per_layout):
            sh, _ = simulate_enemy_game_phase(
                union_mask,
                base_heat_phase,
                disp_counts,
                "heat",
                rng,
                board_size=board_size,
            )
            ss, _ = simulate_enemy_game_phase(
                union_mask,
                base_heat_phase,
                disp_counts,
                "seq",
                rng,
                board_size=board_size,
            )
            total_heat += sh
            total_seq += ss
        avg_heat = total_heat / sim_games_per_layout
        avg_seq = total_seq / sim_games_per_layout
        robust = min(avg_heat, avg_seq)
        if robust > best_robust:
            best_robust = robust
            best_layout = layout
            best_mask = union_mask
            best_heat = avg_heat
            best_seq = avg_seq

    return best_layout, best_mask, best_robust, best_heat, best_seq
