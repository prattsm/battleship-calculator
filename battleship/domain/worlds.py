import math
import os
import pickle
import random
import multiprocessing as mp
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

from .board import board_masks, make_mask
from .config import (
    BOARD_SIZE,
    ENUMERATION_PRODUCT_LIMIT,
    WORLD_MAX_ATTEMPTS_FACTOR,
    WORLD_SAMPLE_TARGET,
)

PlacementLike = object


def build_placement_index(
    placements: Dict[str, List[PlacementLike]],
    board_size: int = BOARD_SIZE,
) -> Dict[str, Dict[int, List[int]]]:
    index: Dict[str, Dict[int, List[int]]] = {}
    for ship, plist in placements.items():
        cell_map: Dict[int, List[int]] = {}
        for i, p in enumerate(plist):
            cells = getattr(p, "cells", None)
            if not cells:
                continue
            for r, c in cells:
                idx = r * board_size + c
                cell_map.setdefault(idx, []).append(i)
        index[ship] = cell_map
    return index


def filter_allowed_placements(
    placements: Dict[str, List[PlacementLike]],
    hit_mask: int,
    miss_mask: int,
    confirmed_sunk: Set[str],
    assigned_hits: Dict[str, Set[Tuple[int, int]]],
    board_size: int = BOARD_SIZE,
    placement_index: Optional[Dict[str, Dict[int, List[int]]]] = None,
) -> Dict[str, List[PlacementLike]]:
    filtered: Dict[str, List[PlacementLike]] = {}
    for ship, plist in placements.items():
        required_cells = assigned_hits.get(ship, set())
        required_mask = make_mask(list(required_cells), board_size) if required_cells else 0
        is_sunk = ship in confirmed_sunk
        candidates = plist
        if required_cells and placement_index:
            cell_map = placement_index.get(ship) or {}
            candidate_indices: Optional[Set[int]] = None
            for r, c in required_cells:
                idx = r * board_size + c
                idx_list = cell_map.get(idx)
                if not idx_list:
                    candidate_indices = set()
                    break
                if candidate_indices is None:
                    candidate_indices = set(idx_list)
                else:
                    candidate_indices &= set(idx_list)
                if not candidate_indices:
                    break
            if candidate_indices is not None:
                candidates = [plist[i] for i in sorted(candidate_indices)]

        new_list: List[PlacementLike] = []
        for p in candidates:
            # cannot overlap misses
            if p.mask & miss_mask:
                continue
            # must cover all assigned hits for this ship
            if required_mask and (required_mask & ~p.mask) != 0:
                continue
            # if user marked ship sunk, placement must lie entirely on hits
            if is_sunk and (p.mask & ~hit_mask) != 0:
                continue
            new_list.append(p)
        filtered[ship] = new_list
    return filtered


def _weighted_choice(rng: random.Random, options: List[PlacementLike], weights: Optional[List[float]] = None):
    if not options:
        return None
    if not weights or len(weights) != len(options):
        return rng.choice(options)
    total = float(sum(weights))
    if total <= 1e-12:
        return rng.choice(options)
    x = rng.random() * total
    acc = 0.0
    for opt, w in zip(options, weights):
        acc += float(w)
        if x <= acc:
            return opt
    return options[-1]


def random_world(
    allowed: Dict[str, List[PlacementLike]],
    ship_ids: Sequence[str],
    hit_mask: int,
    rng: random.Random,
    placement_weights: Optional[Dict[str, List[float]]] = None,
) -> Optional[Tuple[int, Tuple[int, ...]]]:
    used_mask = 0
    ship_masks_by_id: Dict[str, int] = {}
    ship_sequence = list(ship_ids)
    rng.shuffle(ship_sequence)
    for ship in ship_sequence:
        options = allowed[ship]
        if not options:
            return None
        placed = False
        for _ in range(50):
            weights = placement_weights.get(ship) if placement_weights else None
            p = _weighted_choice(rng, options, weights)
            if p is None:
                break
            if p.mask & used_mask:
                continue
            used_mask |= p.mask
            ship_masks_by_id[ship] = p.mask
            placed = True
            break
        if not placed:
            return None
    union_mask = used_mask
    if (union_mask & hit_mask) != hit_mask:
        return None
    ship_masks = tuple(ship_masks_by_id[s] for s in ship_ids)
    return union_mask, ship_masks


def _mc_worker(args):
    allowed, ship_ids, hit_mask, seed, target, max_attempts, placement_weights = args
    rng = random.Random(seed)
    seen: Set[int] = set()
    worlds_union: List[int] = []
    worlds_ship_masks: List[Tuple[int, ...]] = []
    attempts = 0
    while len(worlds_union) < target and attempts < max_attempts:
        attempts += 1
        res = random_world(allowed, ship_ids, hit_mask, rng, placement_weights=placement_weights)
        if res is None:
            continue
        union_mask, ship_masks_tuple = res
        if union_mask in seen:
            continue
        seen.add(union_mask)
        worlds_union.append(union_mask)
        worlds_ship_masks.append(ship_masks_tuple)
    return worlds_union, worlds_ship_masks


def filter_worlds_by_constraints(
    worlds_union: List[int],
    worlds_ship_masks: List[Tuple[int, ...]],
    ship_ids: Sequence[str],
    hit_mask: int,
    miss_mask: int,
    confirmed_sunk: Set[str],
    assigned_hit_masks: Sequence[int],
) -> Tuple[List[int], List[Tuple[int, ...]]]:
    filtered_union: List[int] = []
    filtered_ship_masks: List[Tuple[int, ...]] = []
    for union_mask, ship_masks_tuple in zip(worlds_union, worlds_ship_masks):
        if union_mask & miss_mask:
            continue
        if (union_mask & hit_mask) != hit_mask:
            continue
        ok = True
        for i, ship in enumerate(ship_ids):
            m_ship = ship_masks_tuple[i]
            req_mask = assigned_hit_masks[i] if i < len(assigned_hit_masks) else 0
            if req_mask and (req_mask & ~m_ship) != 0:
                ok = False
                break
            if ship in confirmed_sunk and (m_ship & ~hit_mask) != 0:
                ok = False
                break
        if ok:
            filtered_union.append(union_mask)
            filtered_ship_masks.append(ship_masks_tuple)
    return filtered_union, filtered_ship_masks


def summarize_worlds(
    worlds_union: List[int],
    worlds_ship_masks: List[Tuple[int, ...]],
    ship_ids: Sequence[str],
    hit_mask: int,
    board_size: int,
    confirmed_sunk: Optional[Set[str]] = None,
) -> Tuple[List[int], Dict[str, float]]:
    N = len(worlds_union)
    cell_hit_counts = [0] * (board_size * board_size)
    ship_sunk_counts = {s: 0 for s in ship_ids}

    for union_mask, ship_masks_tuple in zip(worlds_union, worlds_ship_masks):
        m = union_mask
        idx = 0
        while idx < board_size * board_size:
            if m & 1:
                cell_hit_counts[idx] += 1
            m >>= 1
            idx += 1

        for i, ship in enumerate(ship_ids):
            m_ship = ship_masks_tuple[i]
            if (m_ship & ~hit_mask) == 0:
                ship_sunk_counts[ship] += 1

    ship_sunk_probs: Dict[str, float] = {}
    for ship in ship_ids:
        if confirmed_sunk and ship in confirmed_sunk:
            ship_sunk_probs[ship] = 1.0
        else:
            ship_sunk_probs[ship] = ship_sunk_counts[ship] / N if N > 0 else 0.0
    return cell_hit_counts, ship_sunk_probs


def enumerate_worlds(
    allowed: Dict[str, List[PlacementLike]],
    ship_ids: Sequence[str],
    hit_mask: int,
    max_worlds: int,
) -> Tuple[List[int], List[Tuple[int, ...]]]:
    worlds_union: List[int] = []
    worlds_ship_masks: List[Tuple[int, ...]] = []

    def backtrack(i: int, used_mask: int, ship_masks: List[int]):
        if len(worlds_union) >= max_worlds:
            return
        if i == len(ship_ids):
            union_mask = used_mask
            if (union_mask & hit_mask) != hit_mask:
                return
            worlds_union.append(union_mask)
            worlds_ship_masks.append(tuple(ship_masks))
            return
        ship = ship_ids[i]
        for p in allowed[ship]:
            if p.mask & used_mask:
                continue
            ship_masks.append(p.mask)
            backtrack(i + 1, used_mask | p.mask, ship_masks)
            ship_masks.pop()

    backtrack(0, 0, [])
    return worlds_union, worlds_ship_masks


def sample_worlds(
    board: List[List[str]],
    placements: Dict[str, List[PlacementLike]],
    ship_ids: Sequence[str],
    confirmed_sunk: Set[str],
    assigned_hits: Dict[str, Set[Tuple[int, int]]],
    rng_seed: Optional[int] = None,
    board_size: Optional[int] = None,
    cell_prior: Optional[List[float]] = None,
    target_worlds: Optional[int] = None,
    max_attempts_factor: Optional[int] = None,
    existing_union: Optional[List[int]] = None,
    existing_ship_masks: Optional[List[Tuple[int, ...]]] = None,
    return_ship_masks: bool = False,
    placement_index: Optional[Dict[str, Dict[int, List[int]]]] = None,
) -> Union[
    Tuple[List[int], List[int], Dict[str, float], int],
    Tuple[List[int], List[Tuple[int, ...]], List[int], Dict[str, float], int],
]:
    """
    Return:
      - list of union masks for each world
      - per-cell hit counts
      - per-ship sunk probabilities
      - number of worlds sampled
    """

    if board_size is None:
        board_size = len(board)

    hit_mask, miss_mask = board_masks(board, board_size)
    allowed = filter_allowed_placements(
        placements,
        hit_mask,
        miss_mask,
        confirmed_sunk,
        assigned_hits,
        board_size,
        placement_index=placement_index,
    )

    # If any ship has no legal placement, there are no consistent worlds.
    for ship in ship_ids:
        if not allowed[ship]:
            return [], [0] * (board_size * board_size), {
                s: 0.0 for s in ship_ids
            }, 0

    # Decide enumeration vs Monte Carlo
    remaining_ships = [s for s in ship_ids if s not in confirmed_sunk]
    force_enumeration = (len(remaining_ships) == 1)

    product = 1
    enumeration = True
    if not force_enumeration:
        for ship in ship_ids:
            n = len(allowed[ship])
            product *= n
            if product > ENUMERATION_PRODUCT_LIMIT:
                enumeration = False
                break
    else:
        enumeration = True

    # If we have a cell prior, prefer Monte Carlo sampling to respect weights.
    if cell_prior is not None and len(cell_prior) == board_size * board_size:
        enumeration = False

    worlds_union: List[int] = []
    worlds_ship_masks: List[Tuple[int, ...]] = []
    seen: Set[int] = set()

    if existing_union and existing_ship_masks and len(existing_union) == len(existing_ship_masks):
        for union_mask, ship_masks_tuple in zip(existing_union, existing_ship_masks):
            if union_mask in seen:
                continue
            seen.add(union_mask)
            worlds_union.append(union_mask)
            worlds_ship_masks.append(ship_masks_tuple)

    if enumeration:
        # Exact enumeration (capped at ENUMERATION_PRODUCT_LIMIT worlds)
        worlds_union, worlds_ship_masks = enumerate_worlds(
            allowed, ship_ids, hit_mask, ENUMERATION_PRODUCT_LIMIT
        )
    else:
        # Monte Carlo sampling of distinct worlds
        rng = random.Random(rng_seed)
        target = target_worlds if isinstance(target_worlds, int) and target_worlds > 0 else WORLD_SAMPLE_TARGET
        attempts = 0
        factor = max_attempts_factor if isinstance(max_attempts_factor, int) and max_attempts_factor > 0 else WORLD_MAX_ATTEMPTS_FACTOR
        needed = max(0, target - len(worlds_union))
        max_attempts = max(1, needed * factor)
        placement_weights: Optional[Dict[str, List[float]]] = None
        if cell_prior is not None and len(cell_prior) == board_size * board_size:
            placement_weights = {}
            for ship, plist in allowed.items():
                if not plist:
                    placement_weights[ship] = []
                    continue
                weights = []
                for p in plist:
                    if not getattr(p, "cells", None):
                        weights.append(1.0)
                        continue
                    s = 0.0
                    for r, c in p.cells:
                        idx = r * board_size + c
                        if 0 <= idx < len(cell_prior):
                            s += float(cell_prior[idx])
                    weights.append(max(1e-6, s))
                placement_weights[ship] = weights

        use_parallel = False
        if needed >= 1000 and os.cpu_count() and os.cpu_count() > 1:
            try:
                start_method = mp.get_start_method()
            except RuntimeError:
                start_method = mp.get_start_method(allow_none=True) or "spawn"
            if start_method == "fork":
                try:
                    pickle.dumps(allowed)
                    use_parallel = True
                except Exception:
                    use_parallel = False

        if use_parallel and needed > 0:
            workers = min(os.cpu_count() or 1, 4)
            per_worker = max(1, int(math.ceil(needed / workers)))
            per_attempts = max(1, per_worker * factor)
            seeds = [rng.randint(0, 2**31 - 1) for _ in range(workers)]
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=workers) as pool:
                args_list = [
                    (allowed, ship_ids, hit_mask, seeds[i], per_worker, per_attempts, placement_weights)
                    for i in range(workers)
                ]
                for worker_union, worker_masks in pool.map(_mc_worker, args_list):
                    for union_mask, ship_masks_tuple in zip(worker_union, worker_masks):
                        if len(worlds_union) >= target:
                            break
                        if union_mask in seen:
                            continue
                        seen.add(union_mask)
                        worlds_union.append(union_mask)
                        worlds_ship_masks.append(ship_masks_tuple)

        while len(worlds_union) < target and attempts < max_attempts:
            attempts += 1
            res = random_world(allowed, ship_ids, hit_mask, rng, placement_weights=placement_weights)
            if res is None:
                continue
            union_mask, ship_masks_tuple = res
            if union_mask in seen:
                continue
            seen.add(union_mask)
            worlds_union.append(union_mask)
            worlds_ship_masks.append(ship_masks_tuple)

    # ---------- NEW: enforce sunk / not-sunk consistency ----------
    # For each world:
    #  - If a ship is marked sunk, all its cells must be hits.
    #  - If a ship is NOT marked sunk, it must have at least one non-hit cell.
    filtered_union: List[int] = []
    filtered_ship_masks: List[Tuple[int, ...]] = []

    for w_idx in range(len(worlds_union)):
        union_mask = worlds_union[w_idx]
        ship_masks_tuple = worlds_ship_masks[w_idx]

        ok = True
        for i, ship in enumerate(ship_ids):
            m_ship = ship_masks_tuple[i]
            is_sunk = ship in confirmed_sunk

            if is_sunk:
                # Marked sunk: cannot extend into any non-hit cell
                if (m_ship & ~hit_mask) != 0:
                    ok = False
                    break
            # If not marked sunk, leave it unconstrained (unknown).

        if ok:
            filtered_union.append(union_mask)
            filtered_ship_masks.append(ship_masks_tuple)

    worlds_union = filtered_union
    worlds_ship_masks = filtered_ship_masks
    # --------------------------------------------------------------

    N = len(worlds_union)
    if N == 0:
        # No worlds consistent with hits/misses + sunk checkboxes
        if return_ship_masks:
            return [], [], [0] * (board_size * board_size), {s: 0.0 for s in ship_ids}, 0
        return [], [0] * (board_size * board_size), {s: 0.0 for s in ship_ids}, 0

    cell_hit_counts, ship_sunk_probs = summarize_worlds(
        worlds_union, worlds_ship_masks, ship_ids, hit_mask, board_size, confirmed_sunk=confirmed_sunk
    )

    if return_ship_masks:
        return worlds_union, worlds_ship_masks, cell_hit_counts, ship_sunk_probs, N
    return worlds_union, cell_hit_counts, ship_sunk_probs, N


def compute_cell_hit_counts_from_worlds(world_masks: List[int], board_size: int) -> List[int]:
    counts = [0] * (board_size * board_size)
    for wm in world_masks:
        m = wm
        idx = 0
        while idx < board_size * board_size:
            if m & 1:
                counts[idx] += 1
            m >>= 1
            idx += 1
    return counts


def compute_min_expected_worlds_after_one_shot(
    world_masks: List[int],
    known_mask: int,
    board_size: int,
) -> float:
    """
    For a given set of consistent worlds and current known_mask, compute
    the minimum expected *log* world count after one additional optimal shot.

    This is an information-theoretic refinement of the previous heuristic:
    instead of minimizing E[N_after], we minimize E[log N_after],
    which is proportional to expected posterior entropy.
    """
    N = len(world_masks)
    if N == 0:
        return 0.0

    cell_hit_counts = compute_cell_hit_counts_from_worlds(world_masks, board_size)
    best_E = float("inf")

    for idx in range(board_size * board_size):
        # Skip cells that are already known (hit or miss).
        if (known_mask >> idx) & 1:
            continue

        n_hit = cell_hit_counts[idx]
        n_miss = N - n_hit

        if n_hit == 0 or n_miss == 0:
            # Outcome of this shot is effectively deterministic;
            # we do not shrink the set of plausible layouts at all.
            # Expected log(worlds) stays at log(N).
            E = math.log(N)
        else:
            p_hit = n_hit / N
            p_miss = n_miss / N
            # Expected log of remaining world count after this shot:
            # branch sizes are n_hit or n_miss.
            E = p_hit * math.log(n_hit) + p_miss * math.log(n_miss)

        if E < best_E:
            best_E = E

    if best_E == float("inf"):
        return math.log(N)
    return best_E
