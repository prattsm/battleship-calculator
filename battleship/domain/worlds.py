import math
import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .board import board_masks, make_mask
from .config import (
    BOARD_SIZE,
    ENUMERATION_PRODUCT_LIMIT,
    WORLD_MAX_ATTEMPTS_FACTOR,
    WORLD_SAMPLE_TARGET,
)

PlacementLike = object


def filter_allowed_placements(
    placements: Dict[str, List[PlacementLike]],
    hit_mask: int,
    miss_mask: int,
    confirmed_sunk: Set[str],
    assigned_hits: Dict[str, Set[Tuple[int, int]]],
    board_size: int = BOARD_SIZE,
) -> Dict[str, List[PlacementLike]]:
    filtered: Dict[str, List[PlacementLike]] = {}
    for ship, plist in placements.items():
        required_cells = assigned_hits.get(ship, set())
        required_mask = make_mask(list(required_cells), board_size) if required_cells else 0
        is_sunk = ship in confirmed_sunk
        new_list: List[PlacementLike] = []
        for p in plist:
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


def random_world(
    allowed: Dict[str, List[PlacementLike]],
    ship_ids: Sequence[str],
    hit_mask: int,
    rng: random.Random,
) -> Optional[Tuple[int, Tuple[int, ...]]]:
    used_mask = 0
    ship_masks: List[int] = []
    ship_sequence = list(ship_ids)
    rng.shuffle(ship_sequence)
    for ship in ship_sequence:
        options = allowed[ship]
        if not options:
            return None
        placed = False
        for _ in range(50):
            p = rng.choice(options)
            if p.mask & used_mask:
                continue
            used_mask |= p.mask
            ship_masks.append(p.mask)
            placed = True
            break
        if not placed:
            return None
    union_mask = used_mask
    if (union_mask & hit_mask) != hit_mask:
        return None
    return union_mask, tuple(ship_masks)


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
) -> Tuple[List[int], List[int], Dict[str, float], int]:
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
        placements, hit_mask, miss_mask, confirmed_sunk, assigned_hits, board_size
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

    worlds_union: List[int] = []
    worlds_ship_masks: List[Tuple[int, ...]] = []
    seen: Set[int] = set()

    if enumeration:
        # Exact enumeration (capped at ENUMERATION_PRODUCT_LIMIT worlds)
        worlds_union, worlds_ship_masks = enumerate_worlds(
            allowed, ship_ids, hit_mask, ENUMERATION_PRODUCT_LIMIT
        )
    else:
        # Monte Carlo sampling of distinct worlds
        rng = random.Random(rng_seed)
        attempts = 0
        max_attempts = WORLD_SAMPLE_TARGET * WORLD_MAX_ATTEMPTS_FACTOR

        while len(worlds_union) < WORLD_SAMPLE_TARGET and attempts < max_attempts:
            attempts += 1
            res = random_world(allowed, ship_ids, hit_mask, rng)
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
            else:
                # Not marked sunk: must still have at least one non-hit cell
                if (m_ship & ~hit_mask) == 0:
                    ok = False
                    break

        if ok:
            filtered_union.append(union_mask)
            filtered_ship_masks.append(ship_masks_tuple)

    worlds_union = filtered_union
    worlds_ship_masks = filtered_ship_masks
    # --------------------------------------------------------------

    N = len(worlds_union)
    if N == 0:
        # No worlds consistent with hits/misses + sunk checkboxes
        return [], [0] * (board_size * board_size), {
            s: 0.0 for s in ship_ids
        }, 0

    # Per-cell hit counts
    cell_hit_counts = [0] * (board_size * board_size)
    ship_sunk_counts = {s: 0 for s in ship_ids}

    for w_idx in range(N):
        union_mask = worlds_union[w_idx]

        # Cell hits
        m = union_mask
        idx = 0
        while idx < board_size * board_size:
            if m & 1:
                cell_hit_counts[idx] += 1
            m >>= 1
            idx += 1

        # Per-ship sunk: in this world, a ship is sunk if all its cells are hits
        ship_masks_tuple = worlds_ship_masks[w_idx]
        for i, ship in enumerate(ship_ids):
            m_ship = ship_masks_tuple[i]
            if (m_ship & ~hit_mask) == 0:
                ship_sunk_counts[ship] += 1

    ship_sunk_probs: Dict[str, float] = {}
    for ship in ship_ids:
        if ship in confirmed_sunk:
            ship_sunk_probs[ship] = 1.0
        else:
            ship_sunk_probs[ship] = ship_sunk_counts[ship] / N

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
