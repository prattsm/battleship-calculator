import random
from typing import Dict, List, Optional, Sequence, Tuple

from battleship.domain.board import cell_index
from battleship.domain.config import BOARD_SIZE, EMPTY, HIT, MISS
from battleship.domain.phase import PHASE_ENDGAME, PHASE_HUNT, PHASE_TARGET, classify_phase
from battleship.strategies.selection import _choose_next_shot_for_strategy


def _simulate_model_game(
    strategy: str,
    placements: Dict[str, List[object]],
    ship_ids: Sequence[str],
    board_size: int = BOARD_SIZE,
    rng: Optional[random.Random] = None,
    params: Optional[Dict[str, float]] = None,
    track_phases: bool = False,
) -> Tuple[int, Optional[Dict[str, int]]]:
    if rng is None:
        rng = random.Random()

    # 1. Generate a "True World" with full ship details
    # We need to know exactly where every ship is to simulate "Sunk" messages.
    true_layout: Dict[str, object] = {}
    used_mask = 0

    # Retry loop to ensure valid board (simple rejection sampling)
    while len(true_layout) < len(ship_ids):
        true_layout = {}
        used_mask = 0
        valid_board = True

        # Randomize order to prevent bias
        shuffled_ships = list(ship_ids)
        rng.shuffle(shuffled_ships)

        for ship in shuffled_ships:
            options = placements[ship]
            # Try 50 times to place this ship
            placed = False
            for _ in range(50):
                p = rng.choice(options)
                if not (p.mask & used_mask):
                    true_layout[ship] = p
                    used_mask |= p.mask
                    placed = True
                    break
            if not placed:
                valid_board = False
                break

        if not valid_board:
            continue  # Try generating the whole board again

    # 2. Setup Game State
    board = [[EMPTY for _ in range(board_size)] for _ in range(board_size)]
    confirmed_sunk = set()
    assigned_hits = {s: set() for s in ship_ids}

    # Track which cells belong to which ship for fast lookup
    cell_to_ship = {}
    for ship, p in true_layout.items():
        for r, c in p.cells:
            cell_to_ship[(r, c)] = ship

    shots = 0
    total_cells = board_size * board_size
    ships_remaining = len(ship_ids)

    # 3. Game Loop
    phase_counts: Optional[Dict[str, int]] = None
    if track_phases:
        phase_counts = {PHASE_HUNT: 0, PHASE_TARGET: 0, PHASE_ENDGAME: 0}

    while ships_remaining > 0 and shots < total_cells:
        if phase_counts is not None:
            phase = classify_phase(
                board,
                confirmed_sunk,
                assigned_hits,
                ship_ids,
                board_size=board_size,
            )
            phase_counts[phase] = phase_counts.get(phase, 0) + 1

        # Pass our "God Mode" knowledge into the bot
        r, c = _choose_next_shot_for_strategy(
            strategy,
            board,
            placements,
            rng,
            ship_ids,
            board_size,
            known_sunk=confirmed_sunk,
            known_assigned=assigned_hits,
            params=params,
        )

        idx = cell_index(r, c, board_size)
        is_hit = (used_mask >> idx) & 1

        board[r][c] = HIT if is_hit else MISS
        shots += 1

        if is_hit:
            # Identify which ship was hit
            hit_ship = cell_to_ship[(r, c)]

            # FEATURE 1: Auto-Assign Hits (Perfect Play)
            # The bot "knows" which ship it hit.
            assigned_hits[hit_ship].add((r, c))

            # FEATURE 2: Auto-Mark Sunk (Standard Rules)
            # Check if this ship is now fully sunk
            ship_cells = true_layout[hit_ship].cells
            is_sunk = all(board[sr][sc] == HIT for sr, sc in ship_cells)

            if is_sunk and hit_ship not in confirmed_sunk:
                confirmed_sunk.add(hit_ship)
                ships_remaining -= 1

    return shots, phase_counts


def simulate_model_game(
    strategy: str,
    placements: Dict[str, List[object]],
    ship_ids: Sequence[str],
    board_size: int = BOARD_SIZE,
    rng: Optional[random.Random] = None,
    params: Optional[Dict[str, float]] = None,
) -> int:
    shots, _ = _simulate_model_game(
        strategy,
        placements,
        ship_ids,
        board_size=board_size,
        rng=rng,
        params=params,
        track_phases=False,
    )
    return shots


def simulate_model_game_with_phases(
    strategy: str,
    placements: Dict[str, List[object]],
    ship_ids: Sequence[str],
    board_size: int = BOARD_SIZE,
    rng: Optional[random.Random] = None,
    params: Optional[Dict[str, float]] = None,
) -> Tuple[int, Dict[str, int]]:
    shots, phases = _simulate_model_game(
        strategy,
        placements,
        ship_ids,
        board_size=board_size,
        rng=rng,
        params=params,
        track_phases=True,
    )
    return shots, phases or {}
