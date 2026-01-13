from typing import Dict, Iterable, Optional, Sequence, Set, Tuple

from .config import HIT

PHASE_HUNT = "HUNT"
PHASE_TARGET = "TARGET"
PHASE_ENDGAME = "ENDGAME"


def classify_phase(
    board: Sequence[Sequence[str]],
    confirmed_sunk: Set[str],
    assigned_hits: Dict[str, Set[Tuple[int, int]]],
    ship_ids: Iterable[str],
    board_size: Optional[int] = None,
    endgame_ship_threshold: int = 2,
) -> str:
    if board_size is None:
        board_size = len(board)

    remaining = [s for s in ship_ids if s not in confirmed_sunk]
    if len(remaining) <= endgame_ship_threshold:
        return PHASE_ENDGAME

    resolved_hits: Set[Tuple[int, int]] = set()
    for ship in confirmed_sunk:
        resolved_hits.update(assigned_hits.get(ship, set()))

    for r in range(board_size):
        for c in range(board_size):
            if board[r][c] == HIT and (r, c) not in resolved_hits:
                return PHASE_TARGET

    return PHASE_HUNT
