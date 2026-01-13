from typing import List, Optional, Tuple

from .config import BOARD_SIZE, EMPTY, HIT, MISS


def cell_index(r: int, c: int, board_size: int = BOARD_SIZE) -> int:
    return r * board_size + c


def make_mask(cells: List[Tuple[int, int]], board_size: int = BOARD_SIZE) -> int:
    m = 0
    for r, c in cells:
        m |= 1 << cell_index(r, c, board_size)
    return m


def create_board(board_size: int = BOARD_SIZE) -> List[List[str]]:
    return [[EMPTY for _ in range(board_size)] for _ in range(board_size)]


def board_masks(board: List[List[str]], board_size: Optional[int] = None) -> Tuple[int, int]:
    if board_size is None:
        board_size = len(board)
    hit_mask = 0
    miss_mask = 0
    for r in range(board_size):
        for c in range(board_size):
            idx = cell_index(r, c, board_size)
            if board[r][c] == HIT:
                hit_mask |= 1 << idx
            elif board[r][c] == MISS:
                miss_mask |= 1 << idx
    return hit_mask, miss_mask
