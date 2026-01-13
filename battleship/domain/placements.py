from typing import Dict, List, Tuple

from .config import BOARD_SIZE
from .types import Placement
from .board import make_mask


def generate_ship_placements() -> Dict[str, List[Placement]]:
    placements: Dict[str, List[Placement]] = {}

    # 2x2 square
    sq_list: List[Placement] = []
    for r in range(BOARD_SIZE - 1):
        for c in range(BOARD_SIZE - 1):
            cells = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
            sq_list.append(Placement("square2", tuple(cells), make_mask(cells)))
    placements["square2"] = sq_list

    # L-tromino: base shape (0,0),(0,1),(1,0)
    base = [(0, 0), (0, 1), (1, 0)]
    oriented_shapes: List[List[Tuple[int, int]]] = []
    seen = set()
    for rot in range(4):
        coords = []
        for x, y in base:
            if rot == 0:
                xr, yr = x, y
            elif rot == 1:
                xr, yr = -y, x
            elif rot == 2:
                xr, yr = -x, -y
            else:
                xr, yr = y, -x
            coords.append((xr, yr))
        min_r = min(r for r, _ in coords)
        min_c = min(c for _, c in coords)
        norm = [(r - min_r, c - min_c) for r, c in coords]
        key = tuple(sorted(norm))
        if key not in seen:
            seen.add(key)
            oriented_shapes.append(norm)
    L_list: List[Placement] = []
    for shape in oriented_shapes:
        max_dr = max(r for r, _ in shape)
        max_dc = max(c for _, c in shape)
        for r0 in range(BOARD_SIZE - max_dr):
            for c0 in range(BOARD_SIZE - max_dc):
                cells = [(r0 + r, c0 + c) for r, c in shape]
                L_list.append(Placement("L3", tuple(cells), make_mask(cells)))
    placements["L3"] = L_list

    # 3-long line
    line3: List[Placement] = []
    # horizontal
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE - 2):
            cells = [(r, c), (r, c + 1), (r, c + 2)]
            line3.append(Placement("line3", tuple(cells), make_mask(cells)))
    # vertical
    for r in range(BOARD_SIZE - 2):
        for c in range(BOARD_SIZE):
            cells = [(r, c), (r + 1, c), (r + 2, c)]
            line3.append(Placement("line3", tuple(cells), make_mask(cells)))
    placements["line3"] = line3

    # 2-long line
    line2: List[Placement] = []
    # horizontal
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE - 1):
            cells = [(r, c), (r, c + 1)]
            line2.append(Placement("line2", tuple(cells), make_mask(cells)))
    # vertical
    for r in range(BOARD_SIZE - 1):
        for c in range(BOARD_SIZE):
            cells = [(r, c), (r + 1, c)]
            line2.append(Placement("line2", tuple(cells), make_mask(cells)))
    placements["line2"] = line2

    return placements


PLACEMENTS = generate_ship_placements()
