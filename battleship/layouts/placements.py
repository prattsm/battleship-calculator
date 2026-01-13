from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple

from .definition import Cell, LayoutDefinition, ShipSpec, normalize_shape_cells


@dataclass(frozen=True)
class LayoutPlacement:
    ship_id: str
    cells: Tuple[Cell, ...]
    mask: int
    adjacency_mask: int


def _mask_from_cells(cells: Iterable[Cell], board_size: int) -> int:
    mask = 0
    for r, c in cells:
        mask |= 1 << (r * board_size + c)
    return mask


def _adjacency_mask(cells: Iterable[Cell], board_size: int) -> int:
    mask = 0
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < board_size and 0 <= cc < board_size:
                    mask |= 1 << (rr * board_size + cc)
    return mask


def _rotate_cells(cells: Iterable[Cell]) -> List[Cell]:
    return [(-c, r) for r, c in cells]


def _orientations(cells: Iterable[Cell], allow_rotations: bool) -> List[Tuple[Cell, ...]]:
    base = normalize_shape_cells(cells)
    if not base:
        return []
    if not allow_rotations:
        return [base]

    seen: Set[Tuple[Cell, ...]] = set()
    cur = list(base)
    for _ in range(4):
        norm = normalize_shape_cells(cur)
        if norm not in seen:
            seen.add(norm)
        cur = _rotate_cells(cur)
    return list(seen)


def _ship_shape(spec: ShipSpec) -> Tuple[Cell, ...]:
    if spec.kind == "line":
        length = int(spec.length or 0)
        return tuple((0, c) for c in range(length))
    return tuple(spec.cells or [])


def generate_ship_placements(spec: ShipSpec, board_size: int, allow_touching: bool) -> List[LayoutPlacement]:
    shape = _ship_shape(spec)
    orientations = _orientations(shape, spec.allow_rotations)
    placements: List[LayoutPlacement] = []

    for orient in orientations:
        max_r = max(r for r, _ in orient)
        max_c = max(c for _, c in orient)
        for r0 in range(board_size - max_r):
            for c0 in range(board_size - max_c):
                cells = tuple((r0 + r, c0 + c) for r, c in orient)
                mask = _mask_from_cells(cells, board_size)
                adj = _adjacency_mask(cells, board_size) if not allow_touching else mask
                placements.append(LayoutPlacement(spec.instance_id, cells, mask, adj))

    return placements


def generate_layout_placements(layout: LayoutDefinition) -> Dict[str, List[LayoutPlacement]]:
    placements: Dict[str, List[LayoutPlacement]] = {}
    for spec in layout.ships:
        placements[spec.instance_id] = generate_ship_placements(
            spec,
            layout.board_size,
            layout.allow_touching,
        )
    return placements
