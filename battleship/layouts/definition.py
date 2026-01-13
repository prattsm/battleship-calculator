import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple


Cell = Tuple[int, int]


def normalize_shape_cells(cells: Iterable[Cell]) -> Tuple[Cell, ...]:
    """Normalize a shape's cells so its min row/col is at (0,0) and sort them."""
    cell_list = list(cells)
    if not cell_list:
        return tuple()
    min_r = min(r for r, _ in cell_list)
    min_c = min(c for _, c in cell_list)
    norm = [(r - min_r, c - min_c) for r, c in cell_list]
    return tuple(sorted(norm))


@dataclass(frozen=True)
class ShipSpec:
    instance_id: str
    kind: str  # "line" or "shape"
    length: Optional[int] = None
    cells: Optional[Tuple[Cell, ...]] = None
    allow_rotations: bool = True
    name: Optional[str] = None

    def normalized(self) -> dict:
        if self.kind == "line":
            return {
                "instance_id": self.instance_id,
                "kind": self.kind,
                "length": int(self.length or 0),
                "allow_rotations": bool(self.allow_rotations),
                "name": self.name or "",
            }
        return {
            "instance_id": self.instance_id,
            "kind": self.kind,
            "cells": list(normalize_shape_cells(self.cells or [])),
            "allow_rotations": bool(self.allow_rotations),
            "name": self.name or "",
        }


@dataclass(frozen=True)
class LayoutDefinition:
    layout_id: str
    name: str
    board_size: int
    ships: Tuple[ShipSpec, ...]
    allow_touching: bool = True
    layout_version: int = 1

    def normalized(self) -> dict:
        ships_sorted = sorted(self.ships, key=lambda s: s.instance_id)
        return {
            "layout_id": self.layout_id,
            "name": self.name,
            "board_size": int(self.board_size),
            "allow_touching": bool(self.allow_touching),
            "ships": [s.normalized() for s in ships_sorted],
        }

    @property
    def layout_hash(self) -> str:
        payload = json.dumps(self.normalized(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def ship_ids(self) -> Tuple[str, ...]:
        return tuple(s.instance_id for s in self.ships)
