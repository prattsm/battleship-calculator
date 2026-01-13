import json
import os
import uuid
from typing import Any, Dict, List, Optional

from battleship.layouts.definition import LayoutDefinition, ShipSpec, normalize_shape_cells


CUSTOM_LAYOUTS_PATH = "battleship_custom_layouts.json"


def is_custom_layout_id(layout_id: str) -> bool:
    return layout_id.startswith("custom-")


def new_custom_layout_id() -> str:
    return f"custom-{uuid.uuid4().hex[:8]}"


def serialize_layout(layout: LayoutDefinition) -> Dict[str, Any]:
    ships = []
    for spec in layout.ships:
        ship = {
            "instance_id": spec.instance_id,
            "kind": spec.kind,
            "length": spec.length,
            "cells": [list(cell) for cell in (spec.cells or [])],
            "allow_rotations": bool(spec.allow_rotations),
            "name": spec.name or "",
        }
        ships.append(ship)

    return {
        "layout_id": layout.layout_id,
        "name": layout.name,
        "board_size": int(layout.board_size),
        "allow_touching": bool(layout.allow_touching),
        "layout_version": int(layout.layout_version),
        "ships": ships,
    }


def _ship_from_dict(data: Dict[str, Any]) -> Optional[ShipSpec]:
    instance_id = str(data.get("instance_id", "")).strip()
    kind = str(data.get("kind", "")).strip()
    name = str(data.get("name", "")).strip() or None
    allow_rotations = bool(data.get("allow_rotations", True))
    if kind == "line":
        length = int(data.get("length", 0))
        return ShipSpec(
            instance_id=instance_id,
            kind="line",
            length=length,
            allow_rotations=allow_rotations,
            name=name,
        )
    if kind == "shape":
        raw_cells = data.get("cells") or []
        cells = []
        for item in raw_cells:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                try:
                    cells.append((int(item[0]), int(item[1])))
                except (TypeError, ValueError):
                    continue
        norm = normalize_shape_cells(cells)
        return ShipSpec(
            instance_id=instance_id,
            kind="shape",
            cells=norm,
            allow_rotations=allow_rotations,
            name=name,
        )
    return None


def deserialize_layout(data: Dict[str, Any]) -> Optional[LayoutDefinition]:
    layout_id = str(data.get("layout_id", "")).strip()
    name = str(data.get("name", "")).strip()
    board_size = int(data.get("board_size", 0))
    allow_touching = bool(data.get("allow_touching", True))
    layout_version = int(data.get("layout_version", 1))
    ships_raw = data.get("ships") or []
    ships: List[ShipSpec] = []
    for item in ships_raw:
        if not isinstance(item, dict):
            continue
        spec = _ship_from_dict(item)
        if spec is not None:
            ships.append(spec)

    if not layout_id or not name or board_size <= 0:
        return None

    return LayoutDefinition(
        layout_id=layout_id,
        name=name,
        board_size=board_size,
        ships=tuple(ships),
        allow_touching=allow_touching,
        layout_version=layout_version,
    )


def load_custom_layouts(path: str = CUSTOM_LAYOUTS_PATH) -> List[LayoutDefinition]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []

    layouts_raw = data.get("layouts") if isinstance(data, dict) else None
    if not isinstance(layouts_raw, list):
        return []

    out: List[LayoutDefinition] = []
    for item in layouts_raw:
        if not isinstance(item, dict):
            continue
        layout = deserialize_layout(item)
        if layout is not None:
            out.append(layout)
    return out


def save_custom_layouts(layouts: List[LayoutDefinition], path: str = CUSTOM_LAYOUTS_PATH) -> None:
    data = {
        "schema": 1,
        "layouts": [serialize_layout(layout) for layout in layouts],
    }
    tmp_path = f"{path}.tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except OSError:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
