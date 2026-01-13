from typing import List, Set

from .definition import LayoutDefinition, ShipSpec


def validate_layout(layout: LayoutDefinition) -> List[str]:
    errors: List[str] = []

    if layout.board_size <= 0:
        errors.append("board_size must be positive")

    if not layout.ships:
        errors.append("layout must define at least one ship")

    ids: Set[str] = set()
    for ship in layout.ships:
        _validate_ship(ship, ids, errors)

    return errors


def _validate_ship(ship: ShipSpec, ids: Set[str], errors: List[str]) -> None:
    if not ship.instance_id:
        errors.append("ship instance_id must be non-empty")
    elif ship.instance_id in ids:
        errors.append(f"duplicate ship instance_id: {ship.instance_id}")
    else:
        ids.add(ship.instance_id)

    if ship.kind == "line":
        length = int(ship.length or 0)
        if length <= 0:
            errors.append(f"line ship {ship.instance_id} must have length > 0")
    elif ship.kind == "shape":
        if not ship.cells:
            errors.append(f"shape ship {ship.instance_id} must define cells")
    else:
        errors.append(f"ship {ship.instance_id} has unsupported kind: {ship.kind}")
