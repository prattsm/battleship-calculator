from typing import Tuple

from .definition import LayoutDefinition, ShipSpec


def legacy_layout() -> LayoutDefinition:
    ships = (
        ShipSpec(
            instance_id="square2",
            kind="shape",
            cells=((0, 0), (0, 1), (1, 0), (1, 1)),
            allow_rotations=True,
            name="2x2 square",
        ),
        ShipSpec(
            instance_id="L3",
            kind="shape",
            cells=((0, 0), (0, 1), (1, 0)),
            allow_rotations=True,
            name="L-tromino",
        ),
        ShipSpec(
            instance_id="line3",
            kind="line",
            length=3,
            allow_rotations=True,
            name="Length-3 line",
        ),
        ShipSpec(
            instance_id="line2",
            kind="line",
            length=2,
            allow_rotations=True,
            name="Length-2 line",
        ),
    )
    return LayoutDefinition(
        layout_id="legacy",
        name="Legacy (8x8 custom)",
        board_size=8,
        ships=ships,
        allow_touching=True,
        layout_version=1,
    )


def classic_layout() -> LayoutDefinition:
    ships = (
        ShipSpec(instance_id="line5", kind="line", length=5, allow_rotations=True, name="Carrier (5)"),
        ShipSpec(instance_id="line4", kind="line", length=4, allow_rotations=True, name="Battleship (4)"),
        ShipSpec(instance_id="line3a", kind="line", length=3, allow_rotations=True, name="Cruiser (3)"),
        ShipSpec(instance_id="line3b", kind="line", length=3, allow_rotations=True, name="Submarine (3)"),
        ShipSpec(instance_id="line2", kind="line", length=2, allow_rotations=True, name="Destroyer (2)"),
    )
    return LayoutDefinition(
        layout_id="classic",
        name="Classic Battleship (10x10)",
        board_size=10,
        ships=ships,
        allow_touching=True,
        layout_version=1,
    )


def builtin_layouts() -> Tuple[LayoutDefinition, ...]:
    return (classic_layout(), legacy_layout())
