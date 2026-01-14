import random

from battleship.layouts.definition import LayoutDefinition, ShipSpec
from battleship.layouts.placements import generate_layout_placements
from battleship.sim.attack_sim import simulate_model_game


def main() -> None:
    layout = LayoutDefinition(
        "smoke_tiny",
        "Smoke Tiny",
        4,
        (
            ShipSpec("s1", "line", length=2),
            ShipSpec("s2", "line", length=2),
        ),
    )
    placements = generate_layout_placements(layout)
    ship_ids = list(layout.ship_ids())

    shots = simulate_model_game(
        "random",
        placements,
        ship_ids,
        layout.board_size,
        rng=random.Random(0),
    )
    print(f"Smoke OK: shots={shots}")


if __name__ == "__main__":
    main()
