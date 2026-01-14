import unittest

from battleship.domain.config import DISP_RADIUS
from battleship.layouts.builtins import legacy_layout
from battleship.layouts.cache import DEFAULT_PLACEMENT_CACHE
from battleship.sim.defense_sim import recommend_layout_ga


class DefenseSimGATests(unittest.TestCase):
    def test_ga_returns_valid_layout(self):
        layout = legacy_layout()
        runtime = DEFAULT_PLACEMENT_CACHE.get(layout)
        board_size = layout.board_size

        hit_counts = [
            [[0.0 for _ in range(board_size)] for _ in range(board_size)]
            for _ in range(4)
        ]
        miss_counts = [
            [[0.0 for _ in range(board_size)] for _ in range(board_size)]
            for _ in range(4)
        ]
        disp_counts = [
            [
                [
                    [0.0 for _ in range(2 * DISP_RADIUS + 1)]
                    for _ in range(2 * DISP_RADIUS + 1)
                ]
                for _ in range(2)
            ]
            for _ in range(4)
        ]

        layout_result, mask, robust, avg_heat, avg_seq = recommend_layout_ga(
            hit_counts,
            miss_counts,
            disp_counts,
            runtime.placements,
            ship_ids=list(layout.ship_ids()),
            board_size=board_size,
            generations=2,
            population_size=6,
            sim_games_per_layout=1,
            rng_seed=1,
        )

        self.assertIsNotNone(layout_result)
        self.assertIsInstance(layout_result, dict)
        self.assertGreater(mask, 0)

        union = 0
        total_cells = 0
        for placement in layout_result.values():
            union |= placement.mask
            total_cells += len(placement.cells)

        self.assertEqual(mask, union)
        self.assertEqual(union.bit_count(), total_cells)
        self.assertGreaterEqual(robust, 0.0)
        self.assertGreaterEqual(avg_heat, 0.0)
        self.assertGreaterEqual(avg_seq, 0.0)


if __name__ == "__main__":
    unittest.main()
