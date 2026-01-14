import unittest

from battleship.layouts.cache import PlacementCache
from battleship.layouts.definition import LayoutDefinition, ShipSpec


class PlacementCacheTests(unittest.TestCase):
    def test_cache_reuses_runtime(self):
        layout = LayoutDefinition(
            "tiny",
            "Tiny",
            2,
            (
                ShipSpec("s1", "line", length=1),
            ),
        )
        cache = PlacementCache()
        first = cache.get(layout)
        second = cache.get(layout)

        self.assertIs(first, second)
        self.assertIn("s1", first.placements)
        self.assertGreater(len(first.placements["s1"]), 0)


if __name__ == "__main__":
    unittest.main()
