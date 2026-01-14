import unittest

from battleship.layouts.definition import LayoutDefinition, ShipSpec
from battleship.layouts.validation import validate_layout


class LayoutValidationTests(unittest.TestCase):
    def test_empty_layout_reports_errors(self):
        layout = LayoutDefinition("test", "Test", 0, tuple())
        errors = validate_layout(layout)
        self.assertIn("board_size must be positive", errors)
        self.assertIn("layout must define at least one ship", errors)

    def test_invalid_ships_report_errors(self):
        layout = LayoutDefinition(
            "test",
            "Test",
            5,
            (
                ShipSpec("a", "line", length=0),
                ShipSpec("a", "shape", cells=tuple()),
            ),
        )
        errors = validate_layout(layout)
        self.assertTrue(any("duplicate ship instance_id" in e for e in errors))
        self.assertTrue(any("must have length > 0" in e for e in errors))
        self.assertTrue(any("must define cells" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
