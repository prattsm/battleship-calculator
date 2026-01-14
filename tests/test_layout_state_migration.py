import json
import os
import tempfile
import unittest

from battleship.layouts.builtins import legacy_layout
from battleship.persistence.layout_state import layout_key, load_layout_state


class LayoutStateMigrationTests(unittest.TestCase):
    def test_legacy_payload_migrates_to_legacy_layout_bucket(self):
        legacy_payload = {"legacy_key": "legacy_value"}
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "state.json")
            with open(path, "w") as handle:
                json.dump(legacy_payload, handle)

            layout = legacy_layout()
            state, data = load_layout_state(path, layout)

            self.assertEqual(state, legacy_payload)
            self.assertIn(layout_key(layout), data.get("layouts", {}))


if __name__ == "__main__":
    unittest.main()
