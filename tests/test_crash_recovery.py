import json
import os
import tempfile
import unittest

from battleship.layouts.builtins import legacy_layout
from battleship.persistence.app_state import load_match_state, load_selected_layout
from battleship.persistence.layout_state import layout_key, load_layout_state
from battleship.persistence.stats import StatsTracker


class CrashRecoveryTests(unittest.TestCase):
    def test_app_state_recovers_from_tmp_file_when_primary_is_corrupt(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "app_state.json")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("{not valid json")
            with open(f"{path}.tmp", "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "schema": 1,
                        "selected_layout": {
                            "layout_id": "classic",
                            "layout_version": 1,
                            "layout_hash": "abc123",
                        },
                        "match": {
                            "active_opponent": "CPU",
                            "records": {"CPU": {"wins": 2, "losses": 1}},
                        },
                    },
                    handle,
                )

            self.assertEqual(load_selected_layout(path)["layout_id"], "classic")
            self.assertEqual(load_match_state(path)["active_opponent"], "CPU")
            self.assertEqual(load_match_state(path)["records"]["CPU"]["wins"], 2)

    def test_app_state_prefers_newer_tmp_snapshot_over_older_primary(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "app_state.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "schema": 1,
                        "match": {
                            "active_opponent": "Old CPU",
                            "records": {"Old CPU": {"wins": 1, "losses": 0}},
                        },
                    },
                    handle,
                )
            with open(f"{path}.tmp", "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "schema": 1,
                        "match": {
                            "active_opponent": "New CPU",
                            "records": {"New CPU": {"wins": 3, "losses": 1}},
                        },
                    },
                    handle,
                )

            os.utime(f"{path}.tmp", None)

            self.assertEqual(load_match_state(path)["active_opponent"], "New CPU")

    def test_layout_state_recovers_from_tmp_file_when_primary_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "attack_state.json")
            layout = legacy_layout()
            expected_state = {"board": [[".", "x"]], "game_over": False}
            with open(f"{path}.tmp", "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "schema": 1,
                        "layouts": {
                            layout_key(layout): expected_state,
                        },
                    },
                    handle,
                )

            state, raw = load_layout_state(path, layout)

            self.assertEqual(state, expected_state)
            self.assertIn(layout_key(layout), raw["layouts"])

    def test_stats_tracker_recovers_from_tmp_file_when_primary_is_corrupt(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "stats.json")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("{broken")
            with open(f"{path}.tmp", "w", encoding="utf-8") as handle:
                json.dump({"games": 9, "wins": 4}, handle)

            tracker = StatsTracker()
            tracker.games = 0
            tracker.wins = 0
            tracker.load(path)

            self.assertEqual(tracker.games, 9)
            self.assertEqual(tracker.wins, 4)


if __name__ == "__main__":
    unittest.main()
