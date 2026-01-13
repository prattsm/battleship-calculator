import json
import os
from typing import Optional


class StatsTracker:
    PATH = "battleship_stats.json"

    def __init__(self):
        self.games = 0
        self.wins = 0
        self.load()

    def load(self, path: Optional[str] = None):
        if path is None:
            path = self.PATH
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        self.games = int(data.get("games", 0))
        self.wins = int(data.get("wins", 0))

    def save(self, path: Optional[str] = None):
        if path is None:
            path = self.PATH
        data = {"games": self.games, "wins": self.wins}
        try:
            with open(path, "w") as f:
                json.dump(data, f)
        except OSError:
            pass

    def record_game(self, win: bool):
        self.games += 1
        if win:
            self.wins += 1
        self.save()

    def summary_text(self) -> str:
        if self.games <= 0:
            return "Games: 0, Wins: 0, Win rate: N/A"
        wr = 100.0 * self.wins / self.games
        return f"Games: {self.games}, Wins: {self.wins}, Win rate: {wr:.1f}%"
