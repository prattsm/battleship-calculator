#!/usr/bin/env python3
import argparse
import hashlib
import statistics
import time
from typing import Iterable, List

from battleship.layouts.builtins import builtin_layouts, classic_layout, legacy_layout
from battleship.layouts.cache import DEFAULT_PLACEMENT_CACHE
from battleship.sim.attack_sim import simulate_model_game
from battleship.strategies.registry import model_defs


def _stable_seed(global_seed: int, model_key: str, game_index: int) -> int:
    payload = f"{int(global_seed)}|{model_key}|{int(game_index)}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFF


def _percentile(values: List[int], pct: float) -> float:
    if not values:
        return 0.0
    if pct <= 0:
        return float(values[0])
    if pct >= 100:
        return float(values[-1])
    k = (len(values) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return float(values[f])
    d0 = values[f] * (c - k)
    d1 = values[c] * (k - f)
    return float(d0 + d1)


def _resolve_layout(name: str):
    name = (name or "classic").strip().lower()
    if name in {"classic", "10x10", "standard"}:
        return classic_layout()
    if name in {"legacy", "8x8"}:
        return legacy_layout()
    for layout in builtin_layouts():
        if layout.layout_id == name:
            return layout
    raise ValueError(f"Unknown layout '{name}'. Use classic or legacy.")


def _resolve_models(raw: str) -> List[str]:
    all_keys = [md.get("key") for md in model_defs() if md.get("key")]
    if not raw or raw.strip().lower() in {"all", "*"}:
        return list(all_keys)
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    resolved = []
    for key in keys:
        if key not in all_keys:
            raise ValueError(f"Unknown model key '{key}'.")
        resolved.append(key)
    return resolved


def _run_model(model_key: str, placements, ship_ids, board_size: int, games: int, seed: int) -> dict:
    shots: List[int] = []
    start = time.perf_counter()
    for i in range(games):
        rng_seed = _stable_seed(seed, model_key, i)
        import random

        rng = random.Random(rng_seed)
        shots.append(
            simulate_model_game(
                model_key,
                placements,
                ship_ids,
                board_size=board_size,
                rng=rng,
            )
        )
    elapsed = time.perf_counter() - start
    shots_sorted = sorted(shots)
    return {
        "games": games,
        "mean": statistics.mean(shots) if shots else 0.0,
        "median": statistics.median(shots_sorted) if shots_sorted else 0.0,
        "p90": _percentile(shots_sorted, 90.0),
        "p95": _percentile(shots_sorted, 95.0),
        "min": shots_sorted[0] if shots_sorted else 0,
        "max": shots_sorted[-1] if shots_sorted else 0,
        "time": elapsed,
    }


def main(argv: Iterable[str] = None) -> int:
    parser = argparse.ArgumentParser(description="Strategy harness: compare models on fixed RNG seeds.")
    parser.add_argument("--layout", default="classic", help="Layout id (classic|legacy)")
    parser.add_argument("--games", type=int, default=200, help="Games per model")
    parser.add_argument("--seed", type=int, default=1337, help="Global seed")
    parser.add_argument("--models", default="all", help="Comma-separated model keys or 'all'")
    args = parser.parse_args(list(argv) if argv is not None else None)

    layout = _resolve_layout(args.layout)
    runtime = DEFAULT_PLACEMENT_CACHE.get(layout)
    placements = runtime.placements
    ship_ids = list(layout.ship_ids())

    models = _resolve_models(args.models)

    print(f"Layout: {layout.name} ({layout.board_size}x{layout.board_size})")
    print(f"Games per model: {args.games}, Seed: {args.seed}")
    print()

    rows = []
    for key in models:
        stats = _run_model(key, placements, ship_ids, layout.board_size, args.games, args.seed)
        rows.append((key, stats))

    header = f"{'Model':<24} {'Mean':>6} {'Median':>6} {'P90':>6} {'P95':>6} {'Min':>5} {'Max':>5} {'Time(s)':>8}"
    print(header)
    print("-" * len(header))
    for key, s in rows:
        print(
            f"{key:<24} {s['mean']:>6.2f} {s['median']:>6.0f} {s['p90']:>6.0f} {s['p95']:>6.0f} "
            f"{s['min']:>5} {s['max']:>5} {s['time']:>8.2f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
