# battleship-calculator
A probability engine and strategy simulator for Battleship. It calculates real-time hit percentages using Monte Carlo methods and allows users to test custom bot algorithms (like Thompson Sampling and geometric targeting) against thousands of board states.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install PyQt5
```

## Run
```bash
python run_battleship.py
```

## Tests
```bash
python -m unittest discover -s tests
```

## Smoke run
```bash
python scripts/smoke_run.py
```

## Strategy harness (CLI)
Compare strategies on fixed RNG seeds (deterministic across runs).

```bash
python scripts/strategy_harness.py --layout classic --games 200 --models all
```

## Model Sim Multiprocessing
Model Stats simulations can run in parallel to reduce wall-clock time. Results are deterministic for a fixed global seed, regardless of worker count or chunk size.

Env flags:
- `SIM_MULTIPROC=auto|1|0` (default: auto; enables if cpu>=4)
- `SIM_WORKERS=<int>` (clamped to 1..cpu; default: cpu-2, cap 12)
- `SIM_CHUNK_GAMES=<int>|auto` (default: 25; auto targets ~0.5-2.0s per chunk)
- `SIM_GLOBAL_SEED=<int>` (default: 1337; guarantees deterministic aggregates)
- `SIM_DEBUG_STATS=1` (adds assertions/logs for phase counts and checkpoint payloads)

## User Guide
- Layout selector: choose built-in or custom layouts; last selection persists across launches.
- Attack tab: click cells to cycle states, use Shift/Alt/Ctrl shortcuts, undo/redo, and review heatmaps, “Why this shot”, and “What-if” previews.
- Defense tab: Place mode for positioning ships; Analyze mode to evaluate placements against attack models.
- Model Stats tab: run background sims, view report cards, and do parameter sweeps; best models are tracked per phase.
- Layouts tab: create/duplicate custom layouts, add line/shape ships, validate, and import/export layouts.
