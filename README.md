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

## User Guide
- Layout selector: choose built-in or custom layouts; last selection persists across launches.
- Attack tab: click cells to cycle states, use Shift/Alt/Ctrl shortcuts, undo/redo, and review heatmaps, “Why this shot”, and “What-if” previews.
- Defense tab: Place mode for positioning ships; Analyze mode to evaluate placements against attack models.
- Model Stats tab: run background sims, view report cards, and do parameter sweeps; best models are tracked per phase.
- Layouts tab: create/duplicate custom layouts, add line/shape ships, validate, and import/export layouts.
