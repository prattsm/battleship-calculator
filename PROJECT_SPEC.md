# Battleship Solver – Full Refactor + Multi-Layout + Phase-Aware Model Selection (Project Spec)

## 1) Goal

Do a **full refactor** of the existing Battleship solver into a **modular, professional, expandable** codebase that supports:

1) **Classic Battleship as the default layout** (10×10, standard ships), while keeping the current ruleset as a **built-in custom/legacy layout**.  
2) **Multiple layouts** (rulesets), including user-defined layouts (board size + ship sets including polyomino shapes).  
3) **Per-layout model training/benchmarking**, and **automatic model selection** for the Attack tab.  
4) **Phase-aware selection**: choose the best model **per game phase** (hunt/target/endgame) for each layout.

Keep this spec **high-level**: Codex should make good engineering decisions, define clean interfaces, and implement the details.

---

## 2) Definitions

- **Layout (ruleset)**: board size + ship set + allowed orientations/rotations + placement constraints (overlap, adjacency rule optional).  
- **Model/Strategy**: algorithm that suggests the next shot given the current knowledge state.  
- **Game phase**: a coarse classification of state used to pick different models depending on situation.

---

## 3) Primary scope (what to implement)

### 3.1 Full refactor (required)
Refactor from a monolithic script to a clean architecture:
- Separate UI, domain logic, simulation engine, persistence, layouts, and strategies.
- Add clear interfaces so new layouts and new models can be added without touching every tab.
- Add reliable cancellation, safe saving, and caching where appropriate.

### 3.2 Features selected for implementation

#### Attack tab (implement: 1, 3, 5)
1) **Input ergonomics**
- Fast state toggles for each cell (unknown/miss/hit/etc).
- Keyboard shortcuts and/or quick actions.
- Turn history with undo/redo.

3) **Explanations (trust-building)**
- “Why this shot” panel: top candidates + score breakdown from the active model.
- Optional “what-if miss/hit” impact preview (high-level).

5) **Useful visualizations**
- Heatmap + alternative overlays (e.g., placement density).
- Inconsistency detection with warnings (if hits/misses imply an impossible state).

Also required: **Auto mode uses best model** (and best phase model) for the active layout.

#### Defense tab
- Add **Place vs Analyze modes**
  - Place mode: build/adjust your own layout.
  - Analyze mode: evaluate placement vulnerability and give suggestions.

- Add a general pathway to **improve accuracy**
  - Create an evaluation harness so defense suggestions can be compared and improved (metrics, baselines, and iterative enhancement).
  - Codex can decide the exact modeling approach (heuristics vs learning), but it must be measurable and testable.

#### Model stats tab (implement: 2, 3, 4, 5)
2) **Metrics beyond average**
- median, std, p90/p95 (shots-to-win), plus number of games.
- Optional tie-break logic for “best model”.

3) **Smarter sweeps**
- coarse-to-fine sweep, early stopping, budget controls (time-based runs allowed).
- Persist sweep configs + results.

4) **Background execution**
- Run simulations off the UI thread.
- Reliable cancel that does not crash.
- Periodic progress updates and optional checkpointing/resume.

5) **Model report cards / comparisons**
- Compare models side-by-side (layout-aware, and phase-aware).
- Head-to-head summaries or distribution comparisons (Codex chooses implementation).

#### Layouts (implement: 1, 2, 3)
1) **Strong validation + previews**
- Validate feasibility and placement counts per ship.
- Prevent invalid layouts from being saved.

2) **Versioning**
- `layout_version` and `layout_hash` (or similar).
- Mark old results as stale if layout changes.

3) **Ship editor supporting line + shape ships**
- Line ships (length, H/V).
- Shape ships (cell set + rotations).
- Support duplicates cleanly (unique ship instance ids).

#### General (implement: 2, 4)
2) **Auto best model guardrails**
- Auto-update toggles (switch only between games; lock model for this game).
- Show current selected model + phase mode.

4) **Performance/caching**
- Cache placements per layout.
- Cache derived masks/bitboards for fast scoring.
- Avoid recomputation on every UI update.

---

## 4) Layout system (default + legacy + custom)

### 4.1 Built-in layouts
1) **Classic Battleship (default)**
- Board: 10×10
- Ships: 5,4,3,3,2 (line ships, H/V only)
- Overlap forbidden
- Adjacency rule: implement as a layout option (default can be “allow touching” unless you prefer strict no-touch).

2) **Legacy/Current ruleset**
- Preserve the existing board size and ship shapes as a selectable built-in custom layout.

### 4.2 Custom layouts
Users can create layouts with:
- board size N×N (Codex can pick supported range)
- ship definitions (line or shape)
- per-ship instance IDs (to support duplicates)
- adjacency constraint option

### 4.3 Placement generation requirements
- Must generate all legal placements for each ship instance given the layout constraints.
- Must support rotations for polyomino ships if enabled.
- Must be cached per layout version/hash.

---

## 5) Phase-aware best model selection

### 5.1 Phase definitions (high-level)
Implement a simple phase classifier based on current attack board state + remaining ships:
- **HUNT**: no active hit clusters (no unresolved hits).
- **TARGET**: at least one unresolved hit cluster exists (hits not fully resolved into sunk ships).
- **ENDGAME**: remaining ship set is small (e.g., <= 2 ships) or similar; Codex picks a robust definition.

These phases must work for both line ships and polyomino ships (Codex decides how to define “unresolved cluster” for polyomino).

### 5.2 Best model per phase (per layout)
For each layout:
- Maintain `best_model_overall`
- Maintain `best_model_by_phase`: HUNT, TARGET, ENDGAME

Attack tab in Auto mode:
- chooses the best model for the current phase if available,
- otherwise falls back to overall best.

Model stats:
- computes metrics **overall** and **by phase**
- saves the best model for each phase and overall.

### 5.3 Guardrails in UI
- “Auto (best by phase)” (default)
- “Auto (best overall)” (optional)
- manual override: select a specific model/params

---

## 6) Tabs: high-level behavior requirements

### 6.1 Attack tab
- Must reflect the active layout (board size, ship list).
- Must support ergonomic entry, explanation panel, and overlays.
- Must display which model is active (and why: auto/phase/locked).
- Must validate the current evidence (hits/misses/sunk constraints) and warn on contradictions.

### 6.2 Defense tab
- Place mode: build a defensive placement for the active layout.
- Analyze mode: quantify vulnerability under chosen opponent attack model(s).
- Must include a measurable approach for improving accuracy:
  - baseline methods + new methods must be comparable with metrics and saved results.

### 6.3 Model stats tab
- Layout-aware and phase-aware benchmarking.
- Smarter sweeps, background execution, cancellation.
- Rich metrics and comparisons.
- “Set as best” for overall and for each phase.

---

## 7) Persistence & migration

### 7.1 Per-layout state
Persist state scoped by layout_id + layout_version/hash:
- Attack: board state, sunk flags, overlays, model override/lock, history (optional).
- Defense: placement, shot history (if relevant), analysis settings.
- Model stats: runs, sweeps, best overall + best-by-phase.

### 7.2 Migration
- Import existing legacy state into the legacy layout bucket.
- Ensure mismatched sizes do not crash loading; old state that cannot be applied should be ignored with a warning.

### 7.3 Atomic + safe saving
- Atomic writes (temp + rename).
- Autosave strategy (Codex chooses) to prevent data loss.

---

## 8) Suggested architecture (feel free to adapt)

Example modules:
- `app/` (entrypoints, wiring)
- `ui/` (PyQt widgets: tabs, dialogs, layout selector)
- `domain/` (board state, constraints, phases, ship definitions)
- `layouts/` (layout definitions, editor, validation, placement generation, caching)
- `strategies/` (strategy interface + implementations)
- `sim/` (simulation engine, sweep engine, metrics, report cards)
- `persistence/` (save/load, schema versions, migration)
- `utils/` (bitboards, rng, timing, logging)

Key interfaces (conceptual):
- `LayoutDefinition`, `LayoutRuntime`
- `Strategy` (suggest_shot(state, layout) -> move + explanation)
- `PhaseClassifier` (state, layout) -> phase
- `BenchmarkRunner` (layout, strategy, params, budget) -> metrics (overall + by phase)

---

## 9) Acceptance criteria

1) Default layout is **Classic Battleship** (10×10, 5 ships).  
2) Legacy layout reproduces current behavior.  
3) Layout switching is stable: boards resize and states are isolated per layout/version.  
4) Model stats runs in background, supports cancel, produces rich metrics and comparisons.  
5) Best model is tracked per layout **overall and by phase**, and Attack tab Auto uses it.  
6) Layout editor supports line + shape ships, validates, and versions layouts; results are marked stale on change.  
7) Caching keeps the UI responsive for reasonable board sizes and ship sets.

---

## 10) Implementation order (recommended)

1) Refactor skeleton + module boundaries + basic tests.  
2) Layout system (Classic + Legacy) + placement generation + caching.  
3) Strategy interface + adapt existing strategies to be layout-parameterized.  
4) Attack tab: ergonomic entry + explanations + overlays + phase-aware auto model selection.  
5) Model stats: background runner, metrics, sweeps, report cards, best overall/by-phase persistence.  
6) Defense tab: place/analyze split + evaluation harness for accuracy improvements.  
7) Custom layout editor + validation + versioning + import/export.  
8) Migration + polishing (autosave, warnings, performance).

