#!/usr/bin/env python3
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set
import json, math, os, random, itertools
from datetime import datetime

from PyQt5 import QtWidgets, QtCore, QtGui
import traceback

import sys

# -----------------------------
# Debug helpers (enable with --debug or env BATTLESHIP_DEBUG=1)
# -----------------------------
DEBUG_ENABLED = False
DEBUG_LOG_PATH = "battleship_debug.log"


def _debug_log_line(line: str) -> None:
    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {line}\n")
    except Exception:
        pass


def debug_event(parent, title: str, message: str, details: str = "", *, force_popup: bool = False,
                level: str = "info") -> None:
    """Log a debug event and optionally show a popup."""
    try:
        _debug_log_line(f"{level.upper()} | {title} | {message}")
        if details:
            for ln in details.splitlines():
                _debug_log_line(f"    {ln}")
    except Exception:
        pass

    if not (DEBUG_ENABLED or force_popup):
        return

    try:
        box = QtWidgets.QMessageBox(parent)
        box.setWindowTitle(title)
        box.setText(message)
        if details:
            box.setDetailedText(details)
        if level == "error":
            box.setIcon(QtWidgets.QMessageBox.Critical)
        elif level == "warning":
            box.setIcon(QtWidgets.QMessageBox.Warning)
        else:
            box.setIcon(QtWidgets.QMessageBox.Information)
        box.exec_()
    except Exception:
        pass


BOARD_SIZE = 8

EMPTY = "."
MISS = "o"
HIT = "x"

WORLD_SAMPLE_TARGET = 10000
WORLD_MAX_ATTEMPTS_FACTOR = 30

# Enumeration: if product of allowed placements for ships is <= this,
# we enumerate all valid layouts exactly instead of sampling.
ENUMERATION_PRODUCT_LIMIT = 80000

NO_SHIP = 0
HAS_SHIP = 1

NO_SHOT = 0
SHOT_MISS = 1
SHOT_HIT = 2

DISP_RADIUS = 4

SHIP_ORDER = ["square2", "L3", "line3", "line2"]


class Theme:
    """Centralized colors used across the UI."""

    # Backgrounds
    BG_DARK = "#020617"  # slate-950
    BG_PANEL = "#0f172a"  # slate-900
    BG_BUTTON = "#1f2937"  # gray-800

    # Generic text
    TEXT_MAIN = "#e5e7eb"  # gray-200
    TEXT_LABEL = "#9ca3af"  # gray-400
    # Secondary label color used by some dialogs/widgets.
    # Alias it to TEXT_LABEL so any older UI code referencing TEXT_MUTED won't crash.
    TEXT_MUTED = TEXT_LABEL
    TEXT_DARK = "#000000"

    # Empty-cell border
    BORDER_EMPTY = "#1f2937"

    # Miss styling
    MISS_BG = "#1e3a8a"
    MISS_TEXT = "#bfdbfe"
    MISS_BORDER = "#2563eb"

    # Hit styling
    HIT_BG = "#7f1d1d"
    HIT_TEXT = "#fecaca"
    HIT_BORDER = "#f97373"

    # Assigned-hit border (per-ship dashed border)
    ASSIGNED_BORDER = "#22c55e"

    # Ship layout cells (defense tab)
    LAYOUT_SHIP_BG = "#064e3b"
    LAYOUT_SHIP_TEXT = "#a7f3d0"
    LAYOUT_SHIP_BORDER = "#10b981"

    # Heatmap overlay (defense tab)
    HEAT_TEXT = "#93c5fd"
    HEAT_BORDER = "#1d4ed8"

    # Status label colors
    STATUS_SUNK = "#f97373"
    STATUS_SUNK_MAYBE = "#fb923c"
    STATUS_AFLOAT = "#38bdf8"
    STATUS_MAYBE = "#facc15"

    # Links / highlights
    LINK = "#38bdf8"
    HIGHLIGHT = "#0ea5e9"
    BORDER_BEST = "#38bdf8"


# Configuration for tunable models (used by the Model Details / custom sim dialog)
PARAM_SPECS = {
    # Softmax temperature T appears in exp((p - pmax) / T).
    # Since p ∈ [0, 1], values above ~1 quickly approach near-uniform sampling,
    # so we cap sweeps to [0.01, 1.00] to keep the grid meaningful.
    "softmax_greedy": [
        {"key": "temperature", "label": "Temperature (T)", "default": 0.10, "min": 0.01, "max": 1.00, "step": 0.01}
    ],

    # Endpoint Targeter weights are linear multipliers; extreme values just drown out the others.
    # A practical, interpretable range is [0, 2] for each term.
    "endpoint_phase": [
        {"key": "w_prob", "label": "Prob Weight", "default": 1.00, "min": 0.00, "max": 2.00, "step": 0.05},
        {"key": "w_neighbor", "label": "Cluster Bonus", "default": 0.20, "min": 0.00, "max": 2.00, "step": 0.05},
        {"key": "w_endpoint", "label": "Endpoint Bonus", "default": 0.40, "min": 0.00, "max": 2.00, "step": 0.05},
    ],
}


# ------------------------------------------------------------
# Global stats tracker for games and win rate
# ------------------------------------------------------------
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


# ------------------------------------------------------------


@dataclass(frozen=True)
class Placement:
    ship: str
    cells: Tuple[Tuple[int, int], ...]
    mask: int


def cell_index(r: int, c: int) -> int:
    return r * BOARD_SIZE + c


def make_mask(cells: List[Tuple[int, int]]) -> int:
    m = 0
    for r, c in cells:
        m |= 1 << cell_index(r, c)
    return m


def generate_ship_placements() -> Dict[str, List[Placement]]:
    placements: Dict[str, List[Placement]] = {}

    # 2x2 square
    sq_list: List[Placement] = []
    for r in range(BOARD_SIZE - 1):
        for c in range(BOARD_SIZE - 1):
            cells = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
            sq_list.append(Placement("square2", tuple(cells), make_mask(cells)))
    placements["square2"] = sq_list

    # L-tromino: base shape (0,0),(0,1),(1,0)
    base = [(0, 0), (0, 1), (1, 0)]
    oriented_shapes: List[List[Tuple[int, int]]] = []
    seen = set()
    for rot in range(4):
        coords = []
        for x, y in base:
            if rot == 0:
                xr, yr = x, y
            elif rot == 1:
                xr, yr = -y, x
            elif rot == 2:
                xr, yr = -x, -y
            else:
                xr, yr = y, -x
            coords.append((xr, yr))
        min_r = min(r for r, _ in coords)
        min_c = min(c for _, c in coords)
        norm = [(r - min_r, c - min_c) for r, c in coords]
        key = tuple(sorted(norm))
        if key not in seen:
            seen.add(key)
            oriented_shapes.append(norm)
    L_list: List[Placement] = []
    for shape in oriented_shapes:
        max_dr = max(r for r, _ in shape)
        max_dc = max(c for _, c in shape)
        for r0 in range(BOARD_SIZE - max_dr):
            for c0 in range(BOARD_SIZE - max_dc):
                cells = [(r0 + r, c0 + c) for r, c in shape]
                L_list.append(Placement("L3", tuple(cells), make_mask(cells)))
    placements["L3"] = L_list

    # 3-long line
    line3: List[Placement] = []
    # horizontal
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE - 2):
            cells = [(r, c), (r, c + 1), (r, c + 2)]
            line3.append(Placement("line3", tuple(cells), make_mask(cells)))
    # vertical
    for r in range(BOARD_SIZE - 2):
        for c in range(BOARD_SIZE):
            cells = [(r, c), (r + 1, c), (r + 2, c)]
            line3.append(Placement("line3", tuple(cells), make_mask(cells)))
    placements["line3"] = line3

    # 2-long line
    line2: List[Placement] = []
    # horizontal
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE - 1):
            cells = [(r, c), (r, c + 1)]
            line2.append(Placement("line2", tuple(cells), make_mask(cells)))
    # vertical
    for r in range(BOARD_SIZE - 1):
        for c in range(BOARD_SIZE):
            cells = [(r, c), (r + 1, c)]
            line2.append(Placement("line2", tuple(cells), make_mask(cells)))
    placements["line2"] = line2

    return placements


PLACEMENTS = generate_ship_placements()


def create_board() -> List[List[str]]:
    return [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]


def board_masks(board: List[List[str]]) -> Tuple[int, int]:
    hit_mask = 0
    miss_mask = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            idx = cell_index(r, c)
            if board[r][c] == HIT:
                hit_mask |= 1 << idx
            elif board[r][c] == MISS:
                miss_mask |= 1 << idx
    return hit_mask, miss_mask


def filter_allowed_placements(
        placements: Dict[str, List[Placement]],
        hit_mask: int,
        miss_mask: int,
        confirmed_sunk: Set[str],
        assigned_hits: Dict[str, Set[Tuple[int, int]]],
) -> Dict[str, List[Placement]]:
    filtered: Dict[str, List[Placement]] = {}
    for ship, plist in placements.items():
        required_cells = assigned_hits.get(ship, set())
        required_mask = make_mask(list(required_cells)) if required_cells else 0
        is_sunk = ship in confirmed_sunk
        new_list: List[Placement] = []
        for p in plist:
            # cannot overlap misses
            if p.mask & miss_mask:
                continue
            # must cover all assigned hits for this ship
            if required_mask and (required_mask & ~p.mask) != 0:
                continue
            # if user marked ship sunk, placement must lie entirely on hits
            if is_sunk and (p.mask & ~hit_mask) != 0:
                continue
            new_list.append(p)
        filtered[ship] = new_list
    return filtered


def random_world(
        allowed: Dict[str, List[Placement]],
        hit_mask: int,
        rng: random.Random,
) -> Optional[Tuple[int, Tuple[int, ...]]]:
    used_mask = 0
    ship_masks: List[int] = []
    ship_sequence = SHIP_ORDER[:]  # copy
    rng.shuffle(ship_sequence)
    for ship in ship_sequence:
        options = allowed[ship]
        if not options:
            return None
        placed = False
        for _ in range(50):
            p = rng.choice(options)
            if p.mask & used_mask:
                continue
            used_mask |= p.mask
            ship_masks.append(p.mask)
            placed = True
            break
        if not placed:
            return None
    union_mask = used_mask
    if (union_mask & hit_mask) != hit_mask:
        return None
    return union_mask, tuple(ship_masks)


def enumerate_worlds(
        allowed: Dict[str, List[Placement]],
        hit_mask: int,
        max_worlds: int,
) -> Tuple[List[int], List[Tuple[int, ...]]]:
    worlds_union: List[int] = []
    worlds_ship_masks: List[Tuple[int, ...]] = []

    def backtrack(i: int, used_mask: int, ship_masks: List[int]):
        if len(worlds_union) >= max_worlds:
            return
        if i == len(SHIP_ORDER):
            union_mask = used_mask
            if (union_mask & hit_mask) != hit_mask:
                return
            worlds_union.append(union_mask)
            worlds_ship_masks.append(tuple(ship_masks))
            return
        ship = SHIP_ORDER[i]
        for p in allowed[ship]:
            if p.mask & used_mask:
                continue
            ship_masks.append(p.mask)
            backtrack(i + 1, used_mask | p.mask, ship_masks)
            ship_masks.pop()

    backtrack(0, 0, [])
    return worlds_union, worlds_ship_masks


def sample_worlds(
        board: List[List[str]],
        placements: Dict[str, List[Placement]],
        confirmed_sunk: Set[str],
        assigned_hits: Dict[str, Set[Tuple[int, int]]],
        rng_seed: Optional[int] = None,
) -> Tuple[List[int], List[int], Dict[str, float], int]:
    """
    Return:
      - list of union masks for each world
      - per-cell hit counts
      - per-ship sunk probabilities
      - number of worlds sampled
    """

    hit_mask, miss_mask = board_masks(board)
    allowed = filter_allowed_placements(
        placements, hit_mask, miss_mask, confirmed_sunk, assigned_hits
    )

    # If any ship has no legal placement, there are no consistent worlds.
    for ship in SHIP_ORDER:
        if not allowed[ship]:
            return [], [0] * (BOARD_SIZE * BOARD_SIZE), {
                s: 0.0 for s in SHIP_ORDER
            }, 0

    # Decide enumeration vs Monte Carlo
    remaining_ships = [s for s in SHIP_ORDER if s not in confirmed_sunk]
    force_enumeration = (len(remaining_ships) == 1)

    product = 1
    enumeration = True
    if not force_enumeration:
        for ship in SHIP_ORDER:
            n = len(allowed[ship])
            product *= n
            if product > ENUMERATION_PRODUCT_LIMIT:
                enumeration = False
                break
    else:
        enumeration = True

    worlds_union: List[int] = []
    worlds_ship_masks: List[Tuple[int, ...]] = []
    seen: Set[int] = set()

    if enumeration:
        # Exact enumeration (capped at ENUMERATION_PRODUCT_LIMIT worlds)
        worlds_union, worlds_ship_masks = enumerate_worlds(
            allowed, hit_mask, ENUMERATION_PRODUCT_LIMIT
        )
    else:
        # Monte Carlo sampling of distinct worlds
        rng = random.Random(rng_seed)
        attempts = 0
        max_attempts = WORLD_SAMPLE_TARGET * WORLD_MAX_ATTEMPTS_FACTOR

        while len(worlds_union) < WORLD_SAMPLE_TARGET and attempts < max_attempts:
            attempts += 1
            res = random_world(allowed, hit_mask, rng)
            if res is None:
                continue
            union_mask, ship_masks_tuple = res
            if union_mask in seen:
                continue
            seen.add(union_mask)
            worlds_union.append(union_mask)
            worlds_ship_masks.append(ship_masks_tuple)

    # ---------- NEW: enforce sunk / not-sunk consistency ----------
    # For each world:
    #  - If a ship is marked sunk, all its cells must be hits.
    #  - If a ship is NOT marked sunk, it must have at least one non-hit cell.
    filtered_union: List[int] = []
    filtered_ship_masks: List[Tuple[int, ...]] = []

    for w_idx in range(len(worlds_union)):
        union_mask = worlds_union[w_idx]
        ship_masks_tuple = worlds_ship_masks[w_idx]

        ok = True
        for i, ship in enumerate(SHIP_ORDER):
            m_ship = ship_masks_tuple[i]
            is_sunk = ship in confirmed_sunk

            if is_sunk:
                # Marked sunk: cannot extend into any non-hit cell
                if (m_ship & ~hit_mask) != 0:
                    ok = False
                    break
            else:
                # Not marked sunk: must still have at least one non-hit cell
                if (m_ship & ~hit_mask) == 0:
                    ok = False
                    break

        if ok:
            filtered_union.append(union_mask)
            filtered_ship_masks.append(ship_masks_tuple)

    worlds_union = filtered_union
    worlds_ship_masks = filtered_ship_masks
    # --------------------------------------------------------------

    N = len(worlds_union)
    if N == 0:
        # No worlds consistent with hits/misses + sunk checkboxes
        return [], [0] * (BOARD_SIZE * BOARD_SIZE), {
            s: 0.0 for s in SHIP_ORDER
        }, 0

    # Per-cell hit counts
    cell_hit_counts = [0] * (BOARD_SIZE * BOARD_SIZE)
    ship_sunk_counts = {s: 0 for s in SHIP_ORDER}

    for w_idx in range(N):
        union_mask = worlds_union[w_idx]

        # Cell hits
        m = union_mask
        idx = 0
        while idx < BOARD_SIZE * BOARD_SIZE:
            if m & 1:
                cell_hit_counts[idx] += 1
            m >>= 1
            idx += 1

        # Per-ship sunk: in this world, a ship is sunk if all its cells are hits
        ship_masks_tuple = worlds_ship_masks[w_idx]
        for i, ship in enumerate(SHIP_ORDER):
            m_ship = ship_masks_tuple[i]
            if (m_ship & ~hit_mask) == 0:
                ship_sunk_counts[ship] += 1

    ship_sunk_probs: Dict[str, float] = {}
    for ship in SHIP_ORDER:
        if ship in confirmed_sunk:
            ship_sunk_probs[ship] = 1.0
        else:
            ship_sunk_probs[ship] = ship_sunk_counts[ship] / N

    return worlds_union, cell_hit_counts, ship_sunk_probs, N


def compute_cell_hit_counts_from_worlds(world_masks: List[int]) -> List[int]:
    counts = [0] * (BOARD_SIZE * BOARD_SIZE)
    for wm in world_masks:
        m = wm
        idx = 0
        while idx < BOARD_SIZE * BOARD_SIZE:
            if m & 1:
                counts[idx] += 1
            m >>= 1
            idx += 1
    return counts


def compute_min_expected_worlds_after_one_shot(
        world_masks: List[int],
        known_mask: int,
) -> float:
    """
    For a given set of consistent worlds and current known_mask, compute
    the minimum expected *log* world count after one additional optimal shot.

    This is an information-theoretic refinement of the previous heuristic:
    instead of minimizing E[N_after], we minimize E[log N_after],
    which is proportional to expected posterior entropy.
    """
    N = len(world_masks)
    if N == 0:
        return 0.0

    cell_hit_counts = compute_cell_hit_counts_from_worlds(world_masks)
    best_E = float("inf")

    for idx in range(BOARD_SIZE * BOARD_SIZE):
        # Skip cells that are already known (hit or miss).
        if (known_mask >> idx) & 1:
            continue

        n_hit = cell_hit_counts[idx]
        n_miss = N - n_hit

        if n_hit == 0 or n_miss == 0:
            # Outcome of this shot is effectively deterministic;
            # we don't shrink the set of plausible layouts at all.
            # Expected log(worlds) stays at log(N).
            E = math.log(N)
        else:
            p_hit = n_hit / N
            p_miss = n_miss / N
            # Expected log of remaining world count after this shot:
            # branch sizes are n_hit or n_miss.
            E = p_hit * math.log(n_hit) + p_miss * math.log(n_miss)

        if E < best_E:
            best_E = E

    if best_E == float("inf"):
        return math.log(N)
    return best_E


def two_ply_selection(
        board: List[List[str]],
        world_masks: List[int],
        cell_hit_counts: List[int],
) -> Tuple[List[float], List[int], float]:
    """
    Shared 2-ply heuristic used by the Attack tab and the model simulator.

    Returns:
        info_gain_values: list[float] per cell, normalized 0..1 (same as Attack overlay).
        best_indices: list[int] of cell indices (0..BOARD_SIZE*BOARD_SIZE-1) tied for best 2-ply score.
        best_p_hit: hit probability of those best cells.
    """
    N = len(world_masks)
    total_cells = BOARD_SIZE * BOARD_SIZE

    if N == 0:
        return [0.0] * total_cells, [], 0.0

    # Build known mask from current board
    known_mask = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != EMPTY:
                known_mask |= 1 << cell_index(r, c)

    best_score1: Optional[float] = None
    scores1: List[Optional[float]] = [None] * total_cells
    info_values: List[float] = [0.0] * total_cells
    candidates: List[int] = []

    # 1-ply split score (same as in Attack tab): smaller is better
    for idx in range(total_cells):
        if (known_mask >> idx) & 1:
            continue
        n_hit = cell_hit_counts[idx]
        n_miss = N - n_hit
        score = (n_hit * n_hit + n_miss * n_miss) / N
        scores1[idx] = score
        candidates.append(idx)
        if best_score1 is None or score < best_score1:
            best_score1 = score

    if best_score1 is None:
        return [0.0] * total_cells, [], 0.0

    # Info gain normalization (same overlay as Attack tab)
    max_gain = 0.0
    for idx in candidates:
        gain = N - scores1[idx]
        info_values[idx] = gain
        if gain > max_gain:
            max_gain = gain
    if max_gain > 0:
        for idx in candidates:
            info_values[idx] /= max_gain

    # 2-ply expected log-worlds criterion
    candidates.sort(key=lambda i: scores1[i])  # best 1-ply splits first
    TOP_K = min(24, len(candidates))
    top_candidates = candidates[:TOP_K]

    best_two_ply: Optional[float] = None
    best_indices: List[int] = []
    best_p = 0.0

    for idx in top_candidates:
        bit = 1 << idx
        worlds_hit = [wm for wm in world_masks if wm & bit]
        worlds_miss = [wm for wm in world_masks if not (wm & bit)]
        Nh = len(worlds_hit)
        Nm = len(worlds_miss)
        if Nh + Nm == 0:
            continue
        p_hit = Nh / (Nh + Nm)

        known_after = known_mask | bit
        Eh = compute_min_expected_worlds_after_one_shot(worlds_hit, known_after) if Nh > 0 else 0.0
        Em = compute_min_expected_worlds_after_one_shot(worlds_miss, known_after) if Nm > 0 else 0.0
        two_ply = p_hit * Eh + (1.0 - p_hit) * Em

        if best_two_ply is None or two_ply < best_two_ply - 1e-9:
            best_two_ply = two_ply
            best_indices = [idx]
            best_p = p_hit
        elif abs(two_ply - best_two_ply) <= 1e-9:
            if p_hit > best_p + 1e-9:
                best_indices = [idx]
                best_p = p_hit
            elif abs(p_hit - best_p) <= 1e-9:
                best_indices.append(idx)

    return info_values, best_indices, best_p


def _choose_next_shot_for_strategy(
        strategy: str,
        board: List[List[str]],
        placements: Dict[str, List[Placement]],
        rng: random.Random,
        known_sunk: Optional[Set[str]] = None,
        known_assigned: Optional[Dict[str, Set[Tuple[int, int]]]] = None,
        params: Optional[Dict[str, float]] = None,
) -> Tuple[int, int]:
    """
    Choose the next shot for the given strategy using the current board state.

    Notes:
    - `known_sunk` and `known_assigned` are carried across turns by the simulator.
    - `params` is optional and is used for tunable strategies (e.g., softmax temperature).
    """
    if params is None:
        params = {}

    unknown_cells = [
        (r, c)
        for r in range(BOARD_SIZE)
        for c in range(BOARD_SIZE)
        if board[r][c] == EMPTY
    ]
    if not unknown_cells:
        return 0, 0

    if strategy == "random":
        return rng.choice(unknown_cells)

    # 1) Build World Model
    confirmed_sunk = known_sunk if known_sunk is not None else set()
    assigned_hits = known_assigned if known_assigned is not None else {s: set() for s in SHIP_ORDER}

    worlds_union, cell_hit_counts, _, N = sample_worlds(
        board,
        placements,
        confirmed_sunk,
        assigned_hits,
        rng_seed=rng.randint(0, 2 ** 31 - 1),
    )

    if N <= 0:
        return rng.choice(unknown_cells)

    cell_probs = [cnt / N for cnt in cell_hit_counts]

    # --- TARGET MODE LOGIC (Refined) ---
    # Only enter target mode if we have *any* hit on the board and the posterior is confident.
    has_any_hit = any(board[r][c] == HIT for r in range(BOARD_SIZE) for c in range(BOARD_SIZE))
    max_p = max(cell_probs[cell_index(r, c)] for r, c in unknown_cells) if unknown_cells else 0.0
    is_target_mode = has_any_hit and (max_p > 0.30)

    # --- ADVANCED STRATEGIES ---

    if strategy == "endpoint_phase":
        # Only use geometry in true target mode; otherwise hunt with entropy.
        if not is_target_mode:
            strategy = "entropy1"
        else:
            hits_all = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if board[r][c] == HIT]
            hit_set = set(hits_all)

            def neighbors4(rr: int, cc: int):
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                        yield nr, nc

            # Connected components of HITs
            components: List[List[Tuple[int, int]]] = []
            visited: Set[Tuple[int, int]] = set()
            for start in hits_all:
                if start in visited:
                    continue
                stack = [start]
                visited.add(start)
                comp: List[Tuple[int, int]] = []
                while stack:
                    cur = stack.pop()
                    comp.append(cur)
                    for nr, nc in neighbors4(cur[0], cur[1]):
                        if (nr, nc) in hit_set and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            stack.append((nr, nc))
                components.append(comp)

            best_local_score = -float("inf")
            best_local_cells: List[Tuple[int, int]] = []

            # Tunable weights
            w_prob = float(params.get("w_prob", 1.0))
            w_neighbor = float(params.get("w_neighbor", 0.2))
            w_endpoint = float(params.get("w_endpoint", 0.4))

            for comp in components:
                # Frontier = unknown neighbors of the component
                frontier: Set[Tuple[int, int]] = set()
                for hr, hc in comp:
                    for nr, nc in neighbors4(hr, hc):
                        if board[nr][nc] == EMPTY:
                            frontier.add((nr, nc))
                if not frontier:
                    continue

                # Alignment detection: if hits form a line, favor endpoints.
                aligned_row = (len(comp) >= 2) and all(r == comp[0][0] for r, _ in comp)
                aligned_col = (len(comp) >= 2) and all(c == comp[0][1] for _, c in comp)

                endpoint_cells: Set[Tuple[int, int]] = set()
                if aligned_row:
                    r0 = comp[0][0]
                    cols = [c for _, c in comp]
                    for cand_c in (min(cols) - 1, max(cols) + 1):
                        if 0 <= cand_c < BOARD_SIZE and board[r0][cand_c] == EMPTY:
                            endpoint_cells.add((r0, cand_c))
                elif aligned_col:
                    c0 = comp[0][1]
                    rows = [r for r, _ in comp]
                    for cand_r in (min(rows) - 1, max(rows) + 1):
                        if 0 <= cand_r < BOARD_SIZE and board[cand_r][c0] == EMPTY:
                            endpoint_cells.add((cand_r, c0))

                for r, c in frontier:
                    idx = cell_index(r, c)
                    p = cell_probs[idx]

                    # Neighbor bonus: prefer cells adjacent to multiple hits (clusters)
                    hit_neighbors = 0
                    for nr, nc in neighbors4(r, c):
                        if board[nr][nc] == HIT:
                            hit_neighbors += 1

                    is_endpoint = 1.0 if (r, c) in endpoint_cells else 0.0
                    score = (w_prob * p) + (w_neighbor * hit_neighbors) + (w_endpoint * is_endpoint)

                    if score > best_local_score + 1e-12:
                        best_local_score = score
                        best_local_cells = [(r, c)]
                    elif abs(score - best_local_score) <= 1e-12:
                        best_local_cells.append((r, c))

            if best_local_cells:
                return rng.choice(best_local_cells)

            # If geometry doesn't find anything useful, fall back to greedy.
            strategy = "greedy"

    if strategy == "thompson_world":
        if is_target_mode:
            strategy = "greedy"
        else:
            if worlds_union:
                chosen_mask = rng.choice(worlds_union)
                targets = [
                    (r, c)
                    for (r, c) in unknown_cells
                    if (chosen_mask >> cell_index(r, c)) & 1
                ]
                if targets:
                    return rng.choice(targets)
            return rng.choice(unknown_cells)

    if strategy == "dynamic_parity":
        if is_target_mode:
            strategy = "greedy"
        else:
            alive = [s for s in SHIP_ORDER if s not in confirmed_sunk]
            step = 2
            # If the only ship left is size-3, a 3-color parity is sufficient.
            if len(alive) == 1 and alive[0] == "line3":
                step = 3

            best_cells: List[Tuple[int, int]] = []
            best_mass = -1.0
            for color in range(step):
                cells = [(r, c) for (r, c) in unknown_cells if (r + c) % step == color]
                if not cells:
                    continue
                mass = sum(cell_probs[cell_index(r, c)] for r, c in cells)
                if mass > best_mass:
                    best_mass = mass
                    best_cells = cells

            if best_cells:
                return max(best_cells, key=lambda rc: cell_probs[cell_index(rc[0], rc[1])])
            return rng.choice(unknown_cells)

    if strategy == "softmax_greedy":
        if is_target_mode:
            strategy = "greedy"
        else:
            T = float(params.get("temperature", 0.10))
            # Filter based on non-zero counts for robustness.
            candidates = [(r, c) for (r, c) in unknown_cells if cell_hit_counts[cell_index(r, c)] > 0]
            if not candidates:
                return rng.choice(unknown_cells)

            ps = [cell_probs[cell_index(r, c)] for r, c in candidates]
            pmax = max(ps)
            denom = max(1e-6, T)
            weights = [math.exp((p - pmax) / denom) for p in ps]
            return rng.choices(candidates, weights=weights, k=1)[0]

    # --- STANDARD STRATEGIES ---

    if strategy == "hybrid_phase":
        strategy = "greedy" if is_target_mode else "entropy1"

    if strategy == "weighted_sample":
        candidates = []
        weights = []
        for r, c in unknown_cells:
            p = cell_probs[cell_index(r, c)]
            if p > 0:
                candidates.append((r, c))
                weights.append(p)
        if not candidates:
            return rng.choice(unknown_cells)
        return rng.choices(candidates, weights=weights, k=1)[0]

    if strategy == "random_checkerboard":
        if is_target_mode:
            strategy = "greedy"
        else:
            whites = [(r, c) for (r, c) in unknown_cells if (r + c) % 2 == 0]
            return rng.choice(whites) if whites else rng.choice(unknown_cells)

    if strategy == "systematic_checkerboard":
        if is_target_mode:
            strategy = "greedy"
        else:
            sorted_cells = sorted(unknown_cells, key=lambda p: (p[0], p[1]))
            whites = [p for p in sorted_cells if (p[0] + p[1]) % 2 == 0]
            return whites[0] if whites else sorted_cells[0]

    if strategy == "diagonal_stripe":
        if is_target_mode:
            strategy = "greedy"
        else:
            diagonals = [p for p in unknown_cells if (p[0] - p[1]) % 4 == 0]
            if diagonals:
                return rng.choice(diagonals)
            secondary = [p for p in unknown_cells if (p[0] - p[1]) % 2 == 0]
            if secondary:
                return rng.choice(secondary)
            return rng.choice(unknown_cells)

    if strategy == "two_ply":
        _, best_indices, _ = two_ply_selection(board, worlds_union, cell_hit_counts)
        if best_indices:
            idx = rng.choice(best_indices)
            return divmod(idx, BOARD_SIZE)
        strategy = "greedy"

    # --- SCORING FUNCTIONS ---

    def score_greedy(r: int, c: int) -> float:
        return cell_probs[cell_index(r, c)]

    def score_entropy(r: int, c: int) -> float:
        idx = cell_index(r, c)
        n_hit = cell_hit_counts[idx]
        n_miss = N - n_hit
        if n_hit == 0 or n_miss == 0:
            return -1.0e9
        p_hit = n_hit / N
        p_miss = 1.0 - p_hit
        # Keep the existing "count-log" proxy but guard edge cases.
        E = p_hit * math.log(max(1, n_hit)) + p_miss * math.log(max(1, n_miss))
        return -E

    def score_adaptive_skew(r: int, c: int) -> float:
        base_score = cell_probs[cell_index(r, c)]
        center = (BOARD_SIZE - 1) / 2.0
        dist = math.sqrt((r - center) ** 2 + (c - center) ** 2)
        norm_dist = dist / math.sqrt(2 * center ** 2)
        unknown_ratio = len(unknown_cells) / (BOARD_SIZE * BOARD_SIZE)
        penalty = 0.0
        if unknown_ratio > 0.5:
            penalty = 0.20 * norm_dist
        return base_score - penalty

    def score_center_weighted(r: int, c: int) -> float:
        center = (BOARD_SIZE - 1) / 2.0
        dist2 = (r - center) ** 2 + (c - center) ** 2
        return cell_probs[cell_index(r, c)] / (1.0 + 0.25 * dist2)

    prefer_even = False
    if strategy == "parity_greedy":
        evens = [(rr, cc) for (rr, cc) in unknown_cells if (rr + cc) % 2 == 0]
        odds = [(rr, cc) for (rr, cc) in unknown_cells if (rr + cc) % 2 == 1]
        even_mass = sum(cell_probs[cell_index(rr, cc)] for rr, cc in evens)
        odd_mass = sum(cell_probs[cell_index(rr, cc)] for rr, cc in odds)
        prefer_even = even_mass >= odd_mass

    def score_parity_greedy(r: int, c: int) -> float:
        is_even = (r + c) % 2 == 0
        if prefer_even != is_even:
            return -1.0e9
        return cell_probs[cell_index(r, c)]

    # --- EXECUTE ---
    scorers = {
        "greedy": score_greedy,
        "entropy1": score_entropy,
        "parity_greedy": score_parity_greedy,
        "center_weighted": score_center_weighted,
        "adaptive_skew": score_adaptive_skew,
    }
    scorer = scorers.get(strategy, score_greedy)

    best_score = -float("inf")
    best_candidates: List[Tuple[int, int]] = []
    for r, c in unknown_cells:
        s = scorer(r, c)
        if s > best_score + 1e-12:
            best_score = s
            best_candidates = [(r, c)]
        elif abs(s - best_score) <= 1e-12:
            best_candidates.append((r, c))

    return rng.choice(best_candidates)


def simulate_model_game(
        strategy: str,
        placements: Dict[str, List[Placement]],
        rng: Optional[random.Random] = None,
        params: Optional[Dict[str, float]] = None,
) -> int:
    if rng is None:
        rng = random.Random()

    # 1. Generate a "True World" with full ship details
    # We need to know exactly where every ship is to simulate "Sunk" messages.
    true_layout: Dict[str, Placement] = {}
    used_mask = 0

    # Retry loop to ensure valid board (simple rejection sampling)
    while len(true_layout) < len(SHIP_ORDER):
        true_layout = {}
        used_mask = 0
        valid_board = True

        # Randomize order to prevent bias
        shuffled_ships = list(SHIP_ORDER)
        rng.shuffle(shuffled_ships)

        for ship in shuffled_ships:
            options = placements[ship]
            # Try 50 times to place this ship
            placed = False
            for _ in range(50):
                p = rng.choice(options)
                if not (p.mask & used_mask):
                    true_layout[ship] = p
                    used_mask |= p.mask
                    placed = True
                    break
            if not placed:
                valid_board = False
                break

        if not valid_board:
            continue  # Try generating the whole board again

    # 2. Setup Game State
    board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    confirmed_sunk = set()
    assigned_hits = {s: set() for s in SHIP_ORDER}

    # Track which cells belong to which ship for fast lookup
    cell_to_ship = {}
    for ship, p in true_layout.items():
        for r, c in p.cells:
            cell_to_ship[(r, c)] = ship

    shots = 0
    total_cells = BOARD_SIZE * BOARD_SIZE
    ships_remaining = len(SHIP_ORDER)

    # 3. Game Loop
    while ships_remaining > 0 and shots < total_cells:
        # Pass our "God Mode" knowledge into the bot
        r, c = _choose_next_shot_for_strategy(
            strategy,
            board,
            placements,
            rng,
            known_sunk=confirmed_sunk,
            known_assigned=assigned_hits,
            params=params,
        )

        idx = cell_index(r, c)
        is_hit = (used_mask >> idx) & 1

        board[r][c] = HIT if is_hit else MISS
        shots += 1

        if is_hit:
            # Identify which ship was hit
            hit_ship = cell_to_ship[(r, c)]

            # FEATURE 1: Auto-Assign Hits (Perfect Play)
            # The bot "knows" which ship it hit.
            assigned_hits[hit_ship].add((r, c))

            # FEATURE 2: Auto-Mark Sunk (Standard Rules)
            # Check if this ship is now fully sunk
            ship_cells = true_layout[hit_ship].cells
            is_sunk = all(board[sr][sc] == HIT for sr, sc in ship_cells)

            if is_sunk and hit_ship not in confirmed_sunk:
                confirmed_sunk.add(hit_ship)
                ships_remaining -= 1

    return shots


# Defense helpers
def build_base_heat(hit_counts, miss_counts):
    heat = [[0.0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    total = 0.0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            v = 1.0 + 2.0 * hit_counts[r][c] - 0.5 * miss_counts[r][c]
            if v < 0.1:
                v = 0.1
            heat[r][c] = v
            total += v
    if total <= 0:
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                heat[r][c] = 1.0 / (BOARD_SIZE * BOARD_SIZE)
        return heat
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            heat[r][c] /= total
    return heat


def weighted_random_choice(indices, weights, rng: random.Random) -> int:
    assert len(indices) == len(weights)
    if not indices:
        raise ValueError("no indices")
    total = sum(weights)
    if total <= 1e-12:
        return rng.choice(indices)
    x = rng.random() * total
    acc = 0.0
    for idx, w in zip(indices, weights):
        acc += w
        if x <= acc:
            return idx
    return indices[-1]


def compute_clusters_mask(layout_mask: int) -> Tuple[List[int], List[int]]:
    visited = [False] * (BOARD_SIZE * BOARD_SIZE)
    owner = [-1] * (BOARD_SIZE * BOARD_SIZE)
    clusters: List[int] = []
    for idx in range(BOARD_SIZE * BOARD_SIZE):
        bit = 1 << idx
        if not (layout_mask & bit) or visited[idx]:
            continue
        stack = [idx]
        visited[idx] = True
        cluster_mask = 0
        while stack:
            cur = stack.pop()
            cluster_mask |= 1 << cur
            owner[cur] = len(clusters)
            r = cur // BOARD_SIZE
            c = cur % BOARD_SIZE
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE:
                    ni = rr * BOARD_SIZE + cc
                    if (layout_mask & (1 << ni)) and not visited[ni]:
                        visited[ni] = True
                        stack.append(ni)
        clusters.append(cluster_mask)
    return clusters, owner


def simulate_enemy_game_phase(
        layout_mask: int,
        base_heat_phase,
        disp_counts,
        model_type: str,
        rng: random.Random,
        max_shots: int = 64,
        initial_shot_board: Optional[List[List[int]]] = None
) -> Tuple[int, int]:
    shot = [False] * (BOARD_SIZE * BOARD_SIZE)
    remaining_mask = layout_mask
    hits = 0
    shots_taken = 0

    if initial_shot_board:
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                s = initial_shot_board[r][c]
                if s != NO_SHOT:
                    idx = r * BOARD_SIZE + c
                    shot[idx] = True
                    shots_taken += 1
                    if (layout_mask >> idx) & 1:
                        hits += 1
                        remaining_mask &= ~(1 << idx)
    frontier: Set[int] = set()

    cluster_masks, owner = compute_clusters_mask(layout_mask)
    cluster_remaining = list(cluster_masks)
    sunk_count = 0

    last_idx: Optional[int] = None
    last_was_hit: bool = False

    disp_sums = [
        [
            sum(
                disp_counts[phase][hm][dr][dc]
                for dr in range(2 * DISP_RADIUS + 1)
                for dc in range(2 * DISP_RADIUS + 1)
            )
            for hm in range(2)
        ]
        for phase in range(4)
    ]

    def neighbours(idx: int):
        r = idx // BOARD_SIZE
        c = idx % BOARD_SIZE
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE:
                yield rr * BOARD_SIZE + cc

    def seq_weight(base_w: float, phase_idx: int, last_idx_local, last_was_hit_local, candidate_idx: int) -> float:
        if model_type != "seq":
            return base_w
        if last_idx_local is None:
            return base_w
        lr = last_idx_local // BOARD_SIZE
        lc = last_idx_local % BOARD_SIZE
        r = candidate_idx // BOARD_SIZE
        c = candidate_idx % BOARD_SIZE
        dr = r - lr
        dc = c - lc
        if abs(dr) > DISP_RADIUS or abs(dc) > DISP_RADIUS:
            return base_w
        hm_idx = 1 if last_was_hit_local else 0
        count = disp_counts[phase_idx][hm_idx][dr + DISP_RADIUS][dc + DISP_RADIUS]
        total = disp_sums[phase_idx][hm_idx]
        if total <= 0:
            return base_w
        alpha = 3.0
        factor = 1.0 + alpha * (count / total)
        return base_w * factor

    while shots_taken < max_shots and remaining_mask != 0:
        phase_idx = sunk_count
        if phase_idx > 3:
            phase_idx = 3
        heat = base_heat_phase[phase_idx]

        if frontier:
            choices = [idx for idx in frontier if not shot[idx]]
            if not choices:
                frontier.clear()
        if frontier:
            weights = []
            for idx in choices:
                r = idx // BOARD_SIZE
                c = idx % BOARD_SIZE
                base_w = heat[r][c]
                w = seq_weight(base_w, phase_idx, last_idx, last_was_hit, idx)
                weights.append(w)
            chosen = weighted_random_choice(choices, weights, rng)
            frontier.discard(chosen)
        else:
            choices = [idx for idx in range(BOARD_SIZE * BOARD_SIZE) if not shot[idx]]
            if not choices:
                break
            weights = []
            for idx in choices:
                r = idx // BOARD_SIZE
                c = idx % BOARD_SIZE
                base_w = heat[r][c]
                w = seq_weight(base_w, phase_idx, last_idx, last_was_hit, idx)
                weights.append(w)
            chosen = weighted_random_choice(choices, weights, rng)

        shot[chosen] = True
        shots_taken += 1
        bit = 1 << chosen
        if layout_mask & bit:
            hits += 1
            remaining_mask &= ~bit
            last_was_hit = True
            cid = owner[chosen]
            if cid != -1:
                cluster_remaining[cid] &= ~bit
                if cluster_remaining[cid] == 0:
                    sunk_count += 1
            for nb in neighbours(chosen):
                if not shot[nb]:
                    frontier.add(nb)
        else:
            last_was_hit = False
        last_idx = chosen

    return shots_taken, hits


def recommend_layout_phase(
        hit_counts_phase,
        miss_counts_phase,
        disp_counts,
        placements,
        n_iter: int = 250,
        sim_games_per_layout: int = 3,
        rng_seed: Optional[int] = None,
):
    rng = random.Random(rng_seed)

    base_heat_phase = []
    for p in range(4):
        base_heat_phase.append(build_base_heat(hit_counts_phase[p], miss_counts_phase[p]))

    scored_placements: Dict[str, List[Tuple[Placement, float]]] = {}
    for ship, plist in placements.items():
        tmp = []
        for p in plist:
            s = sum(base_heat_phase[0][r][c] for (r, c) in p.cells)
            tmp.append((p, s))
        tmp.sort(key=lambda ps: ps[1])
        scored_placements[ship] = tmp

    best_layout = None
    best_mask = 0
    best_robust = -1.0
    best_heat = 0.0
    best_seq = 0.0

    for _ in range(n_iter):
        used_mask = 0
        layout = {}
        ship_order = SHIP_ORDER[:]
        rng.shuffle(ship_order)
        for ship in ship_order:
            candidates = [(p, s) for (p, s) in scored_placements[ship] if not (p.mask & used_mask)]
            if not candidates:
                layout = None
                break
            top_k = min(10, len(candidates))
            p, s = rng.choice(candidates[:top_k])
            layout[ship] = p
            used_mask |= p.mask
        if layout is None:
            continue
        union_mask = 0
        for p in layout.values():
            union_mask |= p.mask
        total_heat = 0.0
        total_seq = 0.0
        for _ in range(sim_games_per_layout):
            sh, _ = simulate_enemy_game_phase(union_mask, base_heat_phase, disp_counts, "heat", rng)
            ss, _ = simulate_enemy_game_phase(union_mask, base_heat_phase, disp_counts, "seq", rng)
            total_heat += sh
            total_seq += ss
        avg_heat = total_heat / sim_games_per_layout
        avg_seq = total_seq / sim_games_per_layout
        robust = min(avg_heat, avg_seq)
        if robust > best_robust:
            best_robust = robust
            best_layout = layout
            best_mask = union_mask
            best_heat = avg_heat
            best_seq = avg_seq

    return best_layout, best_mask, best_robust, best_heat, best_seq


class AttackTab(QtWidgets.QWidget):
    STATE_PATH = "battleship_attack_state.json"

    def __init__(self, stats: "StatsTracker", parent=None):
        super().__init__(parent)
        self.stats = stats
        self.placements = PLACEMENTS
        self.board = create_board()

        self.world_masks: List[int] = []
        self.cell_hit_counts: List[int] = [0] * (BOARD_SIZE * BOARD_SIZE)
        self.cell_probs: List[float] = [0.0] * (BOARD_SIZE * BOARD_SIZE)
        self.info_gain_values: List[float] = [0.0] * (BOARD_SIZE * BOARD_SIZE)
        self.num_world_samples: int = 0
        self.ship_sunk_probs: Dict[str, float] = {s: 0.0 for s in SHIP_ORDER}
        self.confirmed_sunk: Set[str] = set()
        self.assigned_hits: Dict[str, Set[Tuple[int, int]]] = {s: set() for s in SHIP_ORDER}
        self.assign_mode_ship: Optional[str] = None
        self.best_cells: List[Tuple[int, int]] = []
        self.best_prob: float = 0.0
        self.game_over: bool = False

        self.linked_defense_tab = None

        # World-model diagnostics
        self.enumeration_mode: bool = False  # True = exact enumeration, False = Monte Carlo
        self.remaining_ship_count: int = len(SHIP_ORDER)

        self.ship_friendly_names = {
            "line2": "Length-2 line",
            "line3": "Length-3 line",
            "square2": "2×2 square",
            "L3": "L-tromino",
        }

        self._build_ui()
        self.load_state()
        self.recompute()

    def _build_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(24)

        board_container = QtWidgets.QWidget()
        board_layout = QtWidgets.QGridLayout(board_container)
        board_layout.setSpacing(2)
        board_layout.setContentsMargins(24, 24, 24, 24)

        # column labels
        for c in range(BOARD_SIZE):
            lbl = QtWidgets.QLabel(chr(ord("A") + c))
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setStyleSheet(f"color: {Theme.TEXT_LABEL};")
            board_layout.addWidget(lbl, 0, c + 1)
        # row labels
        for r in range(BOARD_SIZE):
            lbl = QtWidgets.QLabel(str(r + 1))
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setStyleSheet(f"color: {Theme.TEXT_LABEL};")
            board_layout.addWidget(lbl, r + 1, 0)

        self.cell_buttons: List[List[QtWidgets.QPushButton]] = []
        for r in range(BOARD_SIZE):
            row = []
            for c in range(BOARD_SIZE):
                btn = QtWidgets.QPushButton("")
                btn.setFixedSize(48, 48)
                btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
                btn.clicked.connect(self._make_cell_handler(r, c))
                row.append(btn)
                board_layout.addWidget(btn, r + 1, c + 1)
            self.cell_buttons.append(row)

        main_layout.addWidget(board_container, stretch=0, alignment=QtCore.Qt.AlignCenter)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setSpacing(8)

        header = QtWidgets.QLabel("Tab 1: You attack")
        hfont = header.font()
        hfont.setPointSize(hfont.pointSize() + 1)
        hfont.setBold(True)
        header.setFont(hfont)
        right_layout.addWidget(header)

        desc = QtWidgets.QLabel(
            "Click cells to record hits/misses.\n"
            "The solver maintains a set of plausible layouts, uses information metrics, "
            "and can optionally enforce which hits belong to which ship."
        )
        desc.setWordWrap(True)
        right_layout.addWidget(desc)

        overlay_layout = QtWidgets.QHBoxLayout()
        overlay_label = QtWidgets.QLabel("Overlay:")
        self.overlay_combo = QtWidgets.QComboBox()
        self.overlay_combo.addItems(["None", "Hit probability (%)", "Info gain (0–100)"])
        self.overlay_combo.currentIndexChanged.connect(self.update_board_view)
        overlay_layout.addWidget(overlay_label)
        overlay_layout.addWidget(self.overlay_combo)
        right_layout.addLayout(overlay_layout)

        self.recompute_button = QtWidgets.QPushButton("Recompute now")
        self.recompute_button.clicked.connect(self.recompute)
        right_layout.addWidget(self.recompute_button)

        self.clear_attack_button = QtWidgets.QPushButton("Clear board (new game)")
        self.clear_attack_button.clicked.connect(self.clear_board)
        right_layout.addWidget(self.clear_attack_button)

        self.summary_label = QtWidgets.QLabel("")
        self.summary_label.setWordWrap(True)
        right_layout.addWidget(self.summary_label)

        self.world_mode_label = QtWidgets.QLabel("")
        self.world_mode_label.setWordWrap(True)
        right_layout.addWidget(self.world_mode_label)

        self.best_label = QtWidgets.QLabel("Best guess: (none)")
        self.best_label.setWordWrap(True)
        right_layout.addWidget(self.best_label)

        sunk_group = QtWidgets.QGroupBox("Mark ships confirmed sunk")
        sunk_layout = QtWidgets.QVBoxLayout(sunk_group)
        self.sunk_checkboxes: Dict[str, QtWidgets.QCheckBox] = {}
        for ship in SHIP_ORDER:
            cb = QtWidgets.QCheckBox(self.ship_friendly_names[ship])
            cb.stateChanged.connect(self._make_sunk_handler(ship))
            sunk_layout.addWidget(cb)
            self.sunk_checkboxes[ship] = cb
        right_layout.addWidget(sunk_group)

        assign_group = QtWidgets.QGroupBox("Assign hits to ships (optional)")
        assign_layout = QtWidgets.QVBoxLayout(assign_group)
        self.assign_none_rb = QtWidgets.QRadioButton("No assignment mode")
        self.assign_none_rb.setChecked(True)
        self.assign_none_rb.toggled.connect(self._assign_mode_changed)
        assign_layout.addWidget(self.assign_none_rb)
        self.assign_ship_rbs: Dict[str, QtWidgets.QRadioButton] = {}
        for ship in SHIP_ORDER:
            rb = QtWidgets.QRadioButton(f"Assign hits to {self.ship_friendly_names[ship]}")
            rb.toggled.connect(self._assign_mode_changed)
            assign_layout.addWidget(rb)
            self.assign_ship_rbs[ship] = rb
        right_layout.addWidget(assign_group)

        status_title = QtWidgets.QLabel("Ship status (sunk probability)")
        stf = status_title.font()
        stf.setBold(True)
        status_title.setFont(stf)
        right_layout.addWidget(status_title)

        self.ship_status_labels: Dict[str, QtWidgets.QLabel] = {}
        for ship in SHIP_ORDER:
            lbl = QtWidgets.QLabel(f"{self.ship_friendly_names[ship]}: unknown")
            right_layout.addWidget(lbl)
            self.ship_status_labels[ship] = lbl

        # --- New: game result + win rate group ---
        stats_group = QtWidgets.QGroupBox("Game result & win rate")
        stats_layout = QtWidgets.QVBoxLayout(stats_group)
        self.stats_label = QtWidgets.QLabel(self.stats.summary_text())
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)

        btn_row = QtWidgets.QHBoxLayout()
        self.win_button = QtWidgets.QPushButton("Record WIN + new game")
        self.loss_button = QtWidgets.QPushButton("Record LOSS + new game")
        self.win_button.clicked.connect(self._record_win)
        self.loss_button.clicked.connect(self._record_loss)
        btn_row.addWidget(self.win_button)
        btn_row.addWidget(self.loss_button)
        stats_layout.addLayout(btn_row)
        right_layout.addWidget(stats_group)
        # --- end new block ---

        self.win_prob_label = QtWidgets.QLabel("Win Probability: N/A")
        self.win_prob_label.setStyleSheet(f"color: {Theme.HIGHLIGHT}; font-weight: bold; font-size: 14px;")
        right_layout.addWidget(self.win_prob_label)  # Add somewhere prominent

        right_layout.addStretch(1)
        main_layout.addWidget(right_panel, stretch=1)

    def save_state(self, path: Optional[str] = None):
        """
        Persist the current attack game state so that if the app closes
        mid-game, we can resume without re-entering hits.
        """
        if path is None:
            path = self.STATE_PATH

        # Serialize assigned_hits as lists of [r, c]
        assigned_hits_serializable = {
            ship: [[r, c] for (r, c) in sorted(list(coords))]
            for ship, coords in self.assigned_hits.items()
            if coords
        }

        data = {
            "board": self.board,  # 8x8 of ".", "o", "x"
            "confirmed_sunk": list(self.confirmed_sunk),
            "assigned_hits": assigned_hits_serializable,
            "overlay_mode": self.overlay_combo.currentIndex(),
            "game_over": self.game_over,
        }

        try:
            with open(path, "w") as f:
                json.dump(data, f)
        except OSError:
            # Don't crash app just because we couldn't save
            pass

    def load_state(self, path: Optional[str] = None):
        """
        Restore a previous in-progress attack game if one exists.

        If the saved state indicates the game was already over (all ships
        marked sunk), we treat it as finished and start fresh.
        """
        if path is None:
            path = self.STATE_PATH
        if not os.path.exists(path):
            return

        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return

        # If the last saved game was already marked as over,
        # treat that as a finished game and do NOT restore it.
        if data.get("game_over", False):
            try:
                os.remove(path)
            except OSError:
                pass
            return

        board = data.get("board")
        if not isinstance(board, list) or len(board) != BOARD_SIZE:
            return

        # Basic sanity check on board shape
        ok = True
        for row in board:
            if not isinstance(row, list) or len(row) != BOARD_SIZE:
                ok = False
                break
        if not ok:
            return

        # Restore board
        self.board = board

        # Restore confirmed_sunk
        conf = data.get("confirmed_sunk", [])
        self.confirmed_sunk = set(s for s in conf if s in SHIP_ORDER)

        # Sync sunk checkboxes
        for ship, cb in self.sunk_checkboxes.items():
            cb.blockSignals(True)
            cb.setChecked(ship in self.confirmed_sunk)
            cb.blockSignals(False)

        # Restore assigned_hits
        self.assigned_hits = {s: set() for s in SHIP_ORDER}
        ah = data.get("assigned_hits", {})
        if isinstance(ah, dict):
            for ship, coords in ah.items():
                if ship not in self.assigned_hits:
                    continue
                if not isinstance(coords, list):
                    continue
                sset = set()
                for pair in coords:
                    if (
                            isinstance(pair, list)
                            and len(pair) == 2
                            and isinstance(pair[0], int)
                            and isinstance(pair[1], int)
                    ):
                        r, c = pair
                        if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                            sset.add((r, c))
                self.assigned_hits[ship] = sset

        # Restore overlay mode (optional)
        overlay_mode = data.get("overlay_mode")
        if isinstance(overlay_mode, int) and 0 <= overlay_mode < self.overlay_combo.count():
            self.overlay_combo.setCurrentIndex(overlay_mode)

    def _record_game_result(self, win: bool):
        self.stats.record_game(win)
        self.stats_label.setText(self.stats.summary_text())
        # Start a completely fresh board for the next game
        self.clear_board()

    def _record_win(self):
        self._record_game_result(True)

    def _record_loss(self):
        self._record_game_result(False)

    def _assign_mode_changed(self):
        if self.assign_none_rb.isChecked():
            self.assign_mode_ship = None
        else:
            for ship, rb in self.assign_ship_rbs.items():
                if rb.isChecked():
                    self.assign_mode_ship = ship
                    break
        self.update_board_view()

    def _make_cell_handler(self, r: int, c: int):
        def handler():
            if self.assign_mode_ship is not None and self.board[r][c] == HIT:
                ship = self.assign_mode_ship
                sset = self.assigned_hits[ship]
                if (r, c) in sset:
                    sset.remove((r, c))
                else:
                    # remove from other ships
                    for other in SHIP_ORDER:
                        if other != ship:
                            self.assigned_hits[other].discard((r, c))
                    sset.add((r, c))
                self.recompute()
                return

            current = self.board[r][c]
            if current == EMPTY:
                choice = self._ask_hit_or_miss(r, c)
                if choice is None:
                    return
                self.board[r][c] = HIT if choice == "hit" else MISS
            else:
                self.board[r][c] = EMPTY

            if self.board[r][c] != HIT:
                for ship in SHIP_ORDER:
                    self.assigned_hits[ship].discard((r, c))

            self.recompute()

        return handler

    def _ask_hit_or_miss(self, r: int, c: int) -> Optional[str]:
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("Record result")
        msg.setText(f"Cell ({r + 1}, {chr(ord('A') + c)}): record result?")
        hit_btn = msg.addButton("Hit", QtWidgets.QMessageBox.AcceptRole)
        miss_btn = msg.addButton("Miss", QtWidgets.QMessageBox.DestructiveRole)
        cancel_btn = msg.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        msg.exec_()
        clicked = msg.clickedButton()
        if clicked == hit_btn:
            return "hit"
        elif clicked == miss_btn:
            return "miss"
        else:
            return None

    def _make_sunk_handler(self, ship: str):
        def handler(state: int):
            if state == QtCore.Qt.Checked:
                self.confirmed_sunk.add(ship)
            else:
                self.confirmed_sunk.discard(ship)
            self.recompute()

        return handler

    def clear_board(self):
        self.board = create_board()
        self.confirmed_sunk.clear()
        for cb in self.sunk_checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)
        for ship in SHIP_ORDER:
            self.assigned_hits[ship].clear()
        self.assign_none_rb.setChecked(True)
        self.game_over = False
        self.recompute()

        # Optional: also wipe any saved in-progress state
        try:
            if os.path.exists(self.STATE_PATH):
                os.remove(self.STATE_PATH)
        except OSError:
            pass

    def _are_all_ships_sunk(self) -> bool:
        # Only trust the user's explicit checkboxes for "game over".
        return len(self.confirmed_sunk) == len(SHIP_ORDER)

    # In AttackTab class, add the prediction logic:
    def update_win_prediction(self, defense_tab):
        """Simulate Me vs Opponent."""
        if not self.world_masks or not defense_tab.layout_board:
            self.win_prob_label.setText("Win Prob: N/A (Need Defense Layout)")
            return

        # 1. Estimate MY remaining shots
        my_rem_samples = []
        rng = random.Random()
        # Sample up to 10 worlds
        sample_worlds = rng.sample(self.world_masks, min(10, len(self.world_masks)))

        for w_mask in sample_worlds:
            # Quick sim of greedy solver vs this world
            # Copy board
            sim_board = [row[:] for row in self.board]
            shots = 0
            rem_targets = bin(w_mask).count('1') - sum(row.count(HIT) for row in self.board)

            while rem_targets > 0 and shots < 64:
                # Simple greedy selection on sim_board
                # (You can copy the logic from _choose_next_shot_for_strategy here
                # or make a fast helper. For speed, just pick random unknown cell)
                unknown = [(r, c) for r in range(8) for c in range(8) if sim_board[r][c] == EMPTY]
                if not unknown: break
                r, c = rng.choice(unknown)  # Fast approximation
                sim_board[r][c] = HIT if (w_mask & (1 << cell_index(r, c))) else MISS
                if sim_board[r][c] == HIT: rem_targets -= 1
                shots += 1
            my_rem_samples.append(shots)

        my_avg_rem = sum(my_rem_samples) / len(my_rem_samples)
        my_total = (sum(row.count(HIT) + row.count(MISS) for row in self.board)) + my_avg_rem

        # 2. Estimate OPPONENT remaining shots
        # Create mask from defense tab layout
        layout_mask = 0
        for r in range(8):
            for c in range(8):
                if defense_tab.layout_board[r][c] == HAS_SHIP:
                    layout_mask |= (1 << cell_index(r, c))

        base_heat = [build_base_heat(defense_tab.hit_counts_phase[p], defense_tab.miss_counts_phase[p]) for p in
                     range(4)]

        opp_rem_samples = []
        for _ in range(10):
            # Run simulation starting from current defense board state
            s_taken, _ = simulate_enemy_game_phase(
                layout_mask, base_heat, defense_tab.disp_counts, "seq", rng,
                initial_shot_board=defense_tab.shot_board
            )
            # simulate_enemy returns TOTAL shots, so we don't need to add current
            opp_rem_samples.append(s_taken)

        opp_avg = sum(opp_rem_samples) / len(opp_rem_samples)

        # 3. Calculate Win % (Simple comparison of distributions)
        wins = 0
        total_comps = 0
        for m in my_rem_samples:
            # adjusting my total vs opp total
            m_tot = (sum(row.count(HIT) + row.count(MISS) for row in self.board)) + m
            for o in opp_rem_samples:
                if m_tot < o: wins += 1  # I finish faster
                total_comps += 1

        prob = (wins / total_comps) * 100 if total_comps else 0
        self.win_prob_label.setText(f"Win Probability: {prob:.1f}% (Me: ~{my_total:.1f} vs Opp: ~{opp_avg:.1f})")

    def recompute(self, defense_tab=None):
        # Rebuild world samples and ship-sunk probabilities
        self.world_masks, self.cell_hit_counts, self.ship_sunk_probs, self.num_world_samples = sample_worlds(
            self.board, self.placements, self.confirmed_sunk, self.assigned_hits
        )

        N = self.num_world_samples
        if N > 0:
            self.cell_probs = [cnt / N for cnt in self.cell_hit_counts]
        else:
            self.cell_probs = [0.0] * (BOARD_SIZE * BOARD_SIZE)

        # Detect whether the underlying world model used exact enumeration
        # or Monte Carlo sampling, mirroring the logic in sample_worlds.
        if N > 0:
            # How many ships remain (not confirmed sunk)?
            remaining_ships = [s for s in SHIP_ORDER if s not in self.confirmed_sunk]
            self.remaining_ship_count = len(remaining_ships)

            # Recompute allowed placements to estimate the search-space size.
            hit_mask, miss_mask = board_masks(self.board)
            allowed = filter_allowed_placements(
                self.placements,
                hit_mask,
                miss_mask,
                self.confirmed_sunk,
                self.assigned_hits,
            )

            force_enumeration = (self.remaining_ship_count == 1)

            product = 1
            enumeration = True
            if not force_enumeration:
                for ship in SHIP_ORDER:
                    n = len(allowed[ship])
                    product *= n
                    if product > ENUMERATION_PRODUCT_LIMIT:
                        enumeration = False
                        break
            else:
                # Endgame: only one ship left → always enumerate in sample_worlds.
                enumeration = True

            self.enumeration_mode = enumeration
        else:
            # No consistent layouts found – mode doesn't really matter.
            self.enumeration_mode = False
            self.remaining_ship_count = len([s for s in SHIP_ORDER if s not in self.confirmed_sunk])

        self.game_over = self._are_all_ships_sunk()

        if not self.game_over and N > 0:
            self._choose_best_cells_with_2ply()
        else:
            self.best_cells = []
            self.best_prob = 0.0
            self.info_gain_values = [0.0] * (BOARD_SIZE * BOARD_SIZE)

        self.update_board_view()
        self.update_status_view()

        if self.linked_defense_tab:
            self.update_win_prediction(self.linked_defense_tab)

    def _choose_best_cells_with_2ply(self):
        if not self.world_masks:
            self.best_cells = []
            self.best_prob = 0.0
            self.info_gain_values = [0.0] * (BOARD_SIZE * BOARD_SIZE)
            return

        info_vals, best_indices, best_p = two_ply_selection(
            self.board,
            self.world_masks,
            self.cell_hit_counts,
        )
        self.info_gain_values = info_vals
        self.best_prob = best_p
        self.best_cells = [
            (idx // BOARD_SIZE, idx % BOARD_SIZE) for idx in best_indices
        ]

    def _get_interpolated_color(self, val: float) -> str:
        """
        val: 0.0 to 1.0
        Returns a hex string or rgb string for background-color.
        Gradient: Dark Slate (#020617) -> Bright Sky Blue (#0ea5e9)
        """
        # Clamp value just in case
        val = max(0.0, min(1.0, val))

        # Start Color (Base background): #020617 -> (2, 6, 23)
        start_r, start_g, start_b = 2, 6, 23

        # End Color (Max heat): #0ea5e9 -> (14, 165, 233)
        end_r, end_g, end_b = 14, 165, 233

        # Linear Interpolation
        r = int(start_r + (end_r - start_r) * val)
        g = int(start_g + (end_g - start_g) * val)
        b = int(start_b + (end_b - start_b) * val)

        return f"rgb({r},{g},{b})"

    # ------------------------------------------------------------------------
    # UPDATED: update_board_view with Heatmap Logic
    # ------------------------------------------------------------------------
    def update_board_view(self):
        overlay_mode = self.overlay_combo.currentIndex()
        show_hit_prob = (overlay_mode == 1 and not self.game_over)
        show_info_gain = (overlay_mode == 2 and not self.game_over)

        best_set = {(r, c) for (r, c) in self.best_cells}

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                btn = self.cell_buttons[r][c]
                state = self.board[r][c]
                idx = cell_index(r, c)

                # Default empty style
                base_color = Theme.BG_DARK
                text_color = Theme.TEXT_MAIN
                border_style = f"1px solid {Theme.BORDER_EMPTY}"
                text = ""

                if state == EMPTY:
                    # --- HEATMAP LOGIC START ---
                    heat_val = 0.0
                    if show_hit_prob:
                        heat_val = self.cell_probs[idx]
                        if heat_val > 0:
                            text = f"{int(round(heat_val * 100))}"
                    elif show_info_gain:
                        heat_val = self.info_gain_values[idx]
                        if heat_val > 0:
                            text = f"{int(round(heat_val * 100))}"

                    # Apply gradient if we have a value to show
                    if heat_val > 0.0:
                        base_color = self._get_interpolated_color(heat_val)
                        # If the background gets too bright, switch text to black for contrast
                        if heat_val > 0.6:
                            text_color = Theme.TEXT_DARK
                    # --- HEATMAP LOGIC END ---

                elif state == MISS:
                    base_color = Theme.MISS_BG
                    text_color = Theme.MISS_TEXT
                    border_style = f"1px solid {Theme.MISS_BORDER}"
                    text = "M"
                elif state == HIT:
                    base_color = Theme.HIT_BG
                    text_color = Theme.HIT_TEXT
                    border_style = f"1px solid {Theme.HIT_BORDER}"
                    text = "H"
                    assigned = any((r, c) in self.assigned_hits[s] for s in SHIP_ORDER)
                    if assigned:
                        border_style = f"2px dashed {Theme.ASSIGNED_BORDER}"

                # Construct the final stylesheet
                style_str = (
                    f"background-color: {base_color};"
                    f"color: {text_color};"
                    f"border: {border_style};"
                )

                # Highlight the "Best Guess" recommended by the solver
                if not self.game_over and (r, c) in best_set and state == EMPTY:
                    # Add a bright cyan border to the best moves
                    style_str += f"border: 2px solid {Theme.BORDER_BEST};"

                btn.setStyleSheet(style_str)
                btn.setText(text)

    def update_status_view(self):
        # Best-guess text
        if self.game_over:
            self.best_label.setText(
                "You marked all ships as sunk. Uncheck a ship to resume suggestions."
            )
        else:
            if self.best_cells:
                cells_str = ", ".join(f"({r + 1},{chr(ord('A') + c)})" for (r, c) in self.best_cells)
                self.best_label.setText(
                    f"Best guess (2-step info): {cells_str} (p_hit≈{self.best_prob:.3f}, layouts≈{self.num_world_samples})"
                )
            else:
                self.best_label.setText("Best guess: (none)")

        # Summary of current evidence
        known_hits = sum(
            1 for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if self.board[r][c] == HIT
        )
        known_misses = sum(
            1 for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if self.board[r][c] == MISS
        )
        unknown = BOARD_SIZE * BOARD_SIZE - known_hits - known_misses
        expected_ship_cells = sum(self.cell_probs)

        self.summary_label.setText(
            f"Known hits: {known_hits}, misses: {known_misses}, unknown: {unknown}\n"
            f"World samples/layouts: {self.num_world_samples}, expected ship cells (total): {expected_ship_cells:.2f}"
        )

        # New: world model mode (exact enumeration vs Monte Carlo)
        if self.num_world_samples == 0:
            self.world_mode_label.setText(
                "World model: no consistent layouts (check hits/misses and sunk flags)."
            )
        else:
            if self.enumeration_mode:
                if self.remaining_ship_count == 1:
                    mode_str = "exact enumeration (endgame: one ship left)"
                else:
                    mode_str = "exact enumeration (small search space)"
            else:
                mode_str = "Monte Carlo sampling"

            self.world_mode_label.setText(
                f"World model: {mode_str}, layouts ≈ {self.num_world_samples}"
            )

        # Per-ship sunk probabilities
        for ship in SHIP_ORDER:
            lbl = self.ship_status_labels[ship]
            prob = self.ship_sunk_probs.get(ship, 0.0)

            if ship in self.confirmed_sunk:
                status = "SUNK"
                color = Theme.STATUS_SUNK
                prob_display = 1.0
            elif prob >= 0.99:
                status = "SUNK?"
                color = Theme.STATUS_SUNK_MAYBE
                prob_display = prob
            elif prob <= 0.01:
                status = "AFLOAT"
                color = Theme.STATUS_AFLOAT
                prob_display = prob
            else:
                status = "MAYBE"
                color = Theme.STATUS_MAYBE
                prob_display = prob

            lbl.setText(
                f"{self.ship_friendly_names[ship]}: {status} (p≈{prob_display:.2f})"
            )
            lbl.setStyleSheet(f"color: {color};")


class DefenseTab(QtWidgets.QWidget):
    STATE_PATH = "battleship_defense_state.json"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.placements = generate_ship_placements()
        self.layout_board = [[NO_SHIP for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.shot_board = [[NO_SHOT for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

        self.hit_counts_phase = [
            [[0.0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
            for _ in range(4)
        ]
        self.miss_counts_phase = [
            [[0.0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
            for _ in range(4)
        ]
        self.disp_counts = [
            [
                [
                    [0.0 for _ in range(2 * DISP_RADIUS + 1)]
                    for _ in range(2 * DISP_RADIUS + 1)
                ]
                for _ in range(2)
            ]
            for _ in range(4)
        ]

        self.history_events: List[Tuple[int, int, bool, int]] = []
        self.last_shot_for_sequence: Optional[Tuple[int, int, int, bool]] = None

        self.recommended_layout = None
        self.recommended_mask = 0
        self.recommended_robust = 0.0
        self.recommended_heat = 0.0
        self.recommended_seq = 0.0

        self._build_ui()
        self.load_state()
        self.update_board_view()
        self.update_summary_labels()

    def _build_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(24)

        board_container = QtWidgets.QWidget()
        board_layout = QtWidgets.QGridLayout(board_container)
        board_layout.setSpacing(2)
        board_layout.setContentsMargins(24, 24, 24, 24)

        for c in range(BOARD_SIZE):
            lbl = QtWidgets.QLabel(chr(ord("A") + c))
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setStyleSheet(f"color: {Theme.TEXT_LABEL};")
            board_layout.addWidget(lbl, 0, c + 1)
        for r in range(BOARD_SIZE):
            lbl = QtWidgets.QLabel(str(r + 1))
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setStyleSheet(f"color: {Theme.TEXT_LABEL};")
            board_layout.addWidget(lbl, r + 1, 0)

        self.cell_buttons_def: List[List[QtWidgets.QPushButton]] = []
        for r in range(BOARD_SIZE):
            row = []
            for c in range(BOARD_SIZE):
                btn = QtWidgets.QPushButton("")
                btn.setFixedSize(48, 48)
                btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
                btn.clicked.connect(self._make_cell_handler(r, c))
                row.append(btn)
                board_layout.addWidget(btn, r + 1, c + 1)
            self.cell_buttons_def.append(row)

        main_layout.addWidget(board_container, stretch=0, alignment=QtCore.Qt.AlignCenter)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setSpacing(8)

        header = QtWidgets.QLabel("Tab 2: Opponent attacks you")
        f = header.font()
        f.setPointSize(f.pointSize() + 1)
        f.setBold(True)
        header.setFont(f)
        right_layout.addWidget(header)

        desc = QtWidgets.QLabel(
            "Paint your ships in 'Edit layout' mode.\n"
            "Switch to 'Record opponent shots' and click where they shoot.\n"
            "The app learns phase-specific heatmaps and shot sequences, and "
            "then suggests layouts that are robust to both behaviours."
        )
        desc.setWordWrap(True)
        right_layout.addWidget(desc)

        mode_group = QtWidgets.QGroupBox("Mode")
        mode_layout = QtWidgets.QVBoxLayout(mode_group)
        self.layout_mode_rb = QtWidgets.QRadioButton("Edit your ship layout")
        self.shot_mode_rb = QtWidgets.QRadioButton("Record opponent shots")
        self.layout_mode_rb.setChecked(True)
        mode_layout.addWidget(self.layout_mode_rb)
        mode_layout.addWidget(self.shot_mode_rb)
        right_layout.addWidget(mode_group)

        ctrl_group = QtWidgets.QGroupBox("Controls")
        ctrl_layout = QtWidgets.QVBoxLayout(ctrl_group)

        self.clear_layout_btn = QtWidgets.QPushButton("Clear layout")
        self.clear_layout_btn.clicked.connect(self.clear_layout)
        ctrl_layout.addWidget(self.clear_layout_btn)

        self.clear_shots_btn = QtWidgets.QPushButton("New game (clear board shots)")
        self.clear_shots_btn.clicked.connect(self.clear_shots)
        ctrl_layout.addWidget(self.clear_shots_btn)

        self.reset_model_btn = QtWidgets.QPushButton("Reset learning (all history)")
        self.reset_model_btn.clicked.connect(self.reset_model)
        ctrl_layout.addWidget(self.reset_model_btn)

        self.suggest_layout_btn = QtWidgets.QPushButton("Suggest layout from history")
        self.suggest_layout_btn.clicked.connect(self.compute_suggested_layout)
        ctrl_layout.addWidget(self.suggest_layout_btn)

        self.apply_suggestion_btn = QtWidgets.QPushButton("Apply suggested layout")
        self.apply_suggested_layout_btn = self.apply_suggestion_btn
        self.apply_suggestion_btn.clicked.connect(self.apply_suggested_layout)
        ctrl_layout.addWidget(self.apply_suggestion_btn)

        right_layout.addWidget(ctrl_group)

        self.heatmap_checkbox = QtWidgets.QCheckBox("Show total shot counts (heatmap overlay)")
        self.heatmap_checkbox.stateChanged.connect(self.update_board_view)
        right_layout.addWidget(self.heatmap_checkbox)

        self.summary_label = QtWidgets.QLabel("")
        self.summary_label.setWordWrap(True)
        right_layout.addWidget(self.summary_label)

        self.recommendation_label = QtWidgets.QLabel(
            "No layout suggestion yet.\nRecord opponent shots, then click 'Suggest layout'."
        )
        self.recommendation_label.setWordWrap(True)
        right_layout.addWidget(self.recommendation_label)

        right_layout.addStretch(1)
        main_layout.addWidget(right_panel, stretch=1)

    def save_state(self, path: Optional[str] = None):
        if path is None:
            path = self.STATE_PATH

        # history_events is List[Tuple[int,int,bool,int]]
        history_serializable = [
            [int(r), int(c), bool(was_hit), int(phase)]
            for (r, c, was_hit, phase) in self.history_events
        ]

        # last_shot_for_sequence is Optional[Tuple[int,int,int,bool]]
        if self.last_shot_for_sequence is None:
            last_seq = None
        else:
            lr, lc, phase, was_hit = self.last_shot_for_sequence
            last_seq = [int(lr), int(lc), int(phase), bool(was_hit)]

        data = {
            # long-term learning stats
            "hit_counts_phase": self.hit_counts_phase,
            "miss_counts_phase": self.miss_counts_phase,
            "disp_counts": self.disp_counts,

            # current game state
            "layout_board": self.layout_board,
            "shot_board": self.shot_board,
            "history_events": history_serializable,
            "last_shot_for_sequence": last_seq,
        }

        try:
            with open(path, "w") as f:
                json.dump(data, f)
        except OSError:
            # Don't crash just because we can't save
            pass

    def load_state(self, path: Optional[str] = None):
        if path is None:
            path = self.STATE_PATH
        if not os.path.exists(path):
            return

        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return

        # --- long-term learning stats (backwards compatible) ---
        hc = data.get("hit_counts_phase")
        mc = data.get("miss_counts_phase")
        dc = data.get("disp_counts")
        if isinstance(hc, list) and isinstance(mc, list) and len(hc) == 4 and len(mc) == 4:
            self.hit_counts_phase = hc
            self.miss_counts_phase = mc
        if isinstance(dc, list) and len(dc) == 4:
            self.disp_counts = dc

        # --- current layout board (optional, for mid-game resume) ---
        lb = data.get("layout_board")
        if (
                isinstance(lb, list)
                and len(lb) == BOARD_SIZE
                and all(isinstance(row, list) and len(row) == BOARD_SIZE for row in lb)
        ):
            self.layout_board = lb

        # --- current shot board (optional, for mid-game resume) ---
        sb = data.get("shot_board")
        if (
                isinstance(sb, list)
                and len(sb) == BOARD_SIZE
                and all(isinstance(row, list) and len(row) == BOARD_SIZE for row in sb)
        ):
            self.shot_board = sb

        # --- shot history for correct undo behaviour ---
        he = data.get("history_events")
        new_history: List[Tuple[int, int, bool, int]] = []
        if isinstance(he, list):
            for item in he:
                if (
                        isinstance(item, list)
                        and len(item) == 4
                        and isinstance(item[0], int)
                        and isinstance(item[1], int)
                        and isinstance(item[3], int)
                ):
                    r, c, was_hit, phase = item
                    if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and 0 <= phase <= 3:
                        new_history.append((r, c, bool(was_hit), phase))
        self.history_events = new_history

        # --- last shot for sequence model (optional) ---
        lss = data.get("last_shot_for_sequence")
        if (
                isinstance(lss, list)
                and len(lss) == 4
                and isinstance(lss[0], int)
                and isinstance(lss[1], int)
                and isinstance(lss[2], int)
        ):
            lr, lc, phase, was_hit = lss
            if 0 <= lr < BOARD_SIZE and 0 <= lc < BOARD_SIZE and 0 <= phase <= 3:
                self.last_shot_for_sequence = (lr, lc, phase, bool(was_hit))
            else:
                self.last_shot_for_sequence = None
        else:
            self.last_shot_for_sequence = None

    def _make_cell_handler(self, r: int, c: int):
        def handler():
            if self.layout_mode_rb.isChecked():
                self.toggle_ship_cell(r, c)
            else:
                self.record_shot_at(r, c)

        return handler

    def toggle_ship_cell(self, r: int, c: int):
        self.layout_board[r][c] = HAS_SHIP if self.layout_board[r][c] == NO_SHIP else NO_SHIP
        self.update_board_view()
        self.update_summary_labels()

    def clear_layout(self):
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                self.layout_board[r][c] = NO_SHIP
        self.update_board_view()
        self.update_summary_labels()

    def _decay_counts(self, factor: float = 0.97):
        """
        Exponentially decay all learned statistics so that recent games
        have more influence than very old ones.

        Counts are stored as floats; we deliberately *do not* truncate to
        integers here, otherwise low-frequency patterns vanish too fast.
        """
        for p in range(4):
            # Per-cell hit/miss counts
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    self.hit_counts_phase[p][r][c] *= factor
                    self.miss_counts_phase[p][r][c] *= factor

            # Shot-displacement counts (sequence model)
            for hm in range(2):
                for dr in range(2 * DISP_RADIUS + 1):
                    for dc in range(2 * DISP_RADIUS + 1):
                        self.disp_counts[p][hm][dr][dc] *= factor

    def _compute_sunk_ship_count(self) -> int:
        visited = [[False for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        sunk_count = 0
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.layout_board[r][c] != HAS_SHIP or visited[r][c]:
                    continue
                stack = [(r, c)]
                visited[r][c] = True
                cells = []
                while stack:
                    cr, cc = stack.pop()
                    cells.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        rr = cr + dr
                        cc2 = cc + dc
                        if 0 <= rr < BOARD_SIZE and 0 <= cc2 < BOARD_SIZE and not visited[rr][cc2] and \
                                self.layout_board[rr][cc2] == HAS_SHIP:
                            visited[rr][cc2] = True
                            stack.append((rr, cc2))
                sunk = True
                for cr, cc in cells:
                    if self.shot_board[cr][cc] != SHOT_HIT:
                        sunk = False
                        break
                if sunk:
                    sunk_count += 1
        return sunk_count

    def record_shot_at(self, r: int, c: int):
        has_ship = self.layout_board[r][c] == HAS_SHIP
        current = self.shot_board[r][c]
        if current == NO_SHOT:
            phase = self._compute_sunk_ship_count()
            if phase > 3:
                phase = 3
            if has_ship:
                self.shot_board[r][c] = SHOT_HIT
                self.hit_counts_phase[phase][r][c] += 1
                was_hit = True
            else:
                self.shot_board[r][c] = SHOT_MISS
                self.miss_counts_phase[phase][r][c] += 1
                was_hit = False
            self.history_events.append((r, c, was_hit, phase))
            if self.last_shot_for_sequence is not None:
                lr, lc, last_phase, last_hit = self.last_shot_for_sequence
                dr = r - lr
                dc = c - lc
                if abs(dr) <= DISP_RADIUS and abs(dc) <= DISP_RADIUS:
                    phase_idx = min(last_phase, 3)
                    type_idx = 1 if last_hit else 0
                    self.disp_counts[phase_idx][type_idx][dr + DISP_RADIUS][dc + DISP_RADIUS] += 1
            self.last_shot_for_sequence = (r, c, phase, was_hit)
        else:
            # undo
            for i in range(len(self.history_events) - 1, -1, -1):
                er, ec, was_hit, phase = self.history_events[i]
                if er == r and ec == c:
                    self.history_events.pop(i)
                    if was_hit:
                        if self.hit_counts_phase[phase][r][c] > 0:
                            self.hit_counts_phase[phase][r][c] -= 1
                    else:
                        if self.miss_counts_phase[phase][r][c] > 0:
                            self.miss_counts_phase[phase][r][c] -= 1
                    break
            self.shot_board[r][c] = NO_SHOT
            if self.last_shot_for_sequence is not None:
                lr, lc, _, _ = self.last_shot_for_sequence
                if lr == r and lc == c:
                    self.last_shot_for_sequence = None
        self.update_board_view()
        self.update_summary_labels()

    def clear_shots(self):
        # Treat this as "end of game": decay historical stats so that
        # patterns from very old games matter less than recent ones.
        self._decay_counts()

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                self.shot_board[r][c] = NO_SHOT

        self.history_events.clear()
        self.last_shot_for_sequence = None
        self.update_board_view()
        self.update_summary_labels()

    def reset_model(self):
        for p in range(4):
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    self.hit_counts_phase[p][r][c] = 0
                    self.miss_counts_phase[p][r][c] = 0
        for p in range(4):
            for hm in range(2):
                for dr in range(2 * DISP_RADIUS + 1):
                    for dc in range(2 * DISP_RADIUS + 1):
                        self.disp_counts[p][hm][dr][dc] = 0
        self.history_events.clear()
        self.last_shot_for_sequence = None
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                self.shot_board[r][c] = NO_SHOT
        self.recommended_layout = None
        self.recommended_mask = 0
        self.recommended_robust = 0.0
        self.recommended_heat = 0.0
        self.recommended_seq = 0.0
        self.update_board_view()
        self.update_summary_labels()
        self.recommendation_label.setText(
            "Learning reset.\nRecord opponent shots, then click 'Suggest layout'."
        )
        try:
            if os.path.exists(self.STATE_PATH):
                os.remove(self.STATE_PATH)
        except OSError:
            pass

    def compute_suggested_layout(self):
        layout, mask, robust, avg_heat, avg_seq = recommend_layout_phase(
            self.hit_counts_phase,
            self.miss_counts_phase,
            self.disp_counts,
            self.placements,
            n_iter=250,
            sim_games_per_layout=10,
        )
        if layout is None:
            self.recommended_layout = None
            self.recommended_mask = 0
            self.recommended_robust = 0.0
            self.recommended_heat = 0.0
            self.recommended_seq = 0.0
            self.recommendation_label.setText(
                "Could not compute a layout suggestion (no valid combination)."
            )
        else:
            self.recommended_layout = layout
            self.recommended_mask = mask
            self.recommended_robust = robust
            self.recommended_heat = avg_heat
            self.recommended_seq = avg_seq
            total_hits, total_misses = self._total_hits_misses()
            total_shots = total_hits + total_misses
            total_hits_i = int(round(total_hits))
            total_misses_i = int(round(total_misses))
            total_shots_i = int(round(total_shots))
            if total_shots == 0:
                extra = "No history yet; layout is effectively random."
            else:
                extra = (
                    f"Using {total_shots_i} recorded shots ({total_hits_i} hits, {total_misses_i} misses).\n"
                    f"Estimated shots to sink all ships:\n"
                    f"  Heatmap model: {avg_heat:.1f}\n"
                    f"  Sequence-aware model: {avg_seq:.1f}\n"
                    f"Robust score (min of the two): {robust:.1f}"
                )
            self.recommendation_label.setText(
                "Suggested layout ready.\n" + extra
            )
        self.update_board_view()

    def apply_suggested_layout(self):
        if not self.recommended_layout:
            QtWidgets.QMessageBox.information(
                self,
                "No suggestion",
                "No suggested layout available.\nRecord some shots and click 'Suggest layout' first.",
            )
            return
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                self.layout_board[r][c] = NO_SHIP
        for p in self.recommended_layout.values():
            for r, c in p.cells:
                self.layout_board[r][c] = HAS_SHIP
        # clear preview
        self.recommended_mask = 0
        self.recommended_layout = None
        self.recommended_robust = 0.0
        self.recommended_heat = 0.0
        self.recommended_seq = 0.0
        self.recommendation_label.setText(
            "Suggested layout applied to current board."
        )
        self.update_board_view()
        self.update_summary_labels()

    def _total_hits_misses(self) -> Tuple[int, int]:
        total_hits = 0
        total_misses = 0
        for p in range(4):
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    total_hits += self.hit_counts_phase[p][r][c]
                    total_misses += self.miss_counts_phase[p][r][c]
        return total_hits, total_misses

    def _total_counts_cell(self, r: int, c: int) -> Tuple[int, int]:
        hits = 0
        misses = 0
        for p in range(4):
            hits += self.hit_counts_phase[p][r][c]
            misses += self.miss_counts_phase[p][r][c]
        return hits, misses

    def _total_disp_samples(self) -> int:
        total = 0
        for p in range(4):
            for hm in range(2):
                for dr in range(2 * DISP_RADIUS + 1):
                    for dc in range(2 * DISP_RADIUS + 1):
                        total += self.disp_counts[p][hm][dr][dc]
        return total

    def update_summary_labels(self):
        total_hits, total_misses = self._total_hits_misses()
        total_shots = total_hits + total_misses
        total_hits_i = int(round(total_hits))
        total_misses_i = int(round(total_misses))
        total_shots_i = int(round(total_shots))
        total_layout_cells = sum(
            self.layout_board[r][c] == HAS_SHIP
            for r in range(BOARD_SIZE)
            for c in range(BOARD_SIZE)
        )
        sunk_now = self._compute_sunk_ship_count()
        disp_samples = self._total_disp_samples()
        disp_samples_i = int(round(disp_samples))
        self.summary_label.setText(
            f"Total recorded opponent shots (all games, all phases): {total_shots_i}\n"
            f"Hits: {total_hits_i}, Misses: {total_misses_i}\n"
            f"Current layout: {total_layout_cells} ship cells (ideal is ~12).\n"
            f"Ships sunk this game (approx via clusters): {sunk_now}\n"
            f"Sequence samples (displacements between shots): {disp_samples_i}"
        )

    def update_board_view(self):
        show_heat = self.heatmap_checkbox.isChecked()

        # First pass: max total count for normalization
        max_count = 0.0
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                h, m = self._total_counts_cell(r, c)
                cc = h + m
                if cc > max_count:
                    max_count = cc

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                btn = self.cell_buttons_def[r][c]
                has_ship = (self.layout_board[r][c] == HAS_SHIP)
                shot_state = self.shot_board[r][c]

                hcount, mcount = self._total_counts_cell(r, c)
                total_count = hcount + mcount

                base_style_parts: list[str] = []
                text = ""

                if shot_state == NO_SHOT:
                    if has_ship:
                        # Our own ship that hasn't been shot yet
                        base_style_parts.append(f"background-color: {Theme.LAYOUT_SHIP_BG};")
                        base_style_parts.append(f"color: {Theme.LAYOUT_SHIP_TEXT};")
                        base_style_parts.append(f"border: 1px solid {Theme.LAYOUT_SHIP_BORDER};")
                    else:
                        # Empty water, no shot yet
                        base_style_parts.append(f"background-color: {Theme.BG_DARK};")
                        base_style_parts.append(f"color: {Theme.TEXT_MAIN};")
                        base_style_parts.append(f"border: 1px solid {Theme.BORDER_EMPTY};")
                elif shot_state == SHOT_MISS:
                    base_style_parts.append(f"background-color: {Theme.MISS_BG};")
                    base_style_parts.append(f"color: {Theme.MISS_TEXT};")
                    base_style_parts.append(f"border: 1px solid {Theme.MISS_BORDER};")
                    text = "M"
                elif shot_state == SHOT_HIT:
                    base_style_parts.append(f"background-color: {Theme.HIT_BG};")
                    base_style_parts.append(f"color: {Theme.HIT_TEXT};")
                    base_style_parts.append(f"border: 1px solid {Theme.HIT_BORDER};")
                    text = "H"

                # Optional heat overlay (only on unknown cells)
                if show_heat and shot_state == NO_SHOT and total_count > 0:
                    text = str(int(round(total_count)))
                    if max_count > 0:
                        alpha = total_count / max_count
                        border_w = 1 + int(round(alpha * 2))
                        base_style_parts = [
                            f"background-color: {Theme.BG_DARK};",
                            f"color: {Theme.HEAT_TEXT};",
                            f"border: {border_w}px solid {Theme.HEAT_BORDER};",
                        ]

                # Recommended layout highlight
                if self.recommended_mask:
                    idx = cell_index(r, c)
                    if (self.recommended_mask >> idx) & 1:
                        base_style_parts.append(f"border: 2px solid {Theme.HIGHLIGHT};")

                btn.setStyleSheet("".join(base_style_parts))
                btn.setText(text)


def apply_dark_palette(app: QtWidgets.QApplication):
    """Apply a consistent dark theme using the Theme color palette."""
    QtWidgets.QApplication.setStyle("Fusion")
    palette = QtGui.QPalette()

    # Core backgrounds
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(Theme.BG_DARK))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(Theme.BG_DARK))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(Theme.BG_PANEL))

    # Text colors
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(Theme.TEXT_MAIN))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor(Theme.TEXT_MAIN))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(Theme.TEXT_MAIN))

    # Buttons
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(Theme.BG_BUTTON))

    # Links & selection
    palette.setColor(QtGui.QPalette.Link, QtGui.QColor(Theme.LINK))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(Theme.HIGHLIGHT))
    palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)

    # Tooltips / bright text
    palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)

    app.setPalette(palette)


class ModelStatsTab(QtWidgets.QWidget):
    STATE_PATH = "battleship_model_stats.json"

    def __init__(self, placements: Dict[str, List[Placement]], parent=None):
        super().__init__(parent)
        self.placements = placements

        # Strategy definitions

        self.model_defs = [
            {
                "key": "random",
                "name": "Random",
                "description": "Chooses uniformly among unknown cells.",
                "notes": "A sanity-check baseline. Useful to confirm your world-sampling and UI plumbing work; performance should be the worst."
            },
            {
                "key": "greedy",
                "name": "Greedy (Probability)",
                "description": "Shoots the cell with the highest immediate hit probability.",
                "notes": "Strong finisher once the posterior is peaked, but can be myopic in open-board hunt mode."
            },
            {
                "key": "entropy1",
                "name": "Info Gain (1-ply)",
                "description": "Shoots to maximize expected information gain (entropy reduction).",
                "notes": "Great at hunting because it values learning. Can be slightly slower than Greedy at finishing once a ship is basically found."
            },
            {
                "key": "weighted_sample",
                "name": "Weighted Sample",
                "description": "Randomly samples among cells proportional to their hit probability.",
                "notes": "Adds controlled randomness while still respecting the posterior. If all probabilities are 0, it falls back to uniform random."
            },
            {
                "key": "softmax_greedy",
                "name": "Softmax (Stochastic)",
                "description": "Selects shots stochastically via a softmax over probabilities (temperature-controlled).",
                "notes": "Temperature → 0 behaves like Greedy; higher temperature increases exploration. Useful for avoiding deterministic patterns."
            },
            {
                "key": "parity_greedy",
                "name": "Parity Greedy",
                "description": "Chooses Greedy, but constrained to the parity color with higher total probability mass.",
                "notes": "A parity-flavored variant of Greedy. Helps in hunt mode when minimum ship length makes parity efficient."
            },
            {
                "key": "random_checkerboard",
                "name": "Random Checkerboard",
                "description": "Randomly hunts only one checkerboard color (parity).",
                "notes": "Simple parity baseline. If your smallest ship length is 2+, parity typically beats pure random."
            },
            {
                "key": "systematic_checkerboard",
                "name": "Systematic Checkerboard",
                "description": "Deterministically hunts checkerboard cells in row-major order.",
                "notes": "A deterministic parity sweep. Good for debugging and reproducibility; not always optimal vs posterior-driven methods."
            },
            {
                "key": "diagonal_stripe",
                "name": "Diagonal Stripe",
                "description": "Hunts using diagonal stripes (mod patterns) before relaxing to a wider set.",
                "notes": "Another coverage heuristic. Can reduce redundant coverage early but is typically weaker than posterior-based hunt strategies."
            },
            {
                "key": "dynamic_parity",
                "name": "Dynamic Parity",
                "description": "Adapts its parity step based on remaining ships (e.g., step=3 when only length-3 remains).",
                "notes": "A practical endgame accelerator. If only a length-3 ship remains, checking every 3rd cell can reduce wasted shots."
            },
            {
                "key": "hybrid_phase",
                "name": "Hybrid (Hunt/Target)",
                "description": "Uses Info Gain to hunt, switches to Greedy when in target mode.",
                "notes": "A balanced default: information-seeking early, ruthless finishing once the posterior spikes."
            },
            {
                "key": "endpoint_phase",
                "name": "Endpoint Targeter",
                "description": "Uses Info Gain to hunt; when targeting, prefers endpoints of aligned hit clusters.",
                "notes": "Adds geometric intuition: if hits line up, shoot the ends; if they clump, shoot the frontier. Tunable weights let you trade off geometry vs posterior probability."
            },
            {
                "key": "center_weighted",
                "name": "Center-Weighted Greedy",
                "description": "Greedy probability, but slightly favors central cells (distance penalty).",
                "notes": "A mild bias that can help in symmetric boards where placements are roughly uniform, but it can hurt if your placement prior is not center-heavy."
            },
            {
                "key": "adaptive_skew",
                "name": "Adaptive Skew",
                "description": "Greedy probability with an early-game center penalty that fades later.",
                "notes": "Tries to avoid edge-chasing when the board is mostly unknown, then becomes closer to Greedy as information accumulates."
            },
            {
                "key": "thompson_world",
                "name": "Thompson Sampling",
                "description": "Samples one consistent world and plays as if it is true (then resamples next turn).",
                "notes": "Naturally balances exploration/exploitation. Often very strong without explicit hunt/target heuristics."
            },
            {
                "key": "two_ply",
                "name": "Info Gain (2-ply)",
                "description": "Looks one step ahead to maximize future information gain.",
                "notes": "Much slower. It performs an internal lookahead evaluation, so use smaller batches or only for analysis."
            },
        ]

        self.model_stats: Dict[str, Dict[str, object]] = {}
        self.param_sweeps: Dict[str, List[Dict[str, object]]] = {}
        self._ensure_all_models()

        self._sim_thread: Optional[QtCore.QThread] = None
        self._sim_worker: Optional[SimulationWorker] = None

        self._build_ui()
        self.load_state()
        self.refresh_table()
        self.update_summary_label()

    # ---------------- UI ----------------

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Controls row
        controls = QtWidgets.QHBoxLayout()
        layout.addLayout(controls)

        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItem("All models")
        for md in self.model_defs:
            self.model_combo.addItem(md["name"])
        controls.addWidget(QtWidgets.QLabel("Model:"))
        controls.addWidget(self.model_combo)

        self.games_spin = QtWidgets.QSpinBox()
        self.games_spin.setRange(10, 100000)
        self.games_spin.setValue(500)
        self.games_spin.setSingleStep(100)
        controls.addWidget(QtWidgets.QLabel("Games per model:"))
        controls.addWidget(self.games_spin)

        self.run_button = QtWidgets.QPushButton("Run Simulations")
        self.run_button.clicked.connect(self.run_simulations)
        controls.addWidget(self.run_button)
        controls.addStretch(1)

        # Progress bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Table
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(
            [
                "Model",
                "Games",
                "Avg Shots",
                "Std Dev",
                "Min",
                "Max",
                "Notes",
            ]
        )
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.cellDoubleClicked.connect(self.open_model_details)
        layout.addWidget(self.table)

        # Summary label
        self.summary_label = QtWidgets.QLabel()
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

    # ---------------- State helpers ----------------

    def _ensure_all_models(self):
        for md in self.model_defs:
            key = md["key"]
            if key not in self.model_stats:
                self.model_stats[key] = {
                    "total_games": 0,
                    "total_shots": 0.0,
                    "sum_sq_shots": 0.0,
                    "min_shots": 0,
                    "max_shots": 0,
                    "hist": [0] * (BOARD_SIZE * BOARD_SIZE + 1),
                }

    def save_state(self, path: Optional[str] = None):
        if path is None:
            path = self.STATE_PATH
        data = {"model_stats": self.model_stats, "param_sweeps": self.param_sweeps}
        try:
            with open(path, "w") as f:
                json.dump(data, f)
        except OSError:
            pass

    def add_param_sweep(self, model_key: str, sweep_record: Dict[str, object]) -> None:
        self.param_sweeps.setdefault(model_key, [])
        self.param_sweeps[model_key].insert(0, sweep_record)
        self.save_state()

    def get_param_sweeps(self, model_key: str) -> List[Dict[str, object]]:
        sweeps = self.param_sweeps.get(model_key, [])
        return sweeps if isinstance(sweeps, list) else []

    def load_state(self, path: Optional[str] = None):
        if path is None:
            path = self.STATE_PATH
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return

        stats = data.get("model_stats")
        if isinstance(stats, dict):
            self.model_stats = stats
        self._ensure_all_models()

    # ---------------- Stats merge + table ----------------

    def _merge_model_stats(self, key: str, delta: Dict[str, object]):
        self._ensure_all_models()
        cur = self.model_stats[key]

        dg = int(delta.get("total_games", 0))
        if dg <= 0:
            return

        cur_g = int(cur.get("total_games", 0))
        cur_s = float(cur.get("total_shots", 0.0))
        cur_sq = float(cur.get("sum_sq_shots", 0.0))
        cur_min = int(cur.get("min_shots", 0))
        cur_max = int(cur.get("max_shots", 0))
        cur_hist = cur.get("hist")
        if not isinstance(cur_hist, list) or len(cur_hist) < BOARD_SIZE * BOARD_SIZE + 1:
            cur_hist = [0] * (BOARD_SIZE * BOARD_SIZE + 1)

        dg_s = float(delta.get("total_shots", 0.0))
        dg_sq = float(delta.get("sum_sq_shots", 0.0))
        dg_min = int(delta.get("min_shots", cur_min))
        dg_max = int(delta.get("max_shots", cur_max))
        dg_hist = delta.get("hist") or [0] * (BOARD_SIZE * BOARD_SIZE + 1)

        new_g = cur_g + dg
        new_s = cur_s + dg_s
        new_sq = cur_sq + dg_sq

        if cur_g == 0:
            new_min = dg_min
            new_max = dg_max
        else:
            new_min = min(cur_min, dg_min)
            new_max = max(cur_max, dg_max)

        new_hist = [0] * max(len(cur_hist), len(dg_hist))
        for i in range(len(new_hist)):
            c = cur_hist[i] if i < len(cur_hist) else 0
            d = dg_hist[i] if i < len(dg_hist) else 0
            new_hist[i] = c + d

        cur["total_games"] = new_g
        cur["total_shots"] = new_s
        cur["sum_sq_shots"] = new_sq
        cur["min_shots"] = new_min
        cur["max_shots"] = new_max
        cur["hist"] = new_hist

    def refresh_table(self):
        self.table.setRowCount(len(self.model_defs))

        for row, md in enumerate(self.model_defs):
            key = md["key"]
            stats = self.model_stats.get(key, {})

            games = int(stats.get("total_games", 0))
            mean = 0.0
            std = 0.0
            if games > 0:
                total = float(stats.get("total_shots", 0))
                sq = float(stats.get("sum_sq_shots", 0))
                mean = total / games
                var = max(0.0, sq / games - mean * mean)
                std = math.sqrt(var)

            # Column 0: model name (store key in UserRole so lookups never depend on visible text)
            item_name = QtWidgets.QTableWidgetItem(md["name"])
            item_name.setData(QtCore.Qt.UserRole, key)
            self.table.setItem(row, 0, item_name)

            # Numeric columns
            vals = [
                str(games),
                f"{mean:.2f}" if games > 0 else "-",
                f"{std:.2f}" if games > 0 else "-",
                str(stats.get("min_shots", "-")),
                str(stats.get("max_shots", "-")),
            ]
            for c, v in enumerate(vals, start=1):
                it = QtWidgets.QTableWidgetItem(v)
                it.setTextAlignment(QtCore.Qt.AlignCenter)
                self.table.setItem(row, c, it)

            # Notes column (short)
            note = md.get("notes") or md.get("description") or ""
            if len(note) > 120:
                note = note[:117] + "..."
            it_note = QtWidgets.QTableWidgetItem(note)
            it_note.setTextAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            self.table.setItem(row, 6, it_note)

        self.update_summary_label()

    def open_model_details(self, row, col):
        try:
            item = self.table.item(row, 0)
            if item is None:
                return

            key = item.data(QtCore.Qt.UserRole)
            if not key:
                return

            model_def = next((md for md in self.model_defs if md.get('key') == key), None)
            if not model_def:
                return

            dlg = ModelDetailDialog(
                model_def,
                self.model_stats.get(key, {}),
                self.placements,
                stats_tab=self,
                parent=self,
            )
            dlg.exec_()
        except Exception:
            import traceback
            tb = traceback.format_exc()
            try:
                QtWidgets.QMessageBox.critical(
                    self,
                    'Error',
                    'An error occurred while opening model details.\n\n' + tb,
                )
            except Exception:
                # If the GUI is in a bad state, fall back to stderr
                import sys
                print(tb, file=sys.stderr)

    def update_summary_label(self):
        # Highlight the best avg-shots model among those with enough games
        best_name = None
        best_mean = None

        for md in self.model_defs:
            key = md["key"]
            stats = self.model_stats.get(key, {})
            games = int(stats.get("total_games", 0))
            if games < 50:
                continue
            total_shots = float(stats.get("total_shots", 0.0))
            mean = total_shots / games if games > 0 else 0.0
            if best_mean is None or mean < best_mean:
                best_mean = mean
                best_name = md["name"]

        if best_name is None:
            self.summary_label.setText(
                "Run some simulations to compare model efficiency (shots to sink all ships)."
            )
        else:
            self.summary_label.setText(
                f"So far, the best average performance is from <b>{best_name}</b> "
                f"with about <b>{best_mean:.2f}</b> shots per game."
            )

    # ---------------- Simulation control (threaded) ----------------

    def run_simulations(self):
        if self._sim_thread is not None:
            QtWidgets.QMessageBox.information(self, "Busy", "Simulations running.")
            return

        idx = self.model_combo.currentIndex()
        target_games = self.games_spin.value()  # This is the "Goal" number

        # Determine which models to run
        if idx == 0:
            candidates = [md["key"] for md in self.model_defs]
        else:
            candidates = [self.model_defs[idx - 1]["key"]]

        # Calculate actual work needed (Incremental Simulation)
        work_order = {}  # key -> games_to_run
        total_jobs = 0

        for key in candidates:
            current_stats = self.model_stats.get(key, {})
            current_games = int(current_stats.get("total_games", 0))

            if current_games < target_games:
                needed = target_games - current_games
                work_order[key] = needed
                total_jobs += needed

        if total_jobs == 0:
            QtWidgets.QMessageBox.information(
                self, "Done",
                f"All selected models already have at least {target_games} games simulated."
            )
            return

        self.progress_bar.setRange(0, total_jobs)
        self.progress_bar.setValue(0)

        # Disable controls
        self.run_button.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.games_spin.setEnabled(False)

        self._sim_thread = QtCore.QThread(self)
        # Pass the calculated work_order instead of raw numbers
        self._sim_worker = SimulationWorker(work_order, self.placements)
        self._sim_worker.moveToThread(self._sim_thread)

        self._sim_thread.started.connect(self._sim_worker.run)
        self._sim_worker.progress.connect(self._on_worker_progress)
        self._sim_worker.finished.connect(self._on_worker_finished)
        self._sim_worker.finished.connect(self._sim_thread.quit)
        self._sim_worker.finished.connect(self._sim_worker.deleteLater)
        self._sim_thread.finished.connect(self._on_thread_finished)
        self._sim_thread.finished.connect(self._sim_thread.deleteLater)

        self._sim_thread.start()

    @QtCore.pyqtSlot(int, int)
    def _on_worker_progress(self, done: int, total: int):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(done)

    @QtCore.pyqtSlot(dict)
    def _on_worker_finished(self, delta_stats: Dict[str, Dict[str, object]]):
        for key, delta in delta_stats.items():
            self._merge_model_stats(key, delta)
        self.save_state()
        self.refresh_table()
        self.update_summary_label()

    @QtCore.pyqtSlot()
    def _on_thread_finished(self):
        self._sim_thread = None
        self._sim_worker = None
        self.run_button.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.games_spin.setEnabled(True)
        # leave progress bar at its final max
        self.progress_bar.setValue(self.progress_bar.maximum())


class SimulationWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(int, int)  # done, total
    finished = QtCore.pyqtSignal(dict)  # key -> delta stats

    def __init__(
            self,
            work_order: Dict[str, int],  # Map: "greedy" -> 500 more games
            placements: Dict[str, List[Placement]],
    ):
        super().__init__()
        self.work_order = work_order
        self.placements = placements

    @QtCore.pyqtSlot()
    def run(self):
        rng = random.Random()
        total_jobs = sum(self.work_order.values())
        if total_jobs == 0:
            self.finished.emit({})
            return

        # NEW: Update roughly every 1% or every 1 game, whichever is larger
        update_interval = max(1, total_jobs // 100)

        done = 0
        results: Dict[str, Dict[str, object]] = {}

        for key, num_to_run in self.work_order.items():
            total_games = 0
            total_shots = 0.0
            sum_sq = 0.0
            min_shots = 0
            max_shots = 0
            hist = [0] * (BOARD_SIZE * BOARD_SIZE + 1)

            for _ in range(num_to_run):
                shots = simulate_model_game(key, self.placements, rng)

                total_games += 1
                total_shots += shots
                sum_sq += shots * shots

                if total_games == 1:
                    min_shots = max_shots = shots
                else:
                    if shots < min_shots: min_shots = shots
                    if shots > max_shots: max_shots = shots

                if 0 <= shots < len(hist):
                    hist[shots] += 1
                else:
                    hist[-1] += 1

                done += 1
                # NEW: Smoother progress emission
                if done % update_interval == 0 or done == total_jobs:
                    self.progress.emit(done, total_jobs)

            results[key] = {
                "total_games": total_games,
                "total_shots": total_shots,
                "sum_sq_shots": sum_sq,
                "min_shots": min_shots,
                "max_shots": max_shots,
                "hist": hist,
            }

        self.finished.emit(results)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Battleship Solver – 2-ply Attack & Learned Defense")
        self.resize(1250, 700)

        self.stats = StatsTracker()

        tabs = QtWidgets.QTabWidget()
        self.attack_tab = AttackTab(self.stats)
        self.defense_tab = DefenseTab()
        self.model_tab = ModelStatsTab(PLACEMENTS)

        self.attack_tab.linked_defense_tab = self.defense_tab

        tabs.addTab(self.attack_tab, "You attack")
        tabs.addTab(self.defense_tab, "Opponent attacks you")
        tabs.addTab(self.model_tab, "Model stats")

        self.setCentralWidget(tabs)

    # We need to hook up the signal so AttackTab can see DefenseTab
    # Override the recompute connection in AttackTab
    def _connect_tabs(self):
        # When Attack tab recomputes, pass Defense tab
        self.attack_tab.recompute_button.clicked.disconnect()
        self.attack_tab.recompute_button.clicked.connect(
            lambda: self.attack_tab.recompute(self.defense_tab)
        )
        # Also trigger it on cell clicks?
        # A simpler way is to just set a reference:
        self.attack_tab.linked_defense_tab = self.defense_tab

    def closeEvent(self, event: QtGui.QCloseEvent):
        try:
            self.attack_tab.save_state()
        except Exception as e:
            print("Error saving attack state:", e)

        try:
            self.defense_tab.save_state()
        except Exception as e:
            print("Error saving defense state:", e)

        try:
            self.model_tab.save_state()
        except Exception as e:
            print("Error saving model stats:", e)

        try:
            self.stats.save()
        except Exception as e:
            print("Error saving stats:", e)

        event.accept()


class CustomSimWorker(QtCore.QThread):
    """Runs a custom simulation batch off the UI thread."""
    progress = QtCore.pyqtSignal(int)
    result = QtCore.pyqtSignal(float, float, int)  # avg, std, count
    error = QtCore.pyqtSignal(str)

    def __init__(self, strategy: str, placements, count: int, params: dict):
        super().__init__()
        self.strategy = strategy
        self.placements = placements
        self.count = int(count)
        self.params = dict(params or {})
        self.is_cancelled = False

    def cancel(self):
        self.is_cancelled = True

    def run(self):
        try:
            rng = random.Random()
            total_shots = 0
            sum_sq = 0
            ran = 0

            for _ in range(self.count):
                if self.is_cancelled:
                    break

                shots = simulate_model_game(
                    self.strategy,
                    self.placements,
                    rng=rng,
                    params=self.params,
                )
                total_shots += shots
                sum_sq += shots * shots
                ran += 1

                # Emit progress occasionally to reduce signal overhead.
                if ran % 5 == 0 or ran == self.count:
                    self.progress.emit(ran)

            if ran > 0:
                avg = total_shots / ran
                var = max(0.0, sum_sq / ran - avg * avg)
                std = math.sqrt(var)
                self.result.emit(avg, std, ran)
            else:
                self.result.emit(0.0, 0.0, 0)
        except Exception:
            import traceback as _tb
            self.error.emit(_tb.format_exc())


def _sweep_values(min_v: float, max_v: float, step: float) -> List[float]:
    """Inclusive float range helper for parameter sweeps."""
    if step <= 0:
        return [min_v]
    if max_v < min_v:
        min_v, max_v = max_v, min_v
    out: List[float] = []
    # Guard against float drift
    v = float(min_v)
    # Include endpoint with a small epsilon
    eps = abs(step) * 1.0e-9 + 1.0e-12
    while v <= max_v + eps:
        out.append(float(v))
        v += step
    # Snap last value exactly to max if we're within epsilon
    if out and abs(out[-1] - max_v) <= eps:
        out[-1] = float(max_v)
    return out


class ParamSweepWorker(QtCore.QThread):
    """Runs a parameter sweep for a given strategy off the UI thread.

    Notes:
      - Do NOT override QThread.finished; use `result` for returning data.

    Emits:
      - progress(int): number of sweep points completed
      - result(object, int): (results_list, points_completed)
      - error(str): traceback text
    """
    progress = QtCore.pyqtSignal(int)
    result = QtCore.pyqtSignal(object, int)
    error = QtCore.pyqtSignal(str)

    def __init__(self, strategy: str, placements, games_per: int, param_grid: List[Dict[str, float]]):
        super().__init__()
        self.strategy = strategy
        self.placements = placements
        self.games_per = int(games_per)
        self.param_grid = list(param_grid)
        self._cancelled = False
        self.was_cancelled = False
        self.results = []
        self.points_completed = 0

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            self._run_impl()
        except Exception:
            import traceback as _tb
            self.error.emit(_tb.format_exc())

    def _run_impl(self):
        rng = random.Random()
        results: List[Dict[str, object]] = []
        total_cells = BOARD_SIZE * BOARD_SIZE
        points_completed = 0

        for sweep_idx, params in enumerate(self.param_grid, start=1):
            if self._cancelled:
                self.was_cancelled = True
                break

            total_shots = 0
            sum_sq = 0
            ran = 0
            min_shots = None
            max_shots = None
            hist = [0] * (total_cells + 1)

            # Run the batch for this parameter point
            for _ in range(self.games_per):
                if self._cancelled:
                    self.was_cancelled = True
                    break

                shots = simulate_model_game(self.strategy, self.placements, rng=rng, params=params)
                total_shots += shots
                sum_sq += shots * shots
                ran += 1

                if min_shots is None or shots < min_shots:
                    min_shots = shots
                if max_shots is None or shots > max_shots:
                    max_shots = shots

                if 0 <= shots <= total_cells:
                    hist[shots] += 1

            if ran > 0:
                avg = total_shots / ran
                var = max(0.0, (sum_sq / ran) - (avg * avg))
                std = math.sqrt(var)
                results.append({
                    "params": dict(params),
                    "games": ran,
                    "avg": avg,
                    "std": std,
                    "min": int(min_shots) if min_shots is not None else 0,
                    "max": int(max_shots) if max_shots is not None else 0,
                    "hist": hist,
                })
                points_completed += 1

            # progress per sweep point (not per game)
            self.progress.emit(sweep_idx)

            if self.was_cancelled:
                break

        self.results = results
        self.points_completed = points_completed
        # Emit results last; consumer should NOT delete the thread object here.
        self.result.emit(results, points_completed)


class SavedSweepsDialog(QtWidgets.QDialog):
    def __init__(self, model_def: Dict[str, object], sweeps: List[Dict[str, object]], parent=None):
        super().__init__(parent)
        self.model_def = model_def
        self.sweeps = sweeps[:] if isinstance(sweeps, list) else []
        self.setWindowTitle(f"Saved Sweeps: {model_def.get('name', model_def.get('key', 'Model'))}")
        self.resize(900, 520)
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        if not self.sweeps:
            layout.addWidget(QtWidgets.QLabel("No saved sweeps for this model yet."))
            btn_close = QtWidgets.QPushButton("Close")
            btn_close.clicked.connect(self.accept)
            layout.addWidget(btn_close, alignment=QtCore.Qt.AlignRight)
            return

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["Name", "Date", "Games/pt", "Grid", "Best Avg", "Ranges"])
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(5, QtWidgets.QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        self.table.setRowCount(len(self.sweeps))
        for r, sw in enumerate(self.sweeps):
            name = sw.get("name") or "(unnamed sweep)"
            created = sw.get("created_at") or ""
            gpp = sw.get("games_per_point", "")
            grid = sw.get("grid_size", "")
            best = sw.get("best", {}) or {}
            best_avg = best.get("avg", "")
            ranges = sw.get("ranges", {}) or {}

            it0 = QtWidgets.QTableWidgetItem(str(name))
            it0.setData(QtCore.Qt.UserRole, sw)
            self.table.setItem(r, 0, it0)

            self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(str(created)))
            self.table.setItem(r, 2, QtWidgets.QTableWidgetItem(str(gpp)))
            self.table.setItem(r, 3, QtWidgets.QTableWidgetItem(str(grid)))

            try:
                self.table.setItem(r, 4, QtWidgets.QTableWidgetItem("" if best_avg == "" else f"{float(best_avg):.2f}"))
            except Exception:
                self.table.setItem(r, 4, QtWidgets.QTableWidgetItem(str(best_avg)))

            self.table.setItem(r, 5, QtWidgets.QTableWidgetItem(json.dumps(ranges, sort_keys=True)))

        self.table.cellDoubleClicked.connect(self.accept)
        layout.addWidget(self.table)

        hint = QtWidgets.QLabel("Select a sweep and press OK (or double-click) to open it.")
        hint.setStyleSheet(f"color: {Theme.TEXT_MUTED};")
        layout.addWidget(hint)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def selected_sweep(self) -> Optional[Dict[str, object]]:
        if not hasattr(self, "table"):
            return None
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        if not item:
            return None
        sw = item.data(QtCore.Qt.UserRole)
        return sw if isinstance(sw, dict) else None


class SweepResultsDialog(QtWidgets.QDialog):
    def __init__(
            self,
            results: List[Dict[str, object]],
            best: Dict[str, object],
            model_def: Optional[Dict[str, object]] = None,
            sweep_meta: Optional[Dict[str, object]] = None,
            stats_tab: Optional["ModelStatsTab"] = None,
            parent=None,
            view_only: bool = False,
    ):
        super().__init__(parent)
        self.results = results or []
        self.best = best or {}
        self.model_def = model_def or {}
        self.sweep_meta = sweep_meta or {}
        self.stats_tab = stats_tab
        self.view_only = view_only

        title_model = self.model_def.get("name") or self.model_def.get("key") or "Model"
        self.setWindowTitle(f"Parameter Sweep Results — {title_model}")
        self.resize(1100, 650)
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        header = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel("Sweep Summary")
        f = lbl.font()
        f.setPointSize(14)
        f.setBold(True)
        lbl.setFont(f)
        header.addWidget(lbl)
        header.addStretch()

        self.btn_save = QtWidgets.QPushButton("Save Sweep")
        can_save = (not self.view_only) and (self.stats_tab is not None) and bool(self.sweep_meta)
        self.btn_save.setEnabled(bool(can_save))
        self.btn_save.clicked.connect(self.save_sweep)
        header.addWidget(self.btn_save)

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        header.addWidget(btn_close)
        layout.addLayout(header)

        best_avg = self.best.get("avg", None)
        best_params = self.best.get("params", {})

        if isinstance(best_avg, (int, float)):
            summary_txt = (
                f"Best Average Shots: <b>{float(best_avg):.2f}</b>"
                f"   |   Best Params: <b>{json.dumps(best_params)}</b>"
            )
        else:
            summary_txt = f"Best Params: <b>{json.dumps(best_params)}</b>"

        summary = QtWidgets.QLabel(summary_txt)
        summary.setTextFormat(QtCore.Qt.RichText)
        layout.addWidget(summary)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(split, stretch=1)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Params", "Avg", "Std", "Best", "Worst"])
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        self.table.setRowCount(len(self.results))
        for i, row in enumerate(self.results):
            params = row.get("params", {})
            avg = row.get("avg", 0.0)
            std = row.get("std", 0.0)
            min_s = row.get("min", 0)
            max_s = row.get("max", 0)

            it0 = QtWidgets.QTableWidgetItem(json.dumps(params, sort_keys=True))
            it0.setData(QtCore.Qt.UserRole, row.get("hist", []))
            self.table.setItem(i, 0, it0)

            self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{float(avg):.2f}"))
            self.table.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{float(std):.2f}"))
            self.table.setItem(i, 3, QtWidgets.QTableWidgetItem(str(min_s)))
            self.table.setItem(i, 4, QtWidgets.QTableWidgetItem(str(max_s)))

        self.table.cellClicked.connect(self._on_row_selected)
        left_layout.addWidget(self.table)
        split.addWidget(left)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.addWidget(QtWidgets.QLabel("Histogram (hover for exact counts):"))
        self.graph = StatsGraphWidget([])
        right_layout.addWidget(self.graph, stretch=1)
        split.addWidget(right)

        if self.results:
            self.table.selectRow(0)
            self._on_row_selected(0, 0)

    def _on_row_selected(self, row, col):
        item = self.table.item(row, 0)
        if not item:
            return
        hist = item.data(QtCore.Qt.UserRole)
        if not isinstance(hist, list):
            hist = []
        self.graph.set_counts(hist)

    def save_sweep(self):
        if self.view_only or self.stats_tab is None or not self.sweep_meta:
            return

        default_name = self.sweep_meta.get(
            "name") or f"{self.model_def.get('name', self.model_def.get('key', 'model'))} sweep"
        name, ok = QtWidgets.QInputDialog.getText(self, "Save Sweep", "Name:", text=str(default_name))
        if not ok:
            return
        name = str(name).strip() or default_name

        record = dict(self.sweep_meta)
        record["name"] = name
        record["results"] = self.results
        record["best"] = self.best
        record.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
        record.setdefault("id", datetime.now().strftime("%Y%m%d_%H%M%S"))

        model_key = str(record.get("model_key") or self.model_def.get("key") or "unknown")
        self.stats_tab.add_param_sweep(model_key, record)

        self.btn_save.setEnabled(False)
        self.btn_save.setText("Saved")


class ParamSweepDialog(QtWidgets.QDialog):
    """Configure a parameter sweep for a tunable model.

    Returns (games_per_setting, param_grid) where param_grid is a list[dict[str,float]].
    """

    MAX_TOTAL_CONFIGS = 20000  # safety cap to avoid accidental massive sweeps

    def __init__(self, model_key: str, model_name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Parameter Sweep: {model_name}")
        self.model_key = model_key
        self.inputs = {}  # key -> (min_sb, max_sb, step_sb)
        self.sb_games = None
        self.lbl_total = None
        self._total_configs = 0
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        specs = PARAM_SPECS.get(self.model_key, [])
        if not specs:
            layout.addWidget(QtWidgets.QLabel("This model has no tunable parameters to sweep."))
            btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
            btns.rejected.connect(self.reject)
            btns.accepted.connect(self.accept)
            layout.addWidget(btns)
            return

        # --- Grid for consistent label/input alignment ---
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        grid.setColumnMinimumWidth(0, 190)
        grid.setColumnMinimumWidth(1, 110)
        grid.setColumnMinimumWidth(2, 110)
        grid.setColumnMinimumWidth(3, 110)
        grid.setColumnMinimumWidth(4, 120)

        r = 0
        lbl_games = QtWidgets.QLabel("Games per setting:")
        self.sb_games = QtWidgets.QSpinBox()
        self.sb_games.setRange(1, 10000)
        self.sb_games.setValue(200)

        grid.addWidget(lbl_games, r, 0)
        grid.addWidget(self.sb_games, r, 1)
        self.lbl_total = QtWidgets.QLabel("")
        self.lbl_total.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        grid.addWidget(self.lbl_total, r, 2, 1, 3)

        r += 1
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setStyleSheet(f"color: {Theme.TEXT_LABEL};")
        grid.addWidget(line, r, 0, 1, 5)

        r += 1
        # Header row
        grid.addWidget(QtWidgets.QLabel(""), r, 0)
        hdr_min = QtWidgets.QLabel("Min")
        hdr_max = QtWidgets.QLabel("Max")
        hdr_step = QtWidgets.QLabel("Step")
        hdr_cnt = QtWidgets.QLabel("Configs")
        for w in (hdr_min, hdr_max, hdr_step, hdr_cnt):
            w.setStyleSheet(f"color: {Theme.TEXT_LABEL}; font-weight: bold;")
        grid.addWidget(hdr_min, r, 1)
        grid.addWidget(hdr_max, r, 2)
        grid.addWidget(hdr_step, r, 3)
        grid.addWidget(hdr_cnt, r, 4)

        # Param rows
        for spec in specs:
            r += 1
            key = spec["key"]
            label = spec.get("label", key)

            lbl = QtWidgets.QLabel(label)
            lbl.setStyleSheet(f"color: {Theme.TEXT_LABEL};")
            grid.addWidget(lbl, r, 0)

            # Use int spinboxes when values look integral and the step is integral.
            # Otherwise use double spinboxes.
            is_int = (
                    float(spec.get("min", 0)).is_integer()
                    and float(spec.get("max", 0)).is_integer()
                    and float(spec.get("step", 1)).is_integer()
            )

            if is_int:
                sb_min = QtWidgets.QSpinBox()
                sb_max = QtWidgets.QSpinBox()
                sb_step = QtWidgets.QSpinBox()
                sb_min.setRange(int(spec["min"]), int(spec["max"]))
                sb_max.setRange(int(spec["min"]), int(spec["max"]))
                sb_step.setRange(1, max(1, int(spec["max"]) - int(spec["min"])))
                sb_min.setValue(int(spec["min"]))
                sb_max.setValue(int(spec["max"]))
                sb_step.setValue(max(1, int(spec.get("step", 1))))
            else:
                sb_min = QtWidgets.QDoubleSpinBox()
                sb_max = QtWidgets.QDoubleSpinBox()
                sb_step = QtWidgets.QDoubleSpinBox()
                for sb in (sb_min, sb_max, sb_step):
                    sb.setDecimals(6)
                sb_min.setRange(float(spec["min"]), float(spec["max"]))
                sb_max.setRange(float(spec["min"]), float(spec["max"]))
                sb_step.setRange(1e-9, max(1e-9, float(spec["max"]) - float(spec["min"])))
                sb_min.setValue(float(spec["min"]))
                sb_max.setValue(float(spec["max"]))
                sb_step.setValue(float(spec.get("step", 0.1)))

            sb_min.valueChanged.connect(self._update_counts)
            sb_max.valueChanged.connect(self._update_counts)
            sb_step.valueChanged.connect(self._update_counts)

            grid.addWidget(sb_min, r, 1)
            grid.addWidget(sb_max, r, 2)
            grid.addWidget(sb_step, r, 3)

            lbl_count = QtWidgets.QLabel("0")
            lbl_count.setAlignment(QtCore.Qt.AlignCenter)
            grid.addWidget(lbl_count, r, 4)

            self.inputs[key] = (sb_min, sb_max, sb_step, lbl_count, is_int)

        layout.addLayout(grid)

        hint = QtWidgets.QLabel(
            "Tip: keep total configs under the cap to avoid long runs. "
            "You can increase Games per setting once you narrow in."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {Theme.TEXT_LABEL}; margin-top: 8px;")
        layout.addWidget(hint)

        layout.addStretch()

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        self._update_counts()

    @staticmethod
    def _count_for_range(min_v: float, max_v: float, step: float) -> int:
        """Count how many values will be generated for the sweep (inclusive endpoints)."""
        if step <= 0:
            return 0
        lo = min(min_v, max_v)
        hi = max(min_v, max_v)
        # Inclusive count with some float tolerance
        n = int(math.floor((hi - lo) / step + 1e-12)) + 1
        return max(0, n)

    @staticmethod
    def _sync_limits(self, key: str) -> None:
        widgets = self.param_widgets.get(key)
        if not widgets:
            return
        sb_min, sb_max, sb_step = widgets

        if sb_min.value() > sb_max.value():
            sb_max.setValue(sb_min.value())

        sb_max.setMinimum(sb_min.value())
        sb_min.setMaximum(sb_max.value())

        span = max(0.0, sb_max.value() - sb_min.value())
        if span <= 0:
            return
        sb_step.setMaximum(max(sb_step.minimum(), span))
        if sb_step.value() > span:
            sb_step.setValue(span)

    def _values_for_range(self, min_v: float, max_v: float, step: float, is_int: bool) -> list:
        if step <= 0:
            return []
        lo = min(min_v, max_v)
        hi = max(min_v, max_v)

        if is_int:
            lo_i = int(round(lo))
            hi_i = int(round(hi))
            st_i = max(1, int(round(step)))
            return list(range(lo_i, hi_i + 1, st_i))

        # Float stepping: cap length for safety
        out = []
        v = lo
        cap = 5000
        # Add a tiny epsilon to include hi when within tolerance
        while v <= hi + 1e-12 and len(out) < cap:
            out.append(float(round(v, 10)))
            v += step
        return out

    def _update_counts(self):
        total = 1
        # Update per-param counts
        for key, (sb_min, sb_max, sb_step, lbl_count, is_int) in self.inputs.items():
            min_v = float(sb_min.value())
            max_v = float(sb_max.value())
            step = float(sb_step.value())
            n = self._count_for_range(min_v, max_v, step)
            lbl_count.setText(str(n))
            total *= max(1, n)

        self._total_configs = total
        if self.lbl_total is not None:
            if total > self.MAX_TOTAL_CONFIGS:
                self.lbl_total.setText(
                    f"Total configs: {total:,} (over cap {self.MAX_TOTAL_CONFIGS:,})"
                )
                self.lbl_total.setStyleSheet("color: #ff6666; font-weight: bold;")
            else:
                self.lbl_total.setText(f"Total configs: {total:,}")
                self.lbl_total.setStyleSheet(f"color: {Theme.HIGHLIGHT}; font-weight: bold;")

    def accept(self):
        try:
            self._accept_impl()
        except Exception:
            import traceback as _tb
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("Sweep Config Error")
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("An error occurred while validating the sweep settings.")
            msg.setDetailedText(_tb.format_exc())
            msg.exec_()

    def _accept_impl(self):
        # Recompute (in case user never changed focus)
        self._update_counts()

        if self._total_configs > self.MAX_TOTAL_CONFIGS:
            QtWidgets.QMessageBox.warning(
                self,
                "Sweep Too Large",
                f"This sweep would run {self._total_configs:,} configurations.\n\nPlease reduce the ranges or increase the step size so the total is <= {self.MAX_TOTAL_CONFIGS:,}.",
            )
            return

        super().accept()

    def get_config(self):
        games = int(self.sb_games.value()) if self.sb_games is not None else 200

        # Build a list of values for each key
        value_lists = []
        keys = []
        for key, (sb_min, sb_max, sb_step, _lbl_count, is_int) in self.inputs.items():
            min_v = float(sb_min.value())
            max_v = float(sb_max.value())
            step = float(sb_step.value())
            values = self._values_for_range(min_v, max_v, step, is_int)
            # Ensure at least one value
            if not values:
                values = [min_v]
            keys.append(key)
            value_lists.append(values)

        grid = []
        for combo in itertools.product(*value_lists):
            grid.append({k: float(v) for k, v in zip(keys, combo)})

        return games, grid


class ParamTestDialog(QtWidgets.QDialog):
    def __init__(self, model_key: str, model_name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Tune: {model_name}")
        self.model_key = model_key
        self.inputs: Dict[str, QtWidgets.QDoubleSpinBox] = {}
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Games input
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Games to Run:"))
        self.sb_games = QtWidgets.QSpinBox()
        self.sb_games.setRange(1, 10000)
        self.sb_games.setValue(200)
        row.addWidget(self.sb_games)
        layout.addLayout(row)

        layout.addSpacing(10)
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setStyleSheet(f"color: {Theme.TEXT_MUTED};")
        layout.addWidget(line)
        layout.addSpacing(10)

        specs = PARAM_SPECS.get(self.model_key, [])

        if specs:
            for spec in specs:
                row = QtWidgets.QHBoxLayout()
                row.addWidget(QtWidgets.QLabel(spec["label"]))
                sb = QtWidgets.QDoubleSpinBox()
                sb.setRange(float(spec["min"]), float(spec["max"]))
                sb.setSingleStep(float(spec["step"]))
                sb.setValue(float(spec["default"]))
                sb.setDecimals(3)
                row.addWidget(sb)
                layout.addLayout(row)
                self.inputs[spec["key"]] = sb
        else:
            layout.addWidget(QtWidgets.QLabel("No tunable parameters for this model."))
            layout.addWidget(QtWidgets.QLabel("(Running a custom batch with default behavior.)"))

        layout.addStretch()

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_config(self) -> Tuple[int, Dict[str, float]]:
        p: Dict[str, float] = {k: sb.value() for k, sb in self.inputs.items()}
        return int(self.sb_games.value()), p


class ModelDetailDialog(QtWidgets.QDialog):
    def __init__(self, model_def: Dict, stats: Dict[str, object], placements,
                 stats_tab: Optional['ModelStatsTab'] = None, parent=None):
        super().__init__(parent)
        self.model_def = model_def
        self.stats = stats
        self.placements = placements
        self.stats_tab = stats_tab
        self.worker: Optional[CustomSimWorker] = None
        self.progress: Optional[QtWidgets.QProgressDialog] = None

        self.setWindowTitle(f"Analysis: {model_def.get('name', model_def.get('key', 'Model'))}")
        self.resize(900, 650)
        self._build_ui()

    def _build_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)

        # Header
        title = QtWidgets.QLabel(self.model_def.get("name", self.model_def.get("key", "Model")))
        f = title.font()
        f.setPointSize(16)
        f.setBold(True)
        title.setFont(f)
        main_layout.addWidget(title)

        # Notes / Description
        notes_label = QtWidgets.QLabel("Model Notes:")
        notes_label.setStyleSheet(f"color: {Theme.TEXT_MUTED}; font-weight: bold; margin-top: 10px;")
        main_layout.addWidget(notes_label)

        full_text = self.model_def.get("description", "No description.")
        if self.model_def.get("notes"):
            full_text += "\n\n" + self.model_def["notes"]

        desc_box = QtWidgets.QTextEdit()
        desc_box.setReadOnly(True)
        desc_box.setPlainText(full_text)
        desc_box.setMaximumHeight(120)
        desc_box.setStyleSheet(
            f"background-color: {Theme.BG_PANEL}; border: 1px solid {Theme.BG_BUTTON}; color: {Theme.TEXT_MAIN};"
        )
        main_layout.addWidget(desc_box)

        # Stats & Graph
        mid_layout = QtWidgets.QHBoxLayout()

        # Left: metrics
        stats_panel = QtWidgets.QWidget()
        stats_panel.setFixedWidth(260)
        sp_layout = QtWidgets.QVBoxLayout(stats_panel)

        games = int(self.stats.get("total_games", 0))
        mean = 0.0
        std = 0.0
        if games > 0:
            total = float(self.stats.get("total_shots", 0))
            sq = float(self.stats.get("sum_sq_shots", 0))
            mean = total / games
            var = max(0.0, sq / games - mean * mean)
            std = math.sqrt(var)

        metrics = [
            ("Total Games", f"{games}"),
            ("Avg Shots", f"{mean:.2f}"),
            ("Std Dev", f"{std:.2f}"),
            ("Best Game", f"{self.stats.get('min_shots', '-')}"),
            ("Worst Game", f"{self.stats.get('max_shots', '-')}"),
        ]

        for label, val in metrics:
            row = QtWidgets.QHBoxLayout()
            lbl = QtWidgets.QLabel(label)
            lbl.setStyleSheet(f"color: {Theme.TEXT_MUTED}; font-weight: bold;")
            val_lbl = QtWidgets.QLabel(val)
            val_lbl.setAlignment(QtCore.Qt.AlignRight)
            val_lbl.setStyleSheet(f"color: {Theme.HIGHLIGHT}; font-weight: bold;")
            row.addWidget(lbl)
            row.addWidget(val_lbl)
            sp_layout.addLayout(row)

        sp_layout.addStretch()

        self.btn_custom = QtWidgets.QPushButton("Run Custom Sim")
        self.btn_custom.setStyleSheet(
            f"background-color: {Theme.HIGHLIGHT}; color: #000; font-weight: bold; padding: 8px;"
        )
        self.btn_custom.clicked.connect(self.run_custom_sim)
        sp_layout.addWidget(self.btn_custom)

        sweep_specs = PARAM_SPECS.get(self.model_def.get("key", ""), [])
        gb_sweep = QtWidgets.QGroupBox("Parameter Sweep")
        gb_sweep.setEnabled(bool(sweep_specs))
        gb_sweep.setStyleSheet(
            f"""
QGroupBox {{ color: {Theme.TEXT_LABEL}; }}
QGroupBox:disabled {{ color: {Theme.TEXT_MUTED}; }}
QLabel {{ color: {Theme.TEXT_LABEL}; }}
QLabel:disabled {{ color: {Theme.TEXT_MUTED}; }}
QPushButton {{ background-color: {Theme.BG_BUTTON}; color: {Theme.TEXT_MAIN}; padding: 8px; font-weight: bold; }}
QPushButton:disabled {{ background-color: {Theme.BG_PANEL}; color: {Theme.TEXT_MUTED}; font-weight: normal; }}
"""
        )
        gb_layout = QtWidgets.QVBoxLayout(gb_sweep)

        hint = QtWidgets.QLabel("Run a grid sweep over tunable parameters and compare average shots.")
        hint.setWordWrap(True)
        gb_layout.addWidget(hint)

        btn_sweep = QtWidgets.QPushButton("Run Parameter Sweep")
        btn_sweep.setEnabled(bool(sweep_specs))
        btn_sweep.clicked.connect(self.run_param_sweep)
        gb_layout.addWidget(btn_sweep)

        btn_saved = QtWidgets.QPushButton("View Saved Sweeps")
        has_saved = False
        try:
            if self.stats_tab is not None:
                has_saved = bool(self.stats_tab.get_param_sweeps(self.model_def.get("key", "")))
        except Exception:
            has_saved = False
        btn_saved.setEnabled(has_saved)
        btn_saved.clicked.connect(self.open_saved_sweeps)
        gb_layout.addWidget(btn_saved)
        self.btn_saved_sweeps = btn_saved

        sp_layout.addWidget(gb_sweep)
        self.btn_sweep = btn_sweep

        mid_layout.addWidget(stats_panel)

        # Right: histogram graph
        self.graph = StatsGraphWidget(self.stats.get("hist", []))
        mid_layout.addWidget(self.graph, stretch=1)

        main_layout.addLayout(mid_layout)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        main_layout.addWidget(close_btn, alignment=QtCore.Qt.AlignRight)

    def _refresh_saved_sweeps_button(self):
        if not hasattr(self, "btn_saved_sweeps"):
            return
        has_saved = False
        try:
            if self.stats_tab is not None:
                has_saved = bool(self.stats_tab.get_param_sweeps(self.model_def.get("key", "")))
        except Exception:
            has_saved = False
        self.btn_saved_sweeps.setEnabled(has_saved)

    def open_saved_sweeps(self):
        if self.stats_tab is None:
            return
        sweeps = self.stats_tab.get_param_sweeps(self.model_def.get("key", ""))
        dlg = SavedSweepsDialog(self.model_def, sweeps, self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        selected = dlg.selected_sweep()
        if not selected:
            return
        results = selected.get("results", []) or []
        best = selected.get("best", {}) or {}
        sr = SweepResultsDialog(
            results=results,
            best=best,
            model_def=self.model_def,
            sweep_meta=selected,
            stats_tab=self.stats_tab,
            parent=self,
            view_only=True,
        )
        sr.exec_()

    def _safe_progress_set(self, dlg, value: int) -> None:
        """Safely set QProgressDialog value (guards against deleted Qt objects)."""
        try:
            if dlg is None:
                return
            dlg.setValue(int(value))
        except Exception:
            return

    def _safe_progress_close(self, dlg) -> None:
        """Safely close a QProgressDialog/QProgressBar wrapper.

        Note: Closing a QProgressDialog can sometimes emit 'canceled'. We block signals while
        closing so a normal completion isn't mistaken for a user cancel.
        """
        try:
            if dlg is None:
                return
            try:
                prev = dlg.blockSignals(True)
            except Exception:
                prev = None
            try:
                try:
                    dlg.hide()
                except Exception:
                    pass
                try:
                    dlg.close()
                except Exception:
                    pass
            finally:
                try:
                    if prev is not None:
                        dlg.blockSignals(prev)
                except Exception:
                    pass
        except Exception:
            return

    def _on_custom_progress(self, value: int) -> None:
        self._safe_progress_set(getattr(self, "progress", None), value)

    def _on_sweep_progress(self, value: int) -> None:
        self._safe_progress_set(getattr(self, "sweep_progress", None), value)

    def run_custom_sim(self):
        dlg = ParamTestDialog(self.model_def["key"], self.model_def.get("name", self.model_def["key"]), self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        count, params = dlg.get_config()

        # Prevent multiple concurrent runs
        if getattr(self, "worker", None) is not None and self.worker.isRunning():
            return
        if hasattr(self, "btn_custom"):
            self.btn_custom.setEnabled(False)

        self.progress = QtWidgets.QProgressDialog("Running Experiment...", "Cancel", 0, count, self)
        self.progress.setWindowModality(QtCore.Qt.WindowModal)
        self.progress.setMinimumDuration(0)
        self.progress.setValue(0)
        self.progress.setAutoClose(False)
        self.progress.setAutoReset(False)

        self.worker = CustomSimWorker(self.model_def["key"], self.placements, count, params)
        self.worker.progress.connect(self._on_custom_progress)
        self.worker.result.connect(self.on_sim_finished)
        self.worker.error.connect(self.on_sim_error)
        self.worker.finished.connect(self._on_custom_thread_finished)
        self.progress.canceled.connect(self.worker.cancel)

        self.worker.start()

    def on_sim_finished(self, avg: float, std: float, count: int):
        """Handle completion of a custom experiment run."""
        try:
            # Close progress and re-enable UI safely
            try:
                self._safe_progress_close(getattr(self, "progress", None))
            except Exception:
                pass
            self.progress = None
            try:
                if hasattr(self, "btn_custom"):
                    self.btn_custom.setEnabled(True)
            except Exception:
                pass

            if count <= 0:
                return

            params = {}
            try:
                w = getattr(self, "worker", None) or getattr(self, "custom_worker", None)
                if w is not None:
                    params = dict(getattr(w, "params", {}) or {})
            except Exception:
                params = {}

            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("Experiment Results")
            msg.setText(
                f"<b>Model:</b> {self.model_def.get('name', self.model_def['key'])}<br>"
                f"<b>Params:</b> {json.dumps(params)}<br><br>"
                f"<b>Games Ran:</b> {count}<br>"
                f"<b>Average:</b> {avg:.2f}<br>"
                f"<b>Std Dev:</b> {std:.2f}"
            )
            msg.exec_()

            # If the user tried to close while workers were running, finish closing once they're done
            if getattr(self, "_closing_after_workers", False):
                try:
                    sweep_worker = getattr(self, "sweep_worker", None)
                    custom_worker = getattr(self, "worker", None) or getattr(self, "custom_worker", None)
                    sweep_running = bool(sweep_worker) and hasattr(sweep_worker,
                                                                   "isRunning") and sweep_worker.isRunning()
                    custom_running = bool(custom_worker) and hasattr(custom_worker,
                                                                     "isRunning") and custom_worker.isRunning()
                    if not (sweep_running or custom_running):
                        self._closing_after_workers = False
                        self.accept()
                except Exception:
                    pass
        except Exception as e:
            traceback.print_exc()
            try:
                QtWidgets.QMessageBox.critical(self, "Error", "Sim finished handler crashed: " + str(e))
            except Exception:
                pass

    def on_sim_error(self, msg: str) -> None:
        """Handle errors raised by the custom experiment worker."""
        try:
            self._safe_progress_close(getattr(self, "progress", None))
        except Exception:
            pass
        self.progress = None
        try:
            if hasattr(self, "btn_custom"):
                self.btn_custom.setEnabled(True)
        except Exception:
            pass
        QtWidgets.QMessageBox.critical(self, "Experiment error", msg)

        # If the user tried to close while workers were running, finish closing once they're done
        if getattr(self, "_closing_after_workers", False):
            try:
                sweep_worker = getattr(self, "sweep_worker", None)
                custom_worker = getattr(self, "worker", None) or getattr(self, "custom_worker", None)
                sweep_running = bool(sweep_worker) and hasattr(sweep_worker, "isRunning") and sweep_worker.isRunning()
                custom_running = bool(custom_worker) and hasattr(custom_worker,
                                                                 "isRunning") and custom_worker.isRunning()
                if not (sweep_running or custom_running):
                    self._closing_after_workers = False
                    self.accept()
            except Exception:
                pass

    def _on_custom_thread_finished(self):
        """Cleanup after the CustomSimWorker thread has fully stopped."""
        try:
            # Only clear references after the thread is actually finished
            if getattr(self, "worker", None) is not None and hasattr(self.worker,
                                                                     "isRunning") and not self.worker.isRunning():
                self.worker = None
        except Exception:
            pass
        try:
            if hasattr(self, "btn_custom"):
                self.btn_custom.setEnabled(True)
        except Exception:
            pass
        try:
            # If progress dialog somehow lingered, close it.
            self._safe_progress_close(getattr(self, "progress", None))
        except Exception:
            pass
        self.progress = None

        # If the user tried to close while workers were running, finish closing once they're done
        if getattr(self, "_closing_after_workers", False):
            try:
                sweep_worker = getattr(self, "sweep_worker", None)
                custom_worker = getattr(self, "worker", None) or getattr(self, "custom_worker", None)
                sweep_running = bool(sweep_worker) and hasattr(sweep_worker, "isRunning") and sweep_worker.isRunning()
                custom_running = bool(custom_worker) and hasattr(custom_worker,
                                                                 "isRunning") and custom_worker.isRunning()
                if not (sweep_running or custom_running):
                    self._closing_after_workers = False
                    self.accept()
            except Exception:
                pass

    def run_param_sweep(self):
        try:
            self._run_param_sweep_impl()
        except Exception:
            import traceback as _tb
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("Sweep Error")
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Failed to start the parameter sweep.")
            msg.setDetailedText(_tb.format_exc())
            msg.exec_()

    def _run_param_sweep_impl(self):
        # Prevent multiple concurrent sweeps
        if getattr(self, "sweep_worker", None) is not None and self.sweep_worker.isRunning():
            return
        specs = PARAM_SPECS.get(self.model_def.get("key", ""), [])
        if not specs:
            QtWidgets.QMessageBox.information(self, "No Parameters", "This model has no tunable parameters to sweep.")
            return

        dlg = ParamSweepDialog(self.model_def["key"], self.model_def["name"], self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        games_per, param_grid = dlg.get_config()
        self._last_sweep_games_per = int(games_per)
        if hasattr(self, "btn_sweep"):
            self.btn_sweep.setEnabled(False)
        # Track whether the user requested cancellation for this sweep
        self._sweep_cancel_requested = False
        ranges = {}
        try:
            for k, (sb_min, sb_max, sb_step) in dlg.param_widgets.items():
                ranges[str(k)] = {
                    "min": float(sb_min.value()),
                    "max": float(sb_max.value()),
                    "step": float(sb_step.value()),
                }
        except Exception:
            ranges = {}

        self._last_sweep_meta = {
            "id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "model_key": self.model_def.get("key"),
            "model_name": self.model_def.get("name"),
            "games_per_point": int(games_per),
            "grid_size": int(len(param_grid)),
            "ranges": ranges,
        }
        if not param_grid:
            return

        total_cfg = len(param_grid)
        self.sweep_progress = QtWidgets.QProgressDialog("Sweeping parameters...", "Cancel", 0, total_cfg, self)
        self.sweep_progress.setWindowModality(QtCore.Qt.WindowModal)
        self.sweep_progress.setMinimumDuration(0)
        self.sweep_progress.setValue(0)
        self.sweep_progress.setAutoClose(False)
        self.sweep_progress.setAutoReset(False)

        self.sweep_worker = ParamSweepWorker(self.model_def["key"], self.placements, games_per, param_grid)
        self.sweep_worker.progress.connect(self._on_sweep_progress)
        self.sweep_worker.result.connect(self.on_sweep_finished)
        self.sweep_worker.error.connect(self.on_sweep_error)
        self.sweep_worker.finished.connect(self._on_sweep_thread_finished)
        self.sweep_progress.canceled.connect(self._request_cancel_sweep)

        self.sweep_worker.start()

    def on_sweep_finished(self, results, ran: int):
        """Handle completion of a parameter sweep."""
        debug_event(self, "Sweep finished",
                    f"ran={ran} results={len(results) if results else 0} cancel={getattr(self, '_sweep_cancel_requested', False)}")

        try:
            # Close progress and re-enable UI safely
            try:
                self._safe_progress_close(getattr(self, "sweep_progress", None))
            except Exception:
                pass
            self.sweep_progress = None
            try:
                if hasattr(self, "btn_sweep"):
                    self.btn_sweep.setEnabled(True)
            except Exception:
                pass

            # If the user cancelled, stop here (don't pop more dialogs).
            if getattr(self, "_sweep_cancel_requested", False):
                debug_event(self, "Sweep finished", "Skipping results dialog because cancel flag is set.",
                            force_popup=True, level="warning")
                return

            # User may have cancelled immediately or produced no valid points
            if ran <= 0 or not results:
                debug_event(self, "Sweep finished", "No results to show (ran<=0 or empty results).", force_popup=True,
                            level="warning")
                return

            # Choose the best setting by lowest average shots (tie-breaker: std dev)
            best = {}
            try:
                candidates = [r for r in results if isinstance(r, dict)]

                def _key(r):
                    try:
                        avg = float(r.get('avg', math.inf))
                        std = float(r.get('std', math.inf))
                        if math.isnan(avg):
                            avg = math.inf
                        if math.isnan(std):
                            std = math.inf
                        return (avg, std)
                    except Exception:
                        return (math.inf, math.inf)

                if candidates:
                    best = min(candidates, key=_key)
            except Exception:
                best = {}

            debug_event(self, "Sweep results", "Opening SweepResultsDialog.")
            dlg = SweepResultsDialog(
                results=results,
                best=best,
                model_def=getattr(self, 'model_def', None),
                sweep_meta=getattr(self, '_last_sweep_meta', None),
                stats_tab=getattr(self, 'stats_tab', None),
                parent=self,
                view_only=False,
            )
            dlg.exec_()
            try:
                self._refresh_saved_sweeps_button()
            except Exception:
                pass

            # If the user tried to close while workers were running, finish closing once they're done
            if getattr(self, "_closing_after_workers", False):
                try:
                    sweep_worker = getattr(self, "sweep_worker", None)
                    custom_worker = getattr(self, "worker", None) or getattr(self, "custom_worker", None)
                    sweep_running = bool(sweep_worker) and hasattr(sweep_worker,
                                                                   "isRunning") and sweep_worker.isRunning()
                    custom_running = bool(custom_worker) and hasattr(custom_worker,
                                                                     "isRunning") and custom_worker.isRunning()
                    if not (sweep_running or custom_running):
                        self._closing_after_workers = False
                        self.accept()
                except Exception:
                    pass
        except Exception as e:
            traceback.print_exc()
            try:
                QtWidgets.QMessageBox.critical(self, "Error", "Sweep finished handler crashed: " + str(e))
            except Exception:
                pass
        finally:
            # reset cancel flag for next run
            try:
                self._sweep_cancel_requested = False
            except Exception:
                pass

    def on_sweep_error(self, msg: str):
        """Handle errors raised by the sweep worker."""
        debug_event(self, "Sweep error", msg, level="error")

        try:
            self._safe_progress_close(getattr(self, "sweep_progress", None))
        except Exception:
            pass
        self.sweep_progress = None
        try:
            if hasattr(self, "btn_sweep"):
                self.btn_sweep.setEnabled(True)
        except Exception:
            pass
        try:
            self._sweep_cancel_requested = False
        except Exception:
            pass
        QtWidgets.QMessageBox.critical(self, "Sweep error", msg)

        # If the user tried to close while workers were running, finish closing once they're done
        if getattr(self, "_closing_after_workers", False):
            try:
                sweep_worker = getattr(self, "sweep_worker", None)
                custom_worker = getattr(self, "worker", None) or getattr(self, "custom_worker", None)
                sweep_running = bool(sweep_worker) and hasattr(sweep_worker, "isRunning") and sweep_worker.isRunning()
                custom_running = bool(custom_worker) and hasattr(custom_worker,
                                                                 "isRunning") and custom_worker.isRunning()
                if not (sweep_running or custom_running):
                    self._closing_after_workers = False
                    self.accept()
            except Exception:
                pass

    def _on_sweep_thread_finished(self):
        """Cleanup after the ParamSweepWorker thread has fully stopped."""
        try:
            w = getattr(self, "sweep_worker", None)
            if w is not None and hasattr(w, "isRunning") and not w.isRunning():
                self.sweep_worker = None
        except Exception:
            pass
        try:
            if hasattr(self, "btn_sweep"):
                self.btn_sweep.setEnabled(True)
        except Exception:
            pass
        try:
            self._safe_progress_close(getattr(self, "sweep_progress", None))
        except Exception:
            pass
        self.sweep_progress = None

        # If the user tried to close while workers were running, finish closing once they're done
        if getattr(self, "_closing_after_workers", False):
            try:
                sweep_worker = getattr(self, "sweep_worker", None)
                custom_worker = getattr(self, "worker", None) or getattr(self, "custom_worker", None)
                sweep_running = bool(sweep_worker) and hasattr(sweep_worker, "isRunning") and sweep_worker.isRunning()
                custom_running = bool(custom_worker) and hasattr(custom_worker,
                                                                 "isRunning") and custom_worker.isRunning()
                if not (sweep_running or custom_running):
                    self._closing_after_workers = False
                    self.accept()
            except Exception:
                pass

    def _cancel_running_workers(self):
        """Request cancellation for running worker threads without dropping references prematurely."""
        for w_attr, p_attr in (
                ("sweep_worker", "sweep_progress"),
                ("worker", "progress"),
                ("custom_worker", "progress"),
        ):
            worker = getattr(self, w_attr, None)
            prog = getattr(self, p_attr, None)
            if worker is None:
                continue
            try:
                if hasattr(worker, "isRunning") and worker.isRunning() and hasattr(worker, "cancel"):
                    worker.cancel()
                    worker.wait(1500)
            except Exception:
                pass

            try:
                still_running = hasattr(worker, "isRunning") and worker.isRunning()
            except Exception:
                still_running = True
            if still_running:
                continue

            try:
                self._safe_progress_close(prog)
            except Exception:
                pass
            try:
                setattr(self, w_attr, None)
            except Exception:
                pass
            try:
                setattr(self, p_attr, None)
            except Exception:
                pass

    def _request_cancel_sweep(self):
        """Handle sweep cancellation from the progress dialog safely."""
        self._sweep_cancel_requested = True
        try:
            w = getattr(self, "sweep_worker", None)
            if w is not None and hasattr(w, "cancel"):
                w.cancel()
        except Exception:
            traceback.print_exc()
        try:
            p = getattr(self, "sweep_progress", None)
            if p is not None:
                btn = p.cancelButton()
                if btn is not None:
                    btn.setEnabled(False)
                p.setLabelText("Canceling… finishing current game")
        except Exception:
            pass

    def closeEvent(self, event):
        """Ensure background workers are cancelled safely before closing."""
        try:
            sweep_worker = getattr(self, "sweep_worker", None)
            custom_worker = getattr(self, "worker", None) or getattr(self, "custom_worker", None)
            sweep_running = bool(sweep_worker) and hasattr(sweep_worker, "isRunning") and sweep_worker.isRunning()
            custom_running = bool(custom_worker) and hasattr(custom_worker, "isRunning") and custom_worker.isRunning()

            if sweep_running or custom_running:
                setattr(self, "_closing_after_workers", True)
                try:
                    if sweep_running and hasattr(sweep_worker, "cancel"):
                        sweep_worker.cancel()
                except Exception:
                    pass
                try:
                    if custom_running and hasattr(custom_worker, "cancel"):
                        custom_worker.cancel()
                except Exception:
                    pass

                try:
                    sp = getattr(self, "sweep_progress", None)
                    if sp is not None:
                        btn = sp.cancelButton()
                        if btn is not None:
                            btn.setEnabled(False)
                        sp.setLabelText("Canceling… finishing current game")
                except Exception:
                    pass
                try:
                    p = getattr(self, "progress", None)
                    if p is not None:
                        btn = p.cancelButton()
                        if btn is not None:
                            btn.setEnabled(False)
                        p.setLabelText("Canceling… finishing current game")
                except Exception:
                    pass

                event.ignore()
                return
        except Exception:
            traceback.print_exc()

        super().closeEvent(event)


class StatsGraphWidget(QtWidgets.QWidget):
    def __init__(self, hist, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._hover_idx = None
        self._bar_hit_rects = []  # list[QtCore.QRectF]
        self.hist = []
        self.set_hist(hist)

    def set_hist(self, hist):
        self.hist = list(hist) if hist else []
        self._hover_idx = None
        self._bar_hit_rects = []
        self.update()

    # Backward compatibility: older parts of the UI call this method name.
    # The data are histogram counts indexed by "shots".
    def set_counts(self, counts):
        self.set_hist(counts)

    def leaveEvent(self, event):
        self._hover_idx = None
        QtWidgets.QToolTip.hideText()
        self.update()
        return super().leaveEvent(event)

    def mouseMoveEvent(self, event):
        if not self._bar_hit_rects:
            return super().mouseMoveEvent(event)

        pos = event.pos()
        hovered = None
        for i, rect in enumerate(self._bar_hit_rects):
            if rect.contains(pos):
                hovered = i
                break

        if hovered != self._hover_idx:
            self._hover_idx = hovered
            if hovered is not None and 0 <= hovered < len(self.hist):
                count = int(self.hist[hovered])
                total = max(1, sum(self.hist))
                pct = 100.0 * (count / total)
                shots = hovered
                QtWidgets.QToolTip.showText(
                    event.globalPos(),
                    f"Shots: {shots}\nCount: {count}\nShare: {pct:.1f}%"
                )
            else:
                QtWidgets.QToolTip.hideText()
            self.update()

        return super().mouseMoveEvent(event)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, False)

        rect = self.rect()
        painter.fillRect(rect, QtGui.QColor(Theme.BG_PANEL))

        if not self.hist:
            painter.setPen(QtGui.QColor(Theme.TEXT_LABEL))
            painter.drawText(rect, QtCore.Qt.AlignCenter, "No histogram data.")
            return

        total = sum(self.hist)
        max_count = max(self.hist) if self.hist else 0
        if max_count <= 0:
            painter.setPen(QtGui.QColor(Theme.TEXT_LABEL))
            painter.drawText(rect, QtCore.Qt.AlignCenter, "No samples yet.")
            return

        pad = 10
        w = max(1, rect.width() - 2 * pad)
        h = max(1, rect.height() - 2 * pad)
        n = len(self.hist)

        bar_w = w / max(1, n)

        base = QtGui.QColor(Theme.HIGHLIGHT)
        hover = base.lighter(135)
        outline = QtGui.QColor(Theme.BG_BUTTON)

        self._bar_hit_rects = []

        for i, count in enumerate(self.hist):
            x0 = pad + i * bar_w
            hit_rect = QtCore.QRectF(x0, pad, bar_w, h)
            self._bar_hit_rects.append(hit_rect)

            if count <= 0:
                continue

            frac = float(count) / float(max_count)
            bar_h = max(1.0, frac * h)
            y0 = pad + (h - bar_h)
            draw_rect = QtCore.QRectF(x0, y0, max(1.0, bar_w - 1.0), bar_h)

            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(hover if i == self._hover_idx else base)
            painter.drawRect(draw_rect)

            if i == self._hover_idx:
                painter.setPen(QtGui.QPen(outline, 1))
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.drawRect(draw_rect)

        # Border
        painter.setPen(QtGui.QPen(outline, 1))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawRect(QtCore.QRect(pad, pad, w, h))


def main():
    global DEBUG_ENABLED
    # Enable debug via flag or env var (BATTLESHIP_DEBUG=1)
    try:
        argv = list(sys.argv)
        if "--debug" in argv:
            DEBUG_ENABLED = True
            argv.remove("--debug")
        if os.environ.get("BATTLESHIP_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}:
            DEBUG_ENABLED = True
    except Exception:
        argv = sys.argv

    app = QtWidgets.QApplication(argv)
    apply_dark_palette(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()