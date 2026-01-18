import os
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

from battleship.domain.board import cell_index
from battleship.domain.config import BOARD_SIZE, EMPTY, HIT, MISS, WORLD_SAMPLE_TARGET
from battleship.domain.phase import PHASE_ENDGAME, PHASE_HUNT, PHASE_TARGET, classify_phase
from battleship.domain.worlds import sample_worlds
from battleship.strategies.selection import Posterior, _choose_next_shot_for_strategy


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(str(raw).strip())
        return value if value > 0 else default
    except ValueError:
        return default


def _target_worlds_for_strategy(strategy: str, base_target: int) -> int:
    adaptive = _env_flag("SIM_ADAPTIVE_TARGETS", True)
    if not adaptive:
        return base_target
    heavy = {"rollout_mcts", "two_ply"}
    medium = {"entropy1", "hybrid_phase", "endpoint_phase", "thompson_world"}
    if strategy in heavy:
        return base_target
    if strategy in medium:
        return max(2000, min(base_target, 8000))
    return max(1000, min(base_target, 5000))


class SimProfiler:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.games = 0
        self.shots = 0
        self.layout_time = 0.0
        self.sample_time = 0.0
        self.sample_calls = 0
        self.topups = 0
        self.filter_time = 0.0
        self.filter_events = 0
        self.worlds_survived = 0
        self.selection_time = 0.0
        self.selection_ex_sample = 0.0
        self._last_sample_time = 0.0

    def record_layout_time(self, dt: float) -> None:
        self.layout_time += float(dt)

    def record_sample(self, dt: float, n_worlds: int, topup: bool = False) -> None:
        self.sample_time += float(dt)
        self.sample_calls += 1
        if topup:
            self.topups += 1
        self._last_sample_time = float(dt)

    def consume_last_sample_time(self) -> float:
        dt = self._last_sample_time
        self._last_sample_time = 0.0
        return dt

    def record_filter(self, dt: float, n_worlds: int) -> None:
        self.filter_time += float(dt)
        self.filter_events += 1
        self.worlds_survived += int(n_worlds)

    def record_selection(self, dt: float, sample_time: float = 0.0) -> None:
        self.selection_time += float(dt)
        self.selection_ex_sample += max(0.0, float(dt) - float(sample_time))

    def record_shot(self) -> None:
        self.shots += 1

    def record_game(self) -> None:
        self.games += 1

    def format_summary(self, label: str) -> str:
        games = max(1, self.games)
        shots = max(1, self.shots)
        avg_sample_per_game = self.sample_time / games
        avg_sample_per_shot = self.sample_time / shots
        avg_select_ex = self.selection_ex_sample / shots
        avg_filter = self.filter_time / max(1, self.filter_events)
        avg_worlds = self.worlds_survived / max(1, self.filter_events)
        avg_calls = self.sample_calls / games
        avg_topups = self.topups / games
        avg_layout = self.layout_time / games
        return (
            f"[SIM_PROFILE] {label}\n"
            f"  games={self.games} shots={self.shots}\n"
            f"  layout_time={avg_layout:.4f}s/game\n"
            f"  sample_time={self.sample_time:.4f}s "
            f"({avg_sample_per_game:.4f}s/game, {avg_sample_per_shot:.6f}s/shot)\n"
            f"  sample_calls={self.sample_calls} ({avg_calls:.2f}/game), "
            f"topups={self.topups} ({avg_topups:.2f}/game)\n"
            f"  filter_time={self.filter_time:.4f}s ({avg_filter:.6f}s/shot), "
            f"avg_worlds_after_filter={avg_worlds:.1f}\n"
            f"  selection_ex_sampling={self.selection_ex_sample:.4f}s "
            f"({avg_select_ex:.6f}s/shot)"
        )


class WorldCache:
    def __init__(self, ship_ids: Sequence[str], board_size: int) -> None:
        self.ship_ids = list(ship_ids)
        self.board_size = int(board_size)
        self.total_cells = self.board_size * self.board_size
        self.unions: List[int] = []
        self.ship_masks: List[Tuple[int, ...]] = []
        self.cell_hit_counts: List[int] = [0] * self.total_cells
        self._seen: set[int] = set()

    @property
    def size(self) -> int:
        return len(self.unions)

    def clear(self) -> None:
        self.unions = []
        self.ship_masks = []
        self.cell_hit_counts = [0] * self.total_cells
        self._seen = set()

    def _update_counts(self, union_mask: int, delta: int) -> None:
        m = union_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            if 0 <= idx < self.total_cells:
                self.cell_hit_counts[idx] += delta
            m ^= lsb

    def filter_in_place(
        self,
        miss_mask: int,
        assigned_hit_masks: Sequence[int],
        hit_mask: int,
        confirmed_sunk: set,
        profiler: Optional[SimProfiler] = None,
    ) -> int:
        if not self.unions:
            if profiler is not None:
                profiler.record_filter(0.0, 0)
            return 0
        start = time.perf_counter()
        filtered_union: List[int] = []
        filtered_masks: List[Tuple[int, ...]] = []
        removed = 0
        for union_mask, ship_masks_tuple in zip(self.unions, self.ship_masks):
            if union_mask & miss_mask:
                removed += 1
                self._update_counts(union_mask, -1)
                continue
            ok = True
            for i, req_mask in enumerate(assigned_hit_masks):
                if req_mask and (req_mask & ~ship_masks_tuple[i]) != 0:
                    ok = False
                    break
                if confirmed_sunk and (self.ship_ids[i] in confirmed_sunk):
                    if (ship_masks_tuple[i] & ~hit_mask) != 0:
                        ok = False
                        break
            if ok:
                filtered_union.append(union_mask)
                filtered_masks.append(ship_masks_tuple)
            else:
                removed += 1
                self._update_counts(union_mask, -1)
        self.unions = filtered_union
        self.ship_masks = filtered_masks
        if profiler is not None:
            profiler.record_filter(time.perf_counter() - start, len(self.unions))
        return removed

    def top_up(
        self,
        board: List[List[str]],
        placements: Dict[str, List[object]],
        ship_ids: Sequence[str],
        confirmed_sunk: set,
        assigned_hits: Dict[str, set],
        rng: random.Random,
        target_worlds: int,
        profiler: Optional[SimProfiler] = None,
    ) -> int:
        existing_union = self.unions if self.unions else None
        existing_masks = self.ship_masks if self.ship_masks else None
        sample_start = time.perf_counter()
        worlds_union, worlds_ship_masks, _, _, N = sample_worlds(
            board,
            placements,
            ship_ids,
            confirmed_sunk,
            assigned_hits,
            rng_seed=rng.randint(0, 2 ** 31 - 1),
            board_size=self.board_size,
            target_worlds=target_worlds,
            existing_union=existing_union,
            existing_ship_masks=existing_masks,
            return_ship_masks=True,
        )
        sample_time = time.perf_counter() - sample_start
        if profiler is not None:
            profiler.record_sample(sample_time, N, topup=bool(existing_union))

        added = 0
        for union_mask, ship_masks_tuple in zip(worlds_union, worlds_ship_masks):
            if union_mask in self._seen:
                continue
            self._seen.add(union_mask)
            self.unions.append(union_mask)
            self.ship_masks.append(ship_masks_tuple)
            self._update_counts(union_mask, +1)
            added += 1
        return added

    def posterior(self) -> Posterior:
        return Posterior(self.unions, self.cell_hit_counts, len(self.unions))


def _simulate_model_game(
    strategy: str,
    placements: Dict[str, List[object]],
    ship_ids: Sequence[str],
    board_size: int = BOARD_SIZE,
    rng: Optional[random.Random] = None,
    params: Optional[Dict[str, float]] = None,
    track_phases: bool = False,
    profiler: Optional[SimProfiler] = None,
) -> Tuple[int, Optional[Dict[str, int]]]:
    if rng is None:
        rng = random.Random()

    # 1. Generate a "True World" with full ship details
    # We need to know exactly where every ship is to simulate "Sunk" messages.
    true_layout: Dict[str, object] = {}
    used_mask = 0

    # Retry loop to ensure valid board (simple rejection sampling)
    layout_start = time.perf_counter()
    while len(true_layout) < len(ship_ids):
        true_layout = {}
        used_mask = 0
        valid_board = True

        # Randomize order to prevent bias
        shuffled_ships = list(ship_ids)
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
    if profiler is not None:
        profiler.record_layout_time(time.perf_counter() - layout_start)

    # 2. Setup Game State
    board = [[EMPTY for _ in range(board_size)] for _ in range(board_size)]
    confirmed_sunk = set()
    assigned_hits = {s: set() for s in ship_ids}
    assigned_hit_masks = [0 for _ in ship_ids]
    ship_index = {s: i for i, s in enumerate(ship_ids)}
    hit_mask = 0
    miss_mask = 0

    fast_mode = _env_flag("SIM_FAST_MODE", True)
    base_target = _env_int("SIM_TARGET_WORLDS", WORLD_SAMPLE_TARGET)
    target_worlds = _target_worlds_for_strategy(strategy, base_target)
    min_keep = _env_int("SIM_MIN_KEEP", max(1, target_worlds // 5))
    if min_keep > target_worlds:
        min_keep = target_worlds
    world_cache = WorldCache(ship_ids, board_size) if fast_mode else None
    sim_placements = placements
    frozen_ships: set[str] = set()

    # Track which cells belong to which ship for fast lookup
    cell_to_ship = {}
    for ship, p in true_layout.items():
        for r, c in p.cells:
            cell_to_ship[(r, c)] = ship

    shots = 0
    total_cells = board_size * board_size
    ships_remaining = len(ship_ids)

    # 3. Game Loop
    phase_counts: Optional[Dict[str, int]] = None
    if track_phases:
        phase_counts = {PHASE_HUNT: 0, PHASE_TARGET: 0, PHASE_ENDGAME: 0}

    while ships_remaining > 0 and shots < total_cells:
        if phase_counts is not None:
            phase = classify_phase(
                board,
                confirmed_sunk,
                assigned_hits,
                ship_ids,
                board_size=board_size,
            )
            phase_counts[phase] = phase_counts.get(phase, 0) + 1

        # Pass our "God Mode" knowledge into the bot
        if fast_mode and world_cache is not None:
            world_cache.filter_in_place(
                miss_mask,
                assigned_hit_masks,
                hit_mask,
                confirmed_sunk,
                profiler=profiler,
            )
            if world_cache.size == 0:
                world_cache.clear()
            if world_cache.size < min_keep:
                world_cache.top_up(
                    board,
                    sim_placements,
                    ship_ids,
                    confirmed_sunk,
                    assigned_hits,
                    rng,
                    target_worlds=target_worlds,
                    profiler=profiler,
                )
            posterior = world_cache.posterior()
            selection_start = time.perf_counter()
            r, c = _choose_next_shot_for_strategy(
                strategy,
                board,
                sim_placements,
                rng,
                ship_ids,
                board_size,
                known_sunk=confirmed_sunk,
                known_assigned=assigned_hits,
                params=params,
                posterior=posterior,
            )
            if profiler is not None:
                profiler.record_selection(time.perf_counter() - selection_start)
        else:
            selection_start = time.perf_counter()
            r, c = _choose_next_shot_for_strategy(
                strategy,
                board,
                sim_placements,
                rng,
                ship_ids,
                board_size,
                known_sunk=confirmed_sunk,
                known_assigned=assigned_hits,
                params=params,
                profiler=profiler,
            )
            if profiler is not None:
                sample_time = profiler.consume_last_sample_time()
                profiler.record_selection(time.perf_counter() - selection_start, sample_time=sample_time)

        idx = cell_index(r, c, board_size)
        is_hit = (used_mask >> idx) & 1

        board[r][c] = HIT if is_hit else MISS
        shots += 1
        if profiler is not None:
            profiler.record_shot()
        if is_hit:
            hit_mask |= 1 << idx
        else:
            miss_mask |= 1 << idx

        if is_hit:
            # Identify which ship was hit
            hit_ship = cell_to_ship[(r, c)]

            # FEATURE 1: Auto-Assign Hits (Perfect Play)
            # The bot "knows" which ship it hit.
            assigned_hits[hit_ship].add((r, c))
            ship_idx = ship_index.get(hit_ship)
            if ship_idx is not None:
                assigned_hit_masks[ship_idx] |= 1 << idx

            # FEATURE 2: Auto-Mark Sunk (Standard Rules)
            # Check if this ship is now fully sunk
            ship_cells = true_layout[hit_ship].cells
            is_sunk = all(board[sr][sc] == HIT for sr, sc in ship_cells)

            if is_sunk and hit_ship not in confirmed_sunk:
                confirmed_sunk.add(hit_ship)
                ships_remaining -= 1
                ship_idx = ship_index.get(hit_ship)
                if ship_idx is not None:
                    assigned_hits[hit_ship] = set(true_layout[hit_ship].cells)
                    assigned_hit_masks[ship_idx] = true_layout[hit_ship].mask
                if hit_ship not in frozen_ships:
                    if sim_placements is placements:
                        sim_placements = dict(placements)
                    sim_placements[hit_ship] = [true_layout[hit_ship]]
                    frozen_ships.add(hit_ship)

    if profiler is not None:
        profiler.record_game()
    return shots, phase_counts


def simulate_model_game(
    strategy: str,
    placements: Dict[str, List[object]],
    ship_ids: Sequence[str],
    board_size: int = BOARD_SIZE,
    rng: Optional[random.Random] = None,
    params: Optional[Dict[str, float]] = None,
    profiler: Optional[SimProfiler] = None,
) -> int:
    shots, _ = _simulate_model_game(
        strategy,
        placements,
        ship_ids,
        board_size=board_size,
        rng=rng,
        params=params,
        track_phases=False,
        profiler=profiler,
    )
    return shots


def simulate_model_game_with_phases(
    strategy: str,
    placements: Dict[str, List[object]],
    ship_ids: Sequence[str],
    board_size: int = BOARD_SIZE,
    rng: Optional[random.Random] = None,
    params: Optional[Dict[str, float]] = None,
    profiler: Optional[SimProfiler] = None,
) -> Tuple[int, Dict[str, int]]:
    shots, phases = _simulate_model_game(
        strategy,
        placements,
        ship_ids,
        board_size=board_size,
        rng=rng,
        params=params,
        track_phases=True,
        profiler=profiler,
    )
    return shots, phases or {}
