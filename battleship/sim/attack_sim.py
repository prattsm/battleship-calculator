import os
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

from battleship.domain.board import cell_index
from battleship.domain.config import BOARD_SIZE, EMPTY, HIT, MISS, WORLD_SAMPLE_TARGET
from battleship.domain.phase import PHASE_ENDGAME, PHASE_HUNT, PHASE_TARGET, classify_phase
from battleship.domain.worlds import sample_worlds as _sample_worlds
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


_SAMPLE_TRACE_LIMIT = 50
_sample_trace_count = 0
_sample_trace_enabled = False


def _sample_worlds_with_trace(*args, **kwargs):
    global _sample_trace_count
    result = _sample_worlds(*args, **kwargs)
    if _sample_trace_enabled and _sample_trace_count < _SAMPLE_TRACE_LIMIT:
        try:
            import inspect

            frame = inspect.currentframe()
            caller = frame.f_back if frame else None
            if caller is not None:
                info = inspect.getframeinfo(caller)
                print(
                    "[SIM_PROFILE_CALLSITE]",
                    f"{os.path.basename(info.filename)}:{info.function}",
                )
        except Exception:
            pass
        _sample_trace_count += 1
    return result


def _enable_sample_trace(enable: bool) -> None:
    global _sample_trace_enabled, _sample_trace_count
    _sample_trace_enabled = bool(enable)
    if _sample_trace_enabled:
        _sample_trace_count = 0


def _is_heavy_strategy(strategy: str) -> bool:
    return strategy in {"entropy1", "two_ply", "rollout_mcts"}


def _default_topup_targets(
    strategy: str,
    phase: Optional[str],
    base_target: int,
) -> Tuple[int, int]:
    if strategy == "hybrid_phase" and phase:
        if phase == PHASE_HUNT:
            min_keep = 800
            topup_to = 6000
        elif phase == PHASE_TARGET:
            min_keep = 300
            topup_to = 2500
        else:
            min_keep = 500
            topup_to = 4000
        topup_to = min(topup_to, base_target)
        min_keep = min(min_keep, topup_to)
        return min_keep, topup_to

    min_keep = 700
    if _is_heavy_strategy(strategy):
        topup_to = min(base_target, 10000)
    else:
        topup_to = min(base_target, 5000)
    min_keep = min(min_keep, topup_to)
    return min_keep, topup_to


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
        self.topup_reason_lowN = 0
        self.topup_reason_empty = 0
        self.topup_reason_strategy_full = 0
        self.topup_n_sum = 0
        self.topup_events = 0
        self.topup_gap_sum = 0
        self.topup_gap_events = 0
        self._last_topup_shot: Optional[int] = None
        self.filter_time = 0.0
        self.filter_events = 0
        self.worlds_before_filter = 0
        self.worlds_after_filter = 0
        self.selection_time = 0.0
        self.selection_ex_sample = 0.0
        self._last_sample_time = 0.0
        self.posterior_calls = 0
        self.phase_shots = 0
        self.posterior_used_by_phase: Dict[str, int] = {}
        self.posterior_skipped_by_phase: Dict[str, int] = {}
        self.posterior_degraded = 0

    def record_layout_time(self, dt: float) -> None:
        self.layout_time += float(dt)

    def record_sample(self, dt: float, n_worlds: int, topup: bool = False) -> None:
        self.sample_time += float(dt)
        self.sample_calls += 1
        self._last_sample_time = float(dt)

    def consume_last_sample_time(self) -> float:
        dt = self._last_sample_time
        self._last_sample_time = 0.0
        return dt

    def record_filter(self, dt: float, before_count: int, after_count: int) -> None:
        self.filter_time += float(dt)
        self.filter_events += 1
        self.worlds_before_filter += int(before_count)
        self.worlds_after_filter += int(after_count)

    def record_selection(self, dt: float, sample_time: float = 0.0) -> None:
        self.selection_time += float(dt)
        self.selection_ex_sample += max(0.0, float(dt) - float(sample_time))

    def record_posterior_call(self) -> None:
        self.posterior_calls += 1

    def record_posterior_phase(self, phase: Optional[str], used: bool) -> None:
        key = phase or "unknown"
        target = self.posterior_used_by_phase if used else self.posterior_skipped_by_phase
        target[key] = int(target.get(key, 0)) + 1

    def record_posterior_degraded(self) -> None:
        self.posterior_degraded += 1

    def record_phase_shots(self, count: int) -> None:
        self.phase_shots += int(count)

    def record_topup(
        self,
        n_before: int,
        current_shot: int,
        reason_lowN: bool,
        reason_empty: bool,
        reason_strategy_full: bool,
    ) -> None:
        self.topups += 1
        self.topup_events += 1
        self.topup_n_sum += int(n_before)
        if self._last_topup_shot is not None:
            gap = int(current_shot) - int(self._last_topup_shot)
            if gap >= 0:
                self.topup_gap_sum += gap
                self.topup_gap_events += 1
        self._last_topup_shot = int(current_shot)
        if reason_lowN:
            self.topup_reason_lowN += 1
        if reason_empty:
            self.topup_reason_empty += 1
        if reason_strategy_full:
            self.topup_reason_strategy_full += 1

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
        avg_worlds_after = self.worlds_after_filter / max(1, self.filter_events)
        avg_worlds_before = self.worlds_before_filter / max(1, self.filter_events)
        survival_rate = (
            self.worlds_after_filter / self.worlds_before_filter
            if self.worlds_before_filter > 0
            else 0.0
        )
        avg_calls = self.sample_calls / games
        avg_topups = self.topups / games
        avg_topup_n = self.topup_n_sum / max(1, self.topup_events)
        avg_topup_gap = self.topup_gap_sum / max(1, self.topup_gap_events)
        avg_layout = self.layout_time / games
        posterior_per_shot = self.posterior_calls / shots
        return (
            f"[SIM_PROFILE] {label}\n"
            f"  games={self.games} shots={self.shots}\n"
            f"  layout_time={avg_layout:.4f}s/game\n"
            f"  sample_time={self.sample_time:.4f}s "
            f"({avg_sample_per_game:.4f}s/game, {avg_sample_per_shot:.6f}s/shot)\n"
            f"  sample_calls={self.sample_calls} ({avg_calls:.2f}/game), "
            f"topups={self.topups} ({avg_topups:.2f}/game), "
            f"avg_N_at_topup={avg_topup_n:.1f}, "
            f"shots_between_topups={avg_topup_gap:.1f}\n"
            f"  topup_reason_lowN={self.topup_reason_lowN}, "
            f"topup_reason_empty={self.topup_reason_empty}, "
            f"topup_reason_strategy_requires_full={self.topup_reason_strategy_full}\n"
            f"  posterior_calls={self.posterior_calls} ({posterior_per_shot:.2f}/shot)\n"
            f"  posterior_used_by_phase={self.posterior_used_by_phase}\n"
            f"  posterior_skipped_by_phase={self.posterior_skipped_by_phase}\n"
            f"  posterior_degraded={self.posterior_degraded}\n"
            f"  filter_time={self.filter_time:.4f}s ({avg_filter:.6f}s/shot), "
            f"avg_worlds_before_filter={avg_worlds_before:.1f}, "
            f"avg_worlds_after_filter={avg_worlds_after:.1f}, "
            f"survival_rate={survival_rate:.3f}\n"
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
        self.saturated = False
        self.saturated_N = 0
        self.no_progress_topups = 0

    @property
    def size(self) -> int:
        return len(self.unions)

    def clear(self) -> None:
        self.unions = []
        self.ship_masks = []
        self.cell_hit_counts = [0] * self.total_cells
        self._seen = set()
        self.saturated = False
        self.saturated_N = 0
        self.no_progress_topups = 0

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
        if len(assigned_hit_masks) != len(self.ship_ids):
            raise AssertionError("assigned_hit_masks length mismatch for WorldCache")
        if not self.unions:
            if profiler is not None:
                profiler.record_filter(0.0, 0, 0)
            return 0
        sunk_flags = [ship_id in confirmed_sunk for ship_id in self.ship_ids] if confirmed_sunk else None
        start = time.perf_counter()
        before_count = len(self.unions)
        filtered_union: List[int] = []
        filtered_masks: List[Tuple[int, ...]] = []
        removed = 0
        for union_mask, ship_masks_tuple in zip(self.unions, self.ship_masks):
            if union_mask & miss_mask:
                removed += 1
                self._update_counts(union_mask, -1)
                continue
            if (union_mask & hit_mask) != hit_mask:
                removed += 1
                self._update_counts(union_mask, -1)
                continue
            ok = True
            for i, req_mask in enumerate(assigned_hit_masks):
                if req_mask and (req_mask & ~ship_masks_tuple[i]) != 0:
                    ok = False
                    break
                if sunk_flags and sunk_flags[i]:
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
            profiler.record_filter(time.perf_counter() - start, before_count, len(self.unions))
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
        max_attempts_factor: Optional[int] = None,
        profiler: Optional[SimProfiler] = None,
    ) -> Tuple[int, int, int]:
        if tuple(ship_ids) != tuple(self.ship_ids):
            raise AssertionError("ship_ids mismatch for WorldCache")
        def run_sample(attempts_factor: Optional[int]) -> Tuple[int, int, int]:
            existing_union = self.unions if self.unions else None
            existing_masks = self.ship_masks if self.ship_masks else None
            seed_count = len(existing_union) if existing_union is not None else 0
            sample_start = time.perf_counter()
            worlds_union, worlds_ship_masks, _, _, N = _sample_worlds_with_trace(
                board,
                placements,
                ship_ids,
                confirmed_sunk,
                assigned_hits,
                rng_seed=rng.randint(0, 2 ** 31 - 1),
                board_size=self.board_size,
                target_worlds=target_worlds,
                max_attempts_factor=attempts_factor,
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
            return added, N, seed_count

        added, N, seed_count = run_sample(max_attempts_factor)
        if added <= 0 and max_attempts_factor is None:
            retry_factor = _env_int("SIM_TOPUP_RETRY_FACTOR", 25)
            retry_cap_raw = os.getenv("SIM_TOPUP_RETRY_CAP")
            if retry_cap_raw is not None:
                retry_cap = _env_int("SIM_TOPUP_RETRY_CAP", retry_factor)
                if retry_cap > 0:
                    retry_factor = min(retry_factor, retry_cap)
            if retry_factor > 0:
                added_retry, N_retry, seed_count = run_sample(retry_factor)
                added = added_retry
                N = N_retry

        no_progress = (added <= 0)
        if no_progress:
            self.no_progress_topups += 1
        else:
            self.no_progress_topups = 0
            self.saturated = False
        if self.no_progress_topups >= 3:
            self.saturated = True
            self.saturated_N = len(self.unions)
        return added, N, seed_count

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
    log_rng = random.Random()
    trace_enabled = _env_flag("SIM_TRACE_SAMPLE_CALLS", False)
    selection_module = None
    original_selection_sample = None
    if trace_enabled:
        _enable_sample_trace(True)
        try:
            from battleship.strategies import selection as _selection

            selection_module = _selection
            original_selection_sample = getattr(_selection, "sample_worlds", None)
            _selection.sample_worlds = _sample_worlds_with_trace
        except Exception:
            selection_module = None
            original_selection_sample = None

    fast_mode = _env_flag("SIM_FAST_MODE", True)
    base_target = _env_int("SIM_TARGET_WORLDS", WORLD_SAMPLE_TARGET)
    env_min_keep = os.getenv("SIM_MIN_KEEP")
    env_topup_to = os.getenv("SIM_TOPUP_TO") or os.getenv("SIM_REFILL_TARGET")
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
    unknown_cells = [(r, c) for r in range(board_size) for c in range(board_size)]
    unknown_index = {cell: i for i, cell in enumerate(unknown_cells)}

    def _remove_unknown(cell: Tuple[int, int]) -> None:
        idx = unknown_index.pop(cell, None)
        if idx is None:
            return
        last = unknown_cells.pop()
        if idx < len(unknown_cells):
            unknown_cells[idx] = last
            unknown_index[last] = idx
    ships_remaining = len(ship_ids)

    # 3. Game Loop
    phase_counts: Optional[Dict[str, int]] = None
    if track_phases:
        phase_counts = {PHASE_HUNT: 0, PHASE_TARGET: 0, PHASE_ENDGAME: 0}

    try:
        while ships_remaining > 0 and shots < total_cells:
            current_phase: Optional[str] = None
            if phase_counts is not None:
                phase = classify_phase(
                    board,
                    confirmed_sunk,
                    assigned_hits,
                    ship_ids,
                    board_size=board_size,
                )
                phase_counts[phase] = phase_counts.get(phase, 0) + 1
                current_phase = phase
            elif strategy == "hybrid_phase":
                current_phase = classify_phase(
                    board,
                    confirmed_sunk,
                    assigned_hits,
                    ship_ids,
                    board_size=board_size,
                )

            # Pass our "God Mode" knowledge into the bot
            skip_posterior = False
            if strategy == "random":
                skip_posterior = True
            elif strategy in {"random_checkerboard", "systematic_checkerboard", "diagonal_stripe"}:
                if hit_mask == 0:
                    skip_posterior = True

            posterior_used = False
            if skip_posterior:
                if not unknown_cells:
                    r, c = 0, 0
                elif strategy == "random":
                    r, c = rng.choice(unknown_cells)
                elif strategy == "systematic_checkerboard":
                    white_min = None
                    any_min = None
                    for cell in unknown_cells:
                        if any_min is None or cell < any_min:
                            any_min = cell
                        if (cell[0] + cell[1]) % 2 == 0:
                            if white_min is None or cell < white_min:
                                white_min = cell
                    pick = white_min if white_min is not None else any_min
                    r, c = pick if pick is not None else (0, 0)
                elif strategy == "diagonal_stripe":
                    diagonals = [p for p in unknown_cells if (p[0] - p[1]) % 4 == 0]
                    if diagonals:
                        r, c = rng.choice(diagonals)
                    else:
                        secondary = [p for p in unknown_cells if (p[0] - p[1]) % 2 == 0]
                        r, c = rng.choice(secondary) if secondary else rng.choice(unknown_cells)
                else:
                    whites = [p for p in unknown_cells if (p[0] + p[1]) % 2 == 0]
                    r, c = rng.choice(whites) if whites else rng.choice(unknown_cells)
            elif fast_mode and world_cache is not None:
                if strategy in {"placement_factorized", "assigned_target_marginal"}:
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
                        unknown_cells=unknown_cells,
                        has_any_hit=bool(hit_mask),
                        hit_mask=hit_mask,
                        miss_mask=miss_mask,
                    )
                    if profiler is not None:
                        profiler.record_selection(time.perf_counter() - selection_start)
                    posterior_used = False
                else:
                    min_keep, topup_to = _default_topup_targets(strategy, current_phase, base_target)
                    if env_min_keep is not None:
                        min_keep = _env_int("SIM_MIN_KEEP", min_keep)
                    if env_topup_to is not None:
                        if os.getenv("SIM_TOPUP_TO") is not None:
                            topup_to = _env_int("SIM_TOPUP_TO", topup_to)
                        else:
                            topup_to = _env_int("SIM_REFILL_TARGET", topup_to)
                    if topup_to > base_target:
                        topup_to = base_target
                    if min_keep > topup_to:
                        min_keep = topup_to
                    n_before_filter = world_cache.size
                    world_cache.filter_in_place(
                        miss_mask,
                        assigned_hit_masks,
                        hit_mask,
                        confirmed_sunk,
                        profiler=profiler,
                    )
                    n_after_filter = world_cache.size
                    if world_cache.size == 0:
                        world_cache.clear()
                    if not world_cache.saturated and world_cache.size < min_keep:
                        if profiler is not None:
                            profiler.record_topup(
                                n_before=world_cache.size,
                                current_shot=shots,
                                reason_lowN=True,
                                reason_empty=(world_cache.size == 0),
                                reason_strategy_full=_is_heavy_strategy(strategy),
                            )
                        start_size = world_cache.size
                        desired = topup_to
                        added, n_returned, seed_count = world_cache.top_up(
                            board,
                            sim_placements,
                            ship_ids,
                            confirmed_sunk,
                            assigned_hits,
                            rng,
                            target_worlds=desired,
                            max_attempts_factor=None,
                            profiler=profiler,
                        )
                        if profiler is not None and log_rng.random() < 0.01:
                            print(
                                "[SIM_PROFILE_DETAIL] topup",
                                f"N_before_filter={n_before_filter}",
                                f"N_after_filter={n_after_filter}",
                                f"N_before_topup={start_size}",
                                f"MIN_KEEP={min_keep}",
                                f"TOPUP_TO={topup_to}",
                                f"target_worlds={desired}",
                                "max_attempts_factor=default",
                                f"existing_empty={seed_count == 0}",
                                f"N_returned={n_returned}",
                                f"num_added={added}",
                                f"seed_count={seed_count}",
                                f"seen_union_size={len(world_cache._seen)}",
                            )
                    posterior = world_cache.posterior()
                    if profiler is not None and world_cache.saturated:
                        profiler.record_posterior_degraded()
                    posterior_used = True
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
                        unknown_cells=unknown_cells,
                        has_any_hit=bool(hit_mask),
                        hit_mask=hit_mask,
                        miss_mask=miss_mask,
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
                    unknown_cells=unknown_cells,
                    has_any_hit=bool(hit_mask),
                    hit_mask=hit_mask,
                    miss_mask=miss_mask,
                )
                if profiler is not None:
                    sample_time = profiler.consume_last_sample_time()
                    profiler.record_selection(time.perf_counter() - selection_start, sample_time=sample_time)
                posterior_used = True

            if profiler is not None:
                profiler.record_posterior_call()
                profiler.record_posterior_phase(current_phase, used=posterior_used)

            if board[r][c] != EMPTY:
                if _env_flag("SIM_DEBUG_STATS", False):
                    assert board[r][c] == EMPTY, "Selected a non-empty cell"
                if unknown_cells:
                    r, c = rng.choice(unknown_cells)

            idx = cell_index(r, c, board_size)
            is_hit = (used_mask >> idx) & 1

            board[r][c] = HIT if is_hit else MISS
            _remove_unknown((r, c))
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
                        if _env_flag("SIM_DEBUG_ASSERTS", False):
                            assert assigned_hit_masks[ship_idx] == true_layout[hit_ship].mask
                            assert (assigned_hit_masks[ship_idx] & ~hit_mask) == 0
                    if hit_ship not in frozen_ships:
                        if sim_placements is placements:
                            sim_placements = dict(placements)
                        sim_placements[hit_ship] = [true_layout[hit_ship]]
                        frozen_ships.add(hit_ship)
                        if world_cache is not None:
                            world_cache.clear()

    finally:
        if trace_enabled:
            _enable_sample_trace(False)
            if selection_module is not None and original_selection_sample is not None:
                try:
                    selection_module.sample_worlds = original_selection_sample
                except Exception:
                    pass
    if profiler is not None:
        profiler.record_game()
        if _env_flag("SIM_DEBUG_STATS", False):
            total_phase = sum(phase_counts.values()) if phase_counts else 0
            if track_phases:
                assert total_phase == shots, "Phase counts must sum to shots in sim"
            used = sum(profiler.posterior_used_by_phase.values())
            skipped = sum(profiler.posterior_skipped_by_phase.values())
            assert used + skipped == profiler.posterior_calls, "Posterior phase counts must match calls"
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
