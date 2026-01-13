from typing import Dict, Optional, Tuple

from battleship.layouts.definition import LayoutDefinition
from battleship.persistence.layout_state import load_layout_state


def _compute_best_overall(model_stats: Dict[str, object]) -> Optional[str]:
    best_key = None
    best_avg = None
    for key, stats in model_stats.items():
        if not isinstance(stats, dict):
            continue
        total_games = int(stats.get("total_games", 0) or 0)
        total_shots = float(stats.get("total_shots", 0.0) or 0.0)
        if total_games <= 0:
            continue
        avg = total_shots / total_games
        if best_avg is None or avg < best_avg:
            best_avg = avg
            best_key = key
    return best_key


def _compute_best_by_phase(model_stats: Dict[str, object]) -> Dict[str, str]:
    best_by_phase: Dict[str, str] = {}
    if not isinstance(model_stats, dict):
        return best_by_phase
    for key, stats in model_stats.items():
        if not isinstance(stats, dict):
            continue
        phase_block = stats.get("phase")
        if not isinstance(phase_block, dict):
            continue
        for phase, entry in phase_block.items():
            if not isinstance(entry, dict):
                continue
            games = int(entry.get("total_games", stats.get("total_games", 0)))
            if games <= 0:
                continue
            total = float(entry.get("total_shots", 0.0))
            avg = total / games if games > 0 else None
            if avg is None:
                continue
            prev_key = best_by_phase.get(phase)
            if prev_key is None:
                best_by_phase[phase] = key
            else:
                prev_stats = model_stats.get(prev_key, {})
                prev_phase = prev_stats.get("phase", {}).get(phase, {}) if isinstance(prev_stats, dict) else {}
                prev_total = float(prev_phase.get("total_shots", 0.0))
                prev_games = int(prev_phase.get("total_games", prev_stats.get("total_games", 0))) if isinstance(prev_stats, dict) else 0
                prev_avg = prev_total / prev_games if prev_games > 0 else None
                if prev_avg is None or avg < prev_avg:
                    best_by_phase[phase] = key
    return best_by_phase


def load_best_models(
    path: str,
    layout: LayoutDefinition,
) -> Tuple[Optional[str], Dict[str, str]]:
    """
    Return (best_overall, best_by_phase) from persisted model stats for the layout.
    If explicit best fields are missing, compute overall best from average shots.
    """
    data, _raw = load_layout_state(path, layout)
    if not data:
        return None, {}

    best_overall = data.get("best_model_overall")
    if not isinstance(best_overall, str) or not best_overall:
        best_overall = None

    best_by_phase: Dict[str, str] = {}
    raw_phase = data.get("best_model_by_phase")
    if isinstance(raw_phase, dict):
        for phase, key in raw_phase.items():
            if isinstance(phase, str) and isinstance(key, str) and key:
                best_by_phase[phase] = key

    if best_overall is None:
        stats = data.get("model_stats")
        if isinstance(stats, dict):
            best_overall = _compute_best_overall(stats)

    if not best_by_phase:
        stats = data.get("model_stats")
        if isinstance(stats, dict):
            best_by_phase = _compute_best_by_phase(stats)

    return best_overall, best_by_phase
