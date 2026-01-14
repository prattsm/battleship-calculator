import json
import os
from typing import Any, Dict, Optional

from battleship.layouts.definition import LayoutDefinition


APP_STATE_SCHEMA = 1


def _default_state() -> Dict[str, Any]:
    return {
        "schema": APP_STATE_SCHEMA,
        "selected_layout": None,
        "match": {"active_opponent": None, "records": {}},
    }


def _load_raw(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return _default_state()
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return _default_state()
    if not isinstance(data, dict):
        return _default_state()
    if "selected_layout" not in data:
        data["selected_layout"] = None
    if "match" not in data or not isinstance(data.get("match"), dict):
        data["match"] = {"active_opponent": None, "records": {}}
    return data


def _write_atomic(path: str, data: Dict[str, Any]) -> None:
    tmp_path = f"{path}.tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except OSError:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def load_selected_layout(path: str) -> Optional[Dict[str, Any]]:
    data = _load_raw(path)
    selected = data.get("selected_layout")
    return selected if isinstance(selected, dict) else None


def save_selected_layout(path: str, layout: LayoutDefinition) -> None:
    data = _load_raw(path)
    data["selected_layout"] = {
        "layout_id": layout.layout_id,
        "layout_version": layout.layout_version,
        "layout_hash": layout.layout_hash,
    }
    _write_atomic(path, data)


def load_match_state(path: str) -> Dict[str, Any]:
    data = _load_raw(path)
    match = data.get("match")
    if not isinstance(match, dict):
        return {"active_opponent": None, "records": {}}

    active = match.get("active_opponent")
    active_opponent = active if isinstance(active, str) and active.strip() else None

    records_raw = match.get("records")
    records: Dict[str, Dict[str, int]] = {}
    if isinstance(records_raw, dict):
        for name, record in records_raw.items():
            if not isinstance(name, str) or not name.strip():
                continue
            wins = 0
            losses = 0
            if isinstance(record, dict):
                try:
                    wins = int(record.get("wins", 0))
                except (TypeError, ValueError):
                    wins = 0
                try:
                    losses = int(record.get("losses", 0))
                except (TypeError, ValueError):
                    losses = 0
            records[name.strip()] = {
                "wins": max(0, wins),
                "losses": max(0, losses),
            }

    return {"active_opponent": active_opponent, "records": records}


def save_match_state(path: str, match_state: Dict[str, Any]) -> None:
    data = _load_raw(path)
    if not isinstance(match_state, dict):
        match_state = {"active_opponent": None, "records": {}}
    data["match"] = match_state
    _write_atomic(path, data)
