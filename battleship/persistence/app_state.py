import json
import os
from typing import Any, Dict, Optional

from battleship.layouts.definition import LayoutDefinition


APP_STATE_SCHEMA = 1


def _default_state() -> Dict[str, Any]:
    return {"schema": APP_STATE_SCHEMA, "selected_layout": None}


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
