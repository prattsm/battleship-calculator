import json
import os
from typing import Any, Dict, Optional, Tuple

from battleship.layouts.builtins import legacy_layout
from battleship.layouts.definition import LayoutDefinition


def layout_key(layout: LayoutDefinition) -> str:
    return f"{layout.layout_id}:{layout.layout_version}:{layout.layout_hash}"


def _load_raw(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"schema": 1, "layouts": {}}
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"schema": 1, "layouts": {}}

    if isinstance(data, dict) and "layouts" in data and isinstance(data.get("layouts"), dict):
        return data

    # Legacy format: treat entire payload as legacy layout bucket.
    legacy = legacy_layout()
    return {
        "schema": 1,
        "layouts": {
            layout_key(legacy): data,
        },
    }


def load_layout_state(path: str, layout: LayoutDefinition) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    data = _load_raw(path)
    bucket = data.get("layouts", {}).get(layout_key(layout))
    return bucket if isinstance(bucket, dict) else None, data


def save_layout_state(path: str, layout: LayoutDefinition, state: Dict[str, Any]) -> None:
    data = _load_raw(path)
    layouts = data.setdefault("layouts", {})
    layouts[layout_key(layout)] = state
    try:
        with open(path, "w") as f:
            json.dump(data, f)
    except OSError:
        pass


def delete_layout_state(path: str, layout: LayoutDefinition) -> None:
    data = _load_raw(path)
    layouts = data.get("layouts", {})
    key = layout_key(layout)
    if key in layouts:
        del layouts[key]
    try:
        with open(path, "w") as f:
            json.dump(data, f)
    except OSError:
        pass
