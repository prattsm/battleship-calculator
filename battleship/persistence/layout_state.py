from typing import Any, Dict, List, Optional, Tuple

from battleship.layouts.builtins import legacy_layout
from battleship.layouts.definition import LayoutDefinition
from battleship.persistence.io import atomic_write_json, load_json_file


def layout_key(layout: LayoutDefinition) -> str:
    return f"{layout.layout_id}:{layout.layout_version}:{layout.layout_hash}"


def _load_raw(path: str) -> Dict[str, Any]:
    data = load_json_file(path)
    if data is None:
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
    atomic_write_json(path, data)


def delete_layout_state(path: str, layout: LayoutDefinition) -> None:
    data = _load_raw(path)
    layouts = data.get("layouts", {})
    key = layout_key(layout)
    if key in layouts:
        del layouts[key]
    atomic_write_json(path, data)


def find_layout_versions(path: str, layout_id: str) -> List[Tuple[int, str]]:
    data = _load_raw(path)
    layouts = data.get("layouts", {})
    results: List[Tuple[int, str]] = []
    if not isinstance(layouts, dict):
        return results
    for key in layouts.keys():
        if not isinstance(key, str):
            continue
        parts = key.split(":", 2)
        if len(parts) != 3:
            continue
        lid, version_str, layout_hash = parts
        if lid != layout_id:
            continue
        try:
            version = int(version_str)
        except ValueError:
            continue
        results.append((version, layout_hash))
    return results
