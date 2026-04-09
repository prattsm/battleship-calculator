import json
import os
from typing import Any, Optional


def load_json_file(path: str) -> Optional[Any]:
    """Load JSON from the primary file or an interrupted-write temp file."""
    candidates = []
    for candidate in (path, f"{path}.tmp"):
        if not os.path.exists(candidate):
            continue
        try:
            mtime = os.path.getmtime(candidate)
        except OSError:
            mtime = 0.0
        candidates.append((candidate, mtime, candidate.endswith(".tmp")))

    for candidate, _mtime, _is_tmp in sorted(candidates, key=lambda item: (item[1], item[2]), reverse=True):
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
    return None


def _fsync_directory(path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    try:
        dir_fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


def atomic_write_json(path: str, data: Any) -> None:
    """Write JSON atomically and fsync the parent directory for crash safety."""
    tmp_path = f"{path}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        _fsync_directory(path)
    except OSError:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
