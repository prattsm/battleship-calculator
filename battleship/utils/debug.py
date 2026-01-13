from datetime import datetime
from PyQt5 import QtWidgets

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


def debug_event(
    parent,
    title: str,
    message: str,
    details: str = "",
    *,
    force_popup: bool = False,
    level: str = "info",
) -> None:
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
