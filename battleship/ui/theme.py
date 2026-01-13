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
