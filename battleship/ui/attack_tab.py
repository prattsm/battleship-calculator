import math
import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

from battleship.domain.board import board_masks, cell_index, create_board
from battleship.domain.config import (
    EMPTY,
    ENUMERATION_PRODUCT_LIMIT,
    HAS_SHIP,
    HIT,
    MISS,
    PARAM_SPECS,
)
from battleship.domain.phase import (
    PHASE_ENDGAME,
    PHASE_HUNT,
    PHASE_TARGET,
    classify_phase,
)
from battleship.domain.worlds import (
    compute_min_expected_worlds_after_one_shot,
    filter_allowed_placements,
    sample_worlds,
)
from battleship.layouts.cache import LayoutRuntime
from battleship.persistence.layout_state import delete_layout_state, load_layout_state, save_layout_state
from battleship.persistence.model_selection import load_best_models
from battleship.persistence.stats import StatsTracker
from battleship.sim.defense_sim import build_base_heat, simulate_enemy_game_phase
from battleship.strategies.selection import two_ply_selection
from battleship.strategies.registry import model_defs
from battleship.ui.theme import Theme


class AttackTab(QtWidgets.QWidget):
    STATE_PATH = "battleship_attack_state.json"
    MODEL_STATS_PATH = "battleship_model_stats.json"
    AUTO_MODE_PHASE = "auto_phase"
    AUTO_MODE_OVERALL = "auto_overall"
    AUTO_MODE_MANUAL = "manual"

    def __init__(self, stats: StatsTracker, layout_runtime: LayoutRuntime, parent=None):
        super().__init__(parent)
        self.stats = stats
        self.layout_runtime = layout_runtime
        self.layout = layout_runtime.definition
        self.board_size = self.layout.board_size
        self.ship_ids = list(self.layout.ship_ids())
        self.placements = layout_runtime.placements
        self.board = create_board(self.board_size)

        self.world_masks: List[int] = []
        self.cell_hit_counts: List[int] = [0] * (self.board_size * self.board_size)
        self.cell_probs: List[float] = [0.0] * (self.board_size * self.board_size)
        self.info_gain_values: List[float] = [0.0] * (self.board_size * self.board_size)
        self.num_world_samples: int = 0
        self.ship_sunk_probs: Dict[str, float] = {s: 0.0 for s in self.ship_ids}
        self.confirmed_sunk: Set[str] = set()
        self.assigned_hits: Dict[str, Set[Tuple[int, int]]] = {s: set() for s in self.ship_ids}
        self.assign_mode_ship: Optional[str] = None
        self.best_cells: List[Tuple[int, int]] = []
        self.best_prob: float = 0.0
        self.two_ply_best_cells: List[Tuple[int, int]] = []
        self.two_ply_best_prob: float = 0.0
        self.game_over: bool = False
        self.active_cell: Optional[Tuple[int, int]] = None

        # Undo/redo history
        self.undo_stack: List[Dict[str, object]] = []
        self.redo_stack: List[Dict[str, object]] = []
        self._suspend_history: bool = False

        # Model selection state
        self.model_defs = model_defs()
        self.attack_model_defs = list(self.model_defs)
        if not any(md.get("key") == "two_ply" for md in self.attack_model_defs):
            self.attack_model_defs.append(
                {
                    "key": "two_ply",
                    "name": "Two-ply (Legacy)",
                    "description": "Two-ply information heuristic (legacy Attack tab behavior).",
                    "notes": "Uses a 2-step information criterion; keeps the prior UI feel when no stats are available.",
                }
            )
        self.auto_mode = self.AUTO_MODE_PHASE
        self.manual_model_key = "two_ply"
        self.lock_model_for_game = False
        self.locked_model_key: Optional[str] = None
        self.active_model_key: Optional[str] = None
        self.active_model_reason: str = ""
        self.active_phase: str = PHASE_HUNT

        self.linked_defense_tab = None

        # World-model diagnostics
        self.enumeration_mode: bool = False  # True = exact enumeration, False = Monte Carlo
        self.remaining_ship_count: int = len(self.ship_ids)

        self.ship_friendly_names = {
            spec.instance_id: (spec.name or spec.instance_id)
            for spec in self.layout.ships
        }

        self._build_ui()
        self._update_model_controls()
        self._update_history_buttons()
        self.load_state()
        self.recompute()

    def _board_cell_size(self) -> int:
        base = 520
        size = int(base / max(1, self.board_size))
        return max(24, min(48, size))

    def _build_ui(self):
        def wrap_scroll(widget: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
            scroll.setWidget(widget)
            return scroll

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        main_layout.addWidget(splitter, stretch=1)

        # --- Left: board + quick actions ---
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        header_row = QtWidgets.QHBoxLayout()
        header = QtWidgets.QLabel("Attack board")
        hfont = header.font()
        hfont.setPointSize(hfont.pointSize() + 1)
        hfont.setBold(True)
        header.setFont(hfont)
        header_row.addWidget(header)
        header_row.addStretch(1)
        left_layout.addLayout(header_row)

        overlay_layout = QtWidgets.QHBoxLayout()
        overlay_label = QtWidgets.QLabel("Overlay:")
        self.overlay_combo = QtWidgets.QComboBox()
        self.overlay_combo.addItems(["None", "Hit probability (%)", "Info gain (0–100)"])
        self.overlay_combo.currentIndexChanged.connect(self.update_board_view)
        overlay_layout.addWidget(overlay_label)
        overlay_layout.addWidget(self.overlay_combo)
        overlay_layout.addStretch(1)
        left_layout.addLayout(overlay_layout)

        board_container = QtWidgets.QWidget()
        board_container.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        board_layout = QtWidgets.QGridLayout(board_container)
        board_layout.setSpacing(2)
        board_layout.setContentsMargins(8, 8, 8, 8)
        board_layout.setAlignment(QtCore.Qt.AlignCenter)

        self._col_labels = []
        for c in range(self.board_size):
            lbl = QtWidgets.QLabel(chr(ord("A") + c))
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setStyleSheet(f"color: {Theme.TEXT_LABEL};")
            board_layout.addWidget(lbl, 0, c + 1)
            self._col_labels.append(lbl)
        for r in range(self.board_size):
            lbl = QtWidgets.QLabel(str(r + 1))
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setStyleSheet(f"color: {Theme.TEXT_LABEL};")
            board_layout.addWidget(lbl, r + 1, 0)
        self._row_labels = []
        for r in range(self.board_size):
            self._row_labels.append(board_layout.itemAtPosition(r + 1, 0).widget())

        cell_size = self._board_cell_size()
        self.cell_buttons = []
        for r in range(self.board_size):
            row = []
            for c in range(self.board_size):
                btn = QtWidgets.QPushButton("")
                btn.setFixedSize(cell_size, cell_size)
                btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
                btn.clicked.connect(self._make_cell_handler(r, c))
                row.append(btn)
                board_layout.addWidget(btn, r + 1, c + 1)
            self.cell_buttons.append(row)

        self.board_container = board_container
        self.board_layout = board_layout
        board_container.installEventFilter(self)
        left_layout.addWidget(board_container, stretch=1)

        quick_group = QtWidgets.QGroupBox("Quick actions")
        quick_layout = QtWidgets.QVBoxLayout(quick_group)

        quick_row = QtWidgets.QHBoxLayout()
        self.mark_hit_btn = QtWidgets.QPushButton("Hit (H)")
        self.mark_miss_btn = QtWidgets.QPushButton("Miss (M)")
        self.mark_clear_btn = QtWidgets.QPushButton("Clear (U)")
        self.mark_hit_btn.clicked.connect(lambda: self._set_active_cell_state(HIT))
        self.mark_miss_btn.clicked.connect(lambda: self._set_active_cell_state(MISS))
        self.mark_clear_btn.clicked.connect(lambda: self._set_active_cell_state(EMPTY))
        quick_row.addWidget(self.mark_hit_btn)
        quick_row.addWidget(self.mark_miss_btn)
        quick_row.addWidget(self.mark_clear_btn)
        quick_layout.addLayout(quick_row)

        history_row = QtWidgets.QHBoxLayout()
        self.undo_btn = QtWidgets.QPushButton("Undo (Ctrl+Z)")
        self.redo_btn = QtWidgets.QPushButton("Redo (Ctrl+Y)")
        self.undo_btn.clicked.connect(self.undo)
        self.redo_btn.clicked.connect(self.redo)
        history_row.addWidget(self.undo_btn)
        history_row.addWidget(self.redo_btn)
        quick_layout.addLayout(history_row)

        shortcuts_hint = QtWidgets.QLabel(
            "Left-click cycles unknown → hit → miss → unknown.\n"
            "Shift=Hit, Alt=Miss, Ctrl=Clear. H/M/U apply to last selected cell."
        )
        shortcuts_hint.setWordWrap(True)
        shortcuts_hint.setStyleSheet(f"color: {Theme.TEXT_LABEL};")
        quick_layout.addWidget(shortcuts_hint)

        left_layout.addWidget(quick_group)

        action_row = QtWidgets.QHBoxLayout()
        self.recompute_button = QtWidgets.QPushButton("Recompute now")
        self.recompute_button.clicked.connect(self.recompute)
        self.clear_attack_button = QtWidgets.QPushButton("Clear board (new game)")
        self.clear_attack_button.clicked.connect(self.clear_board)
        action_row.addWidget(self.recompute_button)
        action_row.addWidget(self.clear_attack_button)
        left_layout.addLayout(action_row)

        splitter.addWidget(left_panel)

        # --- Right: tabs ---
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        title = QtWidgets.QLabel("Attack assistant")
        tfont = title.font()
        tfont.setPointSize(tfont.pointSize() + 1)
        tfont.setBold(True)
        title.setFont(tfont)
        right_layout.addWidget(title)

        desc = QtWidgets.QLabel("Use the tabs below to review insights, models, ships, and stats.")
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {Theme.TEXT_LABEL};")
        right_layout.addWidget(desc)

        tabs = QtWidgets.QTabWidget()
        right_layout.addWidget(tabs, stretch=1)

        # Insights tab
        insights = QtWidgets.QWidget()
        insights_layout = QtWidgets.QVBoxLayout(insights)

        self.warning_label = QtWidgets.QLabel("")
        self.warning_label.setStyleSheet("color: #ff6b6b; font-weight: bold;")
        self.warning_label.setWordWrap(True)
        insights_layout.addWidget(self.warning_label)

        explain_group = QtWidgets.QGroupBox("Why this shot")
        explain_layout = QtWidgets.QVBoxLayout(explain_group)
        self.explain_table = QtWidgets.QTableWidget(0, 3)
        self.explain_table.setHorizontalHeaderLabels(["Cell", "p(hit)", "Score"])
        self.explain_table.verticalHeader().setVisible(False)
        self.explain_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.explain_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.explain_table.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.explain_table.horizontalHeader().setStretchLastSection(True)
        explain_layout.addWidget(self.explain_table)
        self.explain_summary_label = QtWidgets.QLabel("")
        self.explain_summary_label.setWordWrap(True)
        explain_layout.addWidget(self.explain_summary_label)
        insights_layout.addWidget(explain_group)

        whatif_group = QtWidgets.QGroupBox("What-if preview")
        whatif_layout = QtWidgets.QVBoxLayout(whatif_group)
        self.whatif_label = QtWidgets.QLabel("No preview yet.")
        self.whatif_label.setWordWrap(True)
        whatif_layout.addWidget(self.whatif_label)
        insights_layout.addWidget(whatif_group)

        self.summary_label = QtWidgets.QLabel("")
        self.summary_label.setWordWrap(True)
        insights_layout.addWidget(self.summary_label)

        self.world_mode_label = QtWidgets.QLabel("")
        self.world_mode_label.setWordWrap(True)
        insights_layout.addWidget(self.world_mode_label)

        self.best_label = QtWidgets.QLabel("Best guess: (none)")
        self.best_label.setWordWrap(True)
        insights_layout.addWidget(self.best_label)

        insights_layout.addStretch(1)
        tabs.addTab(wrap_scroll(insights), "Insights")

        # Models tab
        models = QtWidgets.QWidget()
        models_layout = QtWidgets.QVBoxLayout(models)

        model_group = QtWidgets.QGroupBox("Model selection")
        model_layout = QtWidgets.QVBoxLayout(model_group)
        self.auto_phase_rb = QtWidgets.QRadioButton("Auto (best by phase)")
        self.auto_overall_rb = QtWidgets.QRadioButton("Auto (best overall)")
        self.manual_rb = QtWidgets.QRadioButton("Manual")
        self.auto_phase_rb.setChecked(True)
        self.auto_phase_rb.toggled.connect(self._model_mode_changed)
        self.auto_overall_rb.toggled.connect(self._model_mode_changed)
        self.manual_rb.toggled.connect(self._model_mode_changed)
        model_layout.addWidget(self.auto_phase_rb)
        model_layout.addWidget(self.auto_overall_rb)
        model_layout.addWidget(self.manual_rb)

        self.model_combo = QtWidgets.QComboBox()
        for md in self.attack_model_defs:
            self.model_combo.addItem(md["name"], userData=md["key"])
        self.model_combo.currentIndexChanged.connect(self._manual_model_changed)
        model_layout.addWidget(self.model_combo)

        self.lock_model_cb = QtWidgets.QCheckBox("Lock model for this game")
        self.lock_model_cb.toggled.connect(self._lock_model_toggled)
        model_layout.addWidget(self.lock_model_cb)

        self.active_model_label = QtWidgets.QLabel("")
        self.active_model_label.setWordWrap(True)
        model_layout.addWidget(self.active_model_label)

        models_layout.addWidget(model_group)
        models_layout.addStretch(1)
        tabs.addTab(wrap_scroll(models), "Models")

        # Ships tab
        ships = QtWidgets.QWidget()
        ships_layout = QtWidgets.QVBoxLayout(ships)

        sunk_group = QtWidgets.QGroupBox("Mark ships confirmed sunk")
        sunk_layout = QtWidgets.QVBoxLayout(sunk_group)
        self.sunk_checkboxes = {}
        for ship in self.ship_ids:
            cb = QtWidgets.QCheckBox(self.ship_friendly_names[ship])
            cb.stateChanged.connect(self._make_sunk_handler(ship))
            sunk_layout.addWidget(cb)
            self.sunk_checkboxes[ship] = cb
        ships_layout.addWidget(sunk_group)

        assign_group = QtWidgets.QGroupBox("Assign hits to ships (optional)")
        assign_layout = QtWidgets.QVBoxLayout(assign_group)
        self.assign_none_rb = QtWidgets.QRadioButton("No assignment mode")
        self.assign_none_rb.setChecked(True)
        self.assign_none_rb.toggled.connect(self._assign_mode_changed)
        assign_layout.addWidget(self.assign_none_rb)
        self.assign_ship_rbs = {}
        for ship in self.ship_ids:
            rb = QtWidgets.QRadioButton(f"Assign hits to {self.ship_friendly_names[ship]}")
            rb.toggled.connect(self._assign_mode_changed)
            assign_layout.addWidget(rb)
            self.assign_ship_rbs[ship] = rb
        ships_layout.addWidget(assign_group)

        status_group = QtWidgets.QGroupBox("Ship status (sunk probability)")
        status_layout = QtWidgets.QVBoxLayout(status_group)
        self.ship_status_labels = {}
        for ship in self.ship_ids:
            lbl = QtWidgets.QLabel(f"{self.ship_friendly_names[ship]}: unknown")
            status_layout.addWidget(lbl)
            self.ship_status_labels[ship] = lbl
        ships_layout.addWidget(status_group)

        ships_layout.addStretch(1)
        tabs.addTab(wrap_scroll(ships), "Ships")

        # Stats tab
        stats_page = QtWidgets.QWidget()
        stats_layout = QtWidgets.QVBoxLayout(stats_page)

        stats_group = QtWidgets.QGroupBox("Game result & win rate")
        stats_group_layout = QtWidgets.QVBoxLayout(stats_group)
        self.stats_label = QtWidgets.QLabel(self.stats.summary_text())
        self.stats_label.setWordWrap(True)
        stats_group_layout.addWidget(self.stats_label)

        btn_row = QtWidgets.QHBoxLayout()
        self.win_button = QtWidgets.QPushButton("Record WIN + new game")
        self.loss_button = QtWidgets.QPushButton("Record LOSS + new game")
        self.win_button.clicked.connect(self._record_win)
        self.loss_button.clicked.connect(self._record_loss)
        btn_row.addWidget(self.win_button)
        btn_row.addWidget(self.loss_button)
        stats_group_layout.addLayout(btn_row)
        stats_layout.addWidget(stats_group)

        self.win_prob_label = QtWidgets.QLabel("Win Probability: N/A")
        self.win_prob_label.setStyleSheet(f"color: {Theme.HIGHLIGHT}; font-weight: bold; font-size: 14px;")
        stats_layout.addWidget(self.win_prob_label)

        stats_layout.addStretch(1)
        tabs.addTab(wrap_scroll(stats_page), "Stats")

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([800, 450])

        self._setup_shortcuts()
        self._resize_board()

    def eventFilter(self, obj, event):
        if obj is getattr(self, "board_container", None) and event.type() == QtCore.QEvent.Resize:
            self._resize_board()
        return super().eventFilter(obj, event)

    def _resize_board(self) -> None:
        if not hasattr(self, "board_container") or not hasattr(self, "board_layout"):
            return
        size = self.board_container.size()
        spacing = self.board_layout.spacing()
        l, t, r, b = self.board_layout.getContentsMargins()
        n = self.board_size + 1
        avail_w = max(1, size.width() - l - r - spacing * (n - 1))
        avail_h = max(1, size.height() - t - b - spacing * (n - 1))
        cell = max(12, min(avail_w // n, avail_h // n))

        for lbl in getattr(self, "_col_labels", []):
            if lbl:
                lbl.setFixedSize(cell, cell)
        for lbl in getattr(self, "_row_labels", []):
            if lbl:
                lbl.setFixedSize(cell, cell)
        for row in self.cell_buttons:
            for btn in row:
                btn.setFixedSize(cell, cell)

    def _setup_shortcuts(self):
        self.shortcut_hit = QtWidgets.QShortcut(QtGui.QKeySequence("H"), self)
        self.shortcut_miss = QtWidgets.QShortcut(QtGui.QKeySequence("M"), self)
        self.shortcut_clear = QtWidgets.QShortcut(QtGui.QKeySequence("U"), self)
        self.shortcut_cycle = QtWidgets.QShortcut(QtGui.QKeySequence("Space"), self)
        self.shortcut_undo = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Z"), self)
        self.shortcut_redo = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Y"), self)

        for sc in (
            self.shortcut_hit,
            self.shortcut_miss,
            self.shortcut_clear,
            self.shortcut_cycle,
            self.shortcut_undo,
            self.shortcut_redo,
        ):
            sc.setContext(QtCore.Qt.WidgetWithChildrenShortcut)

        self.shortcut_hit.activated.connect(lambda: self._set_active_cell_state(HIT))
        self.shortcut_miss.activated.connect(lambda: self._set_active_cell_state(MISS))
        self.shortcut_clear.activated.connect(lambda: self._set_active_cell_state(EMPTY))
        self.shortcut_cycle.activated.connect(self._cycle_active_cell_state)
        self.shortcut_undo.activated.connect(self.undo)
        self.shortcut_redo.activated.connect(self.redo)

    def _update_model_controls(self):
        self.auto_phase_rb.blockSignals(True)
        self.auto_overall_rb.blockSignals(True)
        self.manual_rb.blockSignals(True)
        self.model_combo.blockSignals(True)
        self.lock_model_cb.blockSignals(True)

        self.auto_phase_rb.setChecked(self.auto_mode == self.AUTO_MODE_PHASE)
        self.auto_overall_rb.setChecked(self.auto_mode == self.AUTO_MODE_OVERALL)
        self.manual_rb.setChecked(self.auto_mode == self.AUTO_MODE_MANUAL)
        self.model_combo.setEnabled(self.auto_mode == self.AUTO_MODE_MANUAL)

        idx = 0
        for i in range(self.model_combo.count()):
            if self.model_combo.itemData(i) == self.manual_model_key:
                idx = i
                break
        self.model_combo.setCurrentIndex(idx)
        self.lock_model_cb.setChecked(bool(self.lock_model_for_game))

        self.auto_phase_rb.blockSignals(False)
        self.auto_overall_rb.blockSignals(False)
        self.manual_rb.blockSignals(False)
        self.model_combo.blockSignals(False)
        self.lock_model_cb.blockSignals(False)

    def _model_mode_changed(self):
        if self.auto_phase_rb.isChecked():
            self.auto_mode = self.AUTO_MODE_PHASE
        elif self.auto_overall_rb.isChecked():
            self.auto_mode = self.AUTO_MODE_OVERALL
        else:
            self.auto_mode = self.AUTO_MODE_MANUAL
        self.model_combo.setEnabled(self.auto_mode == self.AUTO_MODE_MANUAL)
        self.recompute()

    def _manual_model_changed(self, index: int):
        key = self.model_combo.itemData(index)
        if isinstance(key, str) and key:
            self.manual_model_key = key
        if self.auto_mode == self.AUTO_MODE_MANUAL:
            self.recompute()

    def _lock_model_toggled(self, checked: bool):
        self.lock_model_for_game = bool(checked)
        if not checked:
            self.locked_model_key = None
        self.recompute()

    def _snapshot_state(self) -> Dict[str, object]:
        return {
            "board": [row[:] for row in self.board],
            "confirmed_sunk": set(self.confirmed_sunk),
            "assigned_hits": {s: set(coords) for s, coords in self.assigned_hits.items()},
        }

    def _restore_snapshot(self, snap: Dict[str, object]) -> None:
        self._suspend_history = True
        board = snap.get("board")
        if isinstance(board, list) and len(board) == self.board_size:
            self.board = [list(row) for row in board]
        confirmed = snap.get("confirmed_sunk")
        if isinstance(confirmed, set):
            self.confirmed_sunk = set(confirmed)
        elif isinstance(confirmed, list):
            self.confirmed_sunk = set(confirmed)
        assigned = snap.get("assigned_hits")
        if isinstance(assigned, dict):
            self.assigned_hits = {s: set(assigned.get(s, set())) for s in self.ship_ids}

        for ship, cb in self.sunk_checkboxes.items():
            cb.blockSignals(True)
            cb.setChecked(ship in self.confirmed_sunk)
            cb.blockSignals(False)

        self._suspend_history = False
        self.recompute()

    def _push_history(self):
        if self._suspend_history:
            return
        self.undo_stack.append(self._snapshot_state())
        self.redo_stack.clear()
        self._update_history_buttons()

    def _update_history_buttons(self):
        self.undo_btn.setEnabled(bool(self.undo_stack))
        self.redo_btn.setEnabled(bool(self.redo_stack))

    def undo(self):
        if not self.undo_stack:
            return
        current = self._snapshot_state()
        snap = self.undo_stack.pop()
        self.redo_stack.append(current)
        self._restore_snapshot(snap)
        self._update_history_buttons()

    def redo(self):
        if not self.redo_stack:
            return
        current = self._snapshot_state()
        snap = self.redo_stack.pop()
        self.undo_stack.append(current)
        self._restore_snapshot(snap)
        self._update_history_buttons()

    def _cycle_active_cell_state(self):
        if self.active_cell is None:
            return
        r, c = self.active_cell
        self._cycle_cell_state(r, c)

    def _set_active_cell_state(self, state: str):
        if self.active_cell is None:
            return
        r, c = self.active_cell
        self._apply_cell_state(r, c, state)

    def _cycle_cell_state(self, r: int, c: int):
        current = self.board[r][c]
        if current == EMPTY:
            new_state = HIT
        elif current == HIT:
            new_state = MISS
        else:
            new_state = EMPTY
        self._apply_cell_state(r, c, new_state)

    def _apply_cell_state(self, r: int, c: int, new_state: str):
        current = self.board[r][c]
        if current == new_state:
            return
        self._push_history()
        self.board[r][c] = new_state
        if new_state == HIT and self.assign_mode_ship is not None:
            self._toggle_assignment(self.assign_mode_ship, r, c)
        elif new_state != HIT:
            for ship in self.ship_ids:
                self.assigned_hits[ship].discard((r, c))
        self.recompute()

    def _toggle_assignment(self, ship: str, r: int, c: int):
        sset = self.assigned_hits[ship]
        if (r, c) in sset:
            sset.remove((r, c))
        else:
            for other in self.ship_ids:
                if other != ship:
                    self.assigned_hits[other].discard((r, c))
            sset.add((r, c))

    def _expected_ship_cells(self) -> int:
        total = 0
        for spec in self.layout.ships:
            if spec.kind == "line":
                total += int(spec.length or 0)
            else:
                total += len(spec.cells or [])
        return total

    def _is_target_mode(self, unknown_cells: Sequence[Tuple[int, int]]) -> bool:
        if not unknown_cells:
            return False
        has_any_hit = any(
            self.board[r][c] == HIT for r in range(self.board_size) for c in range(self.board_size)
        )
        max_p = max(self.cell_probs[cell_index(r, c, self.board_size)] for r, c in unknown_cells)
        return has_any_hit and (max_p > 0.30)

    def _get_model_name(self, key: Optional[str]) -> str:
        if not key:
            return "Unknown"
        for md in self.attack_model_defs:
            if md.get("key") == key:
                return md.get("name") or key
        return key

    def _resolve_active_model(self, phase: str) -> Tuple[str, str, Optional[str]]:
        best_overall, best_by_phase = load_best_models(self.MODEL_STATS_PATH, self.layout)
        reason = ""
        candidate = None

        if self.auto_mode == self.AUTO_MODE_MANUAL:
            candidate = self.manual_model_key
            reason = "Manual selection"
        elif self.auto_mode == self.AUTO_MODE_OVERALL:
            candidate = best_overall
            reason = "Auto (best overall)"
        else:
            candidate = best_by_phase.get(phase) or best_overall
            reason = "Auto (best by phase)"

        if not candidate:
            candidate = self.manual_model_key or "two_ply"
            reason += " (fallback)"

        if self.lock_model_for_game:
            if not self.locked_model_key:
                self.locked_model_key = candidate
            candidate = self.locked_model_key
            reason += " (locked)"
        else:
            self.locked_model_key = None

        return candidate, reason, best_overall

    def _rank_cells_for_model(
        self,
        model_key: str,
        unknown_cells: Sequence[Tuple[int, int]],
    ) -> Tuple[List[Dict[str, object]], str, bool, str]:
        score_label = "Score"
        higher_better = True
        note = ""

        if not unknown_cells:
            return [], score_label, higher_better, note

        N = self.num_world_samples
        is_target_mode = self._is_target_mode(unknown_cells)

        def p_hit(r: int, c: int) -> float:
            return self.cell_probs[cell_index(r, c, self.board_size)]

        candidates: List[Dict[str, object]] = []

        if model_key == "two_ply":
            score_label = "2-ply"
            higher_better = False
            return self._two_ply_candidates(unknown_cells), score_label, higher_better, note

        if model_key == "entropy1":
            score_label = "Info (1-ply)"

            def score_entropy(r: int, c: int) -> float:
                idx = cell_index(r, c, self.board_size)
                n_hit = self.cell_hit_counts[idx]
                n_miss = N - n_hit
                if n_hit == 0 or n_miss == 0:
                    return -1.0e9
                p_h = n_hit / N
                p_m = 1.0 - p_h
                return -(p_h * math.log(max(1, n_hit)) + p_m * math.log(max(1, n_miss)))

            for r, c in unknown_cells:
                candidates.append({"cell": (r, c), "p_hit": p_hit(r, c), "score": score_entropy(r, c)})
            return candidates, score_label, higher_better, note

        if model_key == "greedy":
            score_label = "p(hit)"
            for r, c in unknown_cells:
                candidates.append({"cell": (r, c), "p_hit": p_hit(r, c), "score": p_hit(r, c)})
            return candidates, score_label, higher_better, note

        if model_key == "hybrid_phase":
            if is_target_mode:
                note = "Target mode: Greedy"
                return self._rank_cells_for_model("greedy", unknown_cells)[:-1] + (note,)
            note = "Hunt mode: Info gain"
            return self._rank_cells_for_model("entropy1", unknown_cells)[:-1] + (note,)

        if model_key == "weighted_sample":
            score_label = "Weight"
            for r, c in unknown_cells:
                candidates.append({"cell": (r, c), "p_hit": p_hit(r, c), "score": p_hit(r, c)})
            note = "Stochastic: weighted by p(hit)"
            return candidates, score_label, higher_better, note

        if model_key == "random":
            score_label = "Uniform"
            for r, c in unknown_cells:
                candidates.append({"cell": (r, c), "p_hit": p_hit(r, c), "score": 1.0})
            note = "Random among unknown cells"
            return candidates, score_label, higher_better, note

        if model_key == "softmax_greedy":
            score_label = "Weight"
            if is_target_mode:
                note = "Target mode: Greedy"
                return self._rank_cells_for_model("greedy", unknown_cells)[:-1] + (note,)
            temp = 0.10
            specs = PARAM_SPECS.get("softmax_greedy", [])
            if specs:
                temp = float(specs[0].get("default", temp))
            ps = [p_hit(r, c) for r, c in unknown_cells]
            pmax = max(ps) if ps else 0.0
            denom = max(1e-6, temp)
            for (r, c), p in zip(unknown_cells, ps):
                weight = math.exp((p - pmax) / denom)
                candidates.append({"cell": (r, c), "p_hit": p, "score": weight})
            note = f"Stochastic: softmax T={temp:.2f}"
            return candidates, score_label, higher_better, note

        if model_key == "parity_greedy":
            score_label = "p(hit)"
            evens = [(r, c) for (r, c) in unknown_cells if (r + c) % 2 == 0]
            odds = [(r, c) for (r, c) in unknown_cells if (r + c) % 2 == 1]
            even_mass = sum(p_hit(r, c) for r, c in evens)
            odd_mass = sum(p_hit(r, c) for r, c in odds)
            prefer_even = even_mass >= odd_mass
            for r, c in unknown_cells:
                if ((r + c) % 2 == 0) == prefer_even:
                    candidates.append({"cell": (r, c), "p_hit": p_hit(r, c), "score": p_hit(r, c)})
            note = "Parity favored"
            return candidates, score_label, higher_better, note

        if model_key == "random_checkerboard":
            if is_target_mode:
                note = "Target mode: Greedy"
                return self._rank_cells_for_model("greedy", unknown_cells)[:-1] + (note,)
            score_label = "Parity"
            for r, c in unknown_cells:
                if (r + c) % 2 == 0:
                    candidates.append({"cell": (r, c), "p_hit": p_hit(r, c), "score": 1.0})
            note = "Random among parity cells"
            return candidates, score_label, higher_better, note

        if model_key == "systematic_checkerboard":
            if is_target_mode:
                note = "Target mode: Greedy"
                return self._rank_cells_for_model("greedy", unknown_cells)[:-1] + (note,)
            score_label = "Order"
            sorted_cells = sorted(unknown_cells, key=lambda p: (p[0], p[1]))
            whites = [p for p in sorted_cells if (p[0] + p[1]) % 2 == 0]
            for rank, (r, c) in enumerate(whites):
                score = float(len(whites) - rank)
                candidates.append({"cell": (r, c), "p_hit": p_hit(r, c), "score": score})
            note = "Systematic parity sweep"
            return candidates, score_label, higher_better, note

        if model_key == "diagonal_stripe":
            if is_target_mode:
                note = "Target mode: Greedy"
                return self._rank_cells_for_model("greedy", unknown_cells)[:-1] + (note,)
            score_label = "Stripe"
            for r, c in unknown_cells:
                if (r - c) % 4 == 0:
                    score = 2.0
                elif (r - c) % 2 == 0:
                    score = 1.0
                else:
                    score = 0.0
                candidates.append({"cell": (r, c), "p_hit": p_hit(r, c), "score": score})
            note = "Diagonal stripe priority"
            return candidates, score_label, higher_better, note

        if model_key == "dynamic_parity":
            if is_target_mode:
                note = "Target mode: Greedy"
                return self._rank_cells_for_model("greedy", unknown_cells)[:-1] + (note,)
            score_label = "p(hit)"
            alive = [s for s in self.ship_ids if s not in self.confirmed_sunk]
            step = 2
            if len(alive) == 1 and alive[0] == "line3":
                step = 3
            best_cells: List[Tuple[int, int]] = []
            best_mass = -1.0
            for color in range(step):
                cells = [(r, c) for (r, c) in unknown_cells if (r + c) % step == color]
                if not cells:
                    continue
                mass = sum(p_hit(r, c) for r, c in cells)
                if mass > best_mass:
                    best_mass = mass
                    best_cells = cells
            for r, c in best_cells:
                candidates.append({"cell": (r, c), "p_hit": p_hit(r, c), "score": p_hit(r, c)})
            note = f"Dynamic parity step={step}"
            return candidates, score_label, higher_better, note

        if model_key == "endpoint_phase":
            if not is_target_mode:
                note = "Hunt mode: Info gain"
                return self._rank_cells_for_model("entropy1", unknown_cells)[:-1] + (note,)

            score_label = "Endpoint score"
            hits_all = [(r, c) for r in range(self.board_size) for c in range(self.board_size) if self.board[r][c] == HIT]
            hit_set = set(hits_all)

            def neighbors4(rr: int, cc: int):
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                        yield nr, nc

            components: List[List[Tuple[int, int]]] = []
            visited: Set[Tuple[int, int]] = set()
            for start in hits_all:
                if start in visited:
                    continue
                stack = [start]
                visited.add(start)
                comp: List[Tuple[int, int]] = []
                while stack:
                    cur = stack.pop()
                    comp.append(cur)
                    for nr, nc in neighbors4(cur[0], cur[1]):
                        if (nr, nc) in hit_set and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            stack.append((nr, nc))
                components.append(comp)

            frontier: Set[Tuple[int, int]] = set()
            endpoint_cells: Set[Tuple[int, int]] = set()
            for comp in components:
                for hr, hc in comp:
                    for nr, nc in neighbors4(hr, hc):
                        if self.board[nr][nc] == EMPTY:
                            frontier.add((nr, nc))

                aligned_row = (len(comp) >= 2) and all(r == comp[0][0] for r, _ in comp)
                aligned_col = (len(comp) >= 2) and all(c == comp[0][1] for _, c in comp)
                if aligned_row:
                    r0 = comp[0][0]
                    cols = [c for _, c in comp]
                    for cand_c in (min(cols) - 1, max(cols) + 1):
                        if 0 <= cand_c < self.board_size and self.board[r0][cand_c] == EMPTY:
                            endpoint_cells.add((r0, cand_c))
                elif aligned_col:
                    c0 = comp[0][1]
                    rows = [r for r, _ in comp]
                    for cand_r in (min(rows) - 1, max(rows) + 1):
                        if 0 <= cand_r < self.board_size and self.board[cand_r][c0] == EMPTY:
                            endpoint_cells.add((cand_r, c0))

            w_prob = 1.0
            w_neighbor = 0.2
            w_endpoint = 0.4
            specs = PARAM_SPECS.get("endpoint_phase", [])
            if len(specs) >= 3:
                w_prob = float(specs[0].get("default", w_prob))
                w_neighbor = float(specs[1].get("default", w_neighbor))
                w_endpoint = float(specs[2].get("default", w_endpoint))

            for r, c in frontier:
                hit_neighbors = sum(1 for nr, nc in neighbors4(r, c) if (nr, nc) in hit_set)
                is_endpoint = 1.0 if (r, c) in endpoint_cells else 0.0
                score = (w_prob * p_hit(r, c)) + (w_neighbor * hit_neighbors) + (w_endpoint * is_endpoint)
                candidates.append({"cell": (r, c), "p_hit": p_hit(r, c), "score": score})

            if not candidates:
                note = "Endpoint fallback: Greedy"
                return self._rank_cells_for_model("greedy", unknown_cells)[:-1] + (note,)

            note = "Target mode: endpoint scoring"
            return candidates, score_label, higher_better, note

        if model_key == "center_weighted":
            score_label = "Center score"
            center = (self.board_size - 1) / 2.0
            for r, c in unknown_cells:
                dist2 = (r - center) ** 2 + (c - center) ** 2
                score = p_hit(r, c) / (1.0 + 0.25 * dist2)
                candidates.append({"cell": (r, c), "p_hit": p_hit(r, c), "score": score})
            return candidates, score_label, higher_better, note

        if model_key == "adaptive_skew":
            score_label = "Skew score"
            center = (self.board_size - 1) / 2.0
            unknown_ratio = len(unknown_cells) / (self.board_size * self.board_size)
            for r, c in unknown_cells:
                dist = math.sqrt((r - center) ** 2 + (c - center) ** 2)
                norm_dist = dist / math.sqrt(2 * center ** 2)
                penalty = 0.20 * norm_dist if unknown_ratio > 0.5 else 0.0
                score = p_hit(r, c) - penalty
                candidates.append({"cell": (r, c), "p_hit": p_hit(r, c), "score": score})
            return candidates, score_label, higher_better, note

        if model_key == "thompson_world":
            if is_target_mode:
                note = "Target mode: Greedy"
                return self._rank_cells_for_model("greedy", unknown_cells)[:-1] + (note,)
            score_label = "Sample"
            if self.world_masks:
                rng = random.Random(0)
                chosen = rng.choice(self.world_masks)
                for r, c in unknown_cells:
                    if (chosen >> cell_index(r, c, self.board_size)) & 1:
                        candidates.append({"cell": (r, c), "p_hit": p_hit(r, c), "score": 1.0})
                note = "Sampled one world"
            if not candidates:
                for r, c in unknown_cells:
                    candidates.append({"cell": (r, c), "p_hit": p_hit(r, c), "score": 1.0})
                note = "Fallback: uniform"
            return candidates, score_label, higher_better, note

        # Default: greedy-like probability
        score_label = "p(hit)"
        for r, c in unknown_cells:
            candidates.append({"cell": (r, c), "p_hit": p_hit(r, c), "score": p_hit(r, c)})
        return candidates, score_label, higher_better, note

    def _two_ply_candidates(self, unknown_cells: Sequence[Tuple[int, int]]) -> List[Dict[str, object]]:
        if not self.world_masks:
            return []
        N = len(self.world_masks)
        if N == 0:
            return []

        known_mask = 0
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r][c] != EMPTY:
                    known_mask |= 1 << cell_index(r, c, self.board_size)

        scores1: Dict[int, float] = {}
        candidates = []
        for r, c in unknown_cells:
            idx = cell_index(r, c, self.board_size)
            n_hit = self.cell_hit_counts[idx]
            n_miss = N - n_hit
            score = (n_hit * n_hit + n_miss * n_miss) / N
            scores1[idx] = score
            candidates.append(idx)

        candidates.sort(key=lambda i: scores1[i])
        top_candidates = candidates[: min(24, len(candidates))]

        entries: List[Dict[str, object]] = []
        for idx in top_candidates:
            bit = 1 << idx
            worlds_hit = [wm for wm in self.world_masks if wm & bit]
            worlds_miss = [wm for wm in self.world_masks if not (wm & bit)]
            Nh = len(worlds_hit)
            Nm = len(worlds_miss)
            if Nh + Nm == 0:
                continue
            p_h = Nh / (Nh + Nm)
            known_after = known_mask | bit
            Eh = compute_min_expected_worlds_after_one_shot(worlds_hit, known_after, self.board_size) if Nh > 0 else 0.0
            Em = compute_min_expected_worlds_after_one_shot(worlds_miss, known_after, self.board_size) if Nm > 0 else 0.0
            two_ply = p_h * Eh + (1.0 - p_h) * Em
            r = idx // self.board_size
            c = idx % self.board_size
            entries.append({"cell": (r, c), "p_hit": p_h, "score": two_ply})

        entries.sort(key=lambda e: e["score"])
        return entries

    def _update_explanation_panel(
        self,
        candidates: List[Dict[str, object]],
        score_label: str,
        higher_better: bool,
        note: str,
    ) -> None:
        self.explain_table.setRowCount(0)
        self.explain_table.setHorizontalHeaderLabels(["Cell", "p(hit)", score_label])

        if not candidates:
            self.explain_summary_label.setText("No candidates (board full or no consistent layouts).")
            return

        candidates_sorted = sorted(
            candidates,
            key=lambda e: (e["score"], e["p_hit"]),
            reverse=higher_better,
        )
        top = candidates_sorted[:8]

        self.explain_table.setRowCount(len(top))
        for row, entry in enumerate(top):
            r, c = entry["cell"]
            cell_txt = f"{r + 1}{chr(ord('A') + c)}"
            p_hit = entry.get("p_hit", 0.0)
            score = entry.get("score", 0.0)
            self.explain_table.setItem(row, 0, QtWidgets.QTableWidgetItem(cell_txt))
            self.explain_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{p_hit:.3f}"))
            self.explain_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{score:.3f}"))

        if note:
            self.explain_summary_label.setText(note)
        else:
            self.explain_summary_label.setText("")

    def _update_whatif_preview(self):
        if self.game_over or self.num_world_samples <= 0:
            self.whatif_label.setText("No preview available.")
            return

        target_cell = None
        if self.active_cell is not None:
            r, c = self.active_cell
            if self.board[r][c] == EMPTY:
                target_cell = (r, c)
        if target_cell is None and self.best_cells:
            target_cell = self.best_cells[0]

        if target_cell is None:
            self.whatif_label.setText("No preview available.")
            return

        r, c = target_cell
        idx = cell_index(r, c, self.board_size)
        n_hit = self.cell_hit_counts[idx]
        n_miss = self.num_world_samples - n_hit
        p_hit = (n_hit / self.num_world_samples) if self.num_world_samples > 0 else 0.0
        expected_worlds = 0.0
        if self.num_world_samples > 0:
            expected_worlds = (n_hit * n_hit + n_miss * n_miss) / self.num_world_samples
        info_gain = self.num_world_samples - expected_worlds
        source = "active cell" if target_cell == self.active_cell else "best suggestion"
        cell_label = f"{r + 1}{chr(ord('A') + c)}"
        self.whatif_label.setText(
            f"Using {source} {cell_label}:\n"
            f"p(hit)={p_hit:.3f}, worlds if hit={n_hit}, if miss={n_miss}\n"
            f"Expected info gain (1-ply): {info_gain:.1f}"
        )

    def _compute_warnings(self) -> str:
        warnings = []
        total_hits = sum(row.count(HIT) for row in self.board)
        expected_cells = self._expected_ship_cells()
        if total_hits > expected_cells:
            warnings.append("Hits exceed total ship cells for this layout.")
        for ship in self.ship_ids:
            assigned = len(self.assigned_hits.get(ship, set()))
            ship_size = 0
            for spec in self.layout.ships:
                if spec.instance_id == ship:
                    ship_size = spec.length if spec.kind == "line" else len(spec.cells or [])
                    break
            if assigned > ship_size > 0:
                warnings.append(f"Assigned hits exceed size for {self.ship_friendly_names[ship]}.")
        if self.num_world_samples == 0:
            warnings.append("No consistent layouts found (check hits/misses and sunk flags).")
        return " ".join(warnings)

    def save_state(self, path: Optional[str] = None):
        """
        Persist the current attack game state so that if the app closes
        mid-game, we can resume without re-entering hits.
        """
        if path is None:
            path = self.STATE_PATH

        # Serialize assigned_hits as lists of [r, c]
        assigned_hits_serializable = {
            ship: [[r, c] for (r, c) in sorted(list(coords))]
            for ship, coords in self.assigned_hits.items()
            if coords
        }

        state = {
            "layout_id": self.layout.layout_id,
            "layout_version": self.layout.layout_version,
            "layout_hash": self.layout.layout_hash,
            "board": self.board,  # board_size x board_size of ".", "o", "x"
            "confirmed_sunk": list(self.confirmed_sunk),
            "assigned_hits": assigned_hits_serializable,
            "overlay_mode": self.overlay_combo.currentIndex(),
            "game_over": self.game_over,
            "auto_mode": self.auto_mode,
            "manual_model_key": self.manual_model_key,
            "lock_model_for_game": bool(self.lock_model_cb.isChecked()),
            "locked_model_key": self.locked_model_key,
        }
        save_layout_state(path, self.layout, state)

    def load_state(self, path: Optional[str] = None):
        """
        Restore a previous in-progress attack game if one exists.

        If the saved state indicates the game was already over (all ships
        marked sunk), we treat it as finished and start fresh.
        """
        if path is None:
            path = self.STATE_PATH
        data, _raw = load_layout_state(path, self.layout)
        if not data:
            return

        # If the last saved game was already marked as over,
        # treat that as a finished game and do NOT restore it.
        if data.get("game_over", False):
            delete_layout_state(path, self.layout)
            return

        board = data.get("board")
        if not isinstance(board, list) or len(board) != self.board_size:
            return

        # Basic sanity check on board shape
        ok = True
        for row in board:
            if not isinstance(row, list) or len(row) != self.board_size:
                ok = False
                break
        if not ok:
            return

        # Restore board
        self.board = board

        # Restore confirmed_sunk
        conf = data.get("confirmed_sunk", [])
        self.confirmed_sunk = set(s for s in conf if s in self.ship_ids)

        # Sync sunk checkboxes
        for ship, cb in self.sunk_checkboxes.items():
            cb.blockSignals(True)
            cb.setChecked(ship in self.confirmed_sunk)
            cb.blockSignals(False)

        # Restore assigned_hits
        self.assigned_hits = {s: set() for s in self.ship_ids}
        ah = data.get("assigned_hits", {})
        if isinstance(ah, dict):
            for ship, coords in ah.items():
                if ship not in self.assigned_hits:
                    continue
                if not isinstance(coords, list):
                    continue
                sset = set()
                for pair in coords:
                    if (
                        isinstance(pair, list)
                        and len(pair) == 2
                        and isinstance(pair[0], int)
                        and isinstance(pair[1], int)
                    ):
                        r, c = pair
                        if 0 <= r < self.board_size and 0 <= c < self.board_size:
                            sset.add((r, c))
                self.assigned_hits[ship] = sset

        # Restore overlay mode (optional)
        overlay_mode = data.get("overlay_mode")
        if isinstance(overlay_mode, int) and 0 <= overlay_mode < self.overlay_combo.count():
            self.overlay_combo.setCurrentIndex(overlay_mode)

        auto_mode = data.get("auto_mode")
        if auto_mode in {self.AUTO_MODE_PHASE, self.AUTO_MODE_OVERALL, self.AUTO_MODE_MANUAL}:
            self.auto_mode = auto_mode
        manual_key = data.get("manual_model_key")
        if isinstance(manual_key, str):
            known_keys = {md["key"] for md in self.attack_model_defs}
            if manual_key in known_keys:
                self.manual_model_key = manual_key
        lock_flag = data.get("lock_model_for_game")
        if isinstance(lock_flag, bool):
            self.lock_model_for_game = lock_flag
        locked_key = data.get("locked_model_key")
        if isinstance(locked_key, str):
            self.locked_model_key = locked_key
        self._update_model_controls()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._update_history_buttons()

    def _record_game_result(self, win: bool):
        self.stats.record_game(win)
        self.stats_label.setText(self.stats.summary_text())
        # Start a completely fresh board for the next game
        self.clear_board()

    def _record_win(self):
        self._record_game_result(True)

    def _record_loss(self):
        self._record_game_result(False)

    def _assign_mode_changed(self):
        if self.assign_none_rb.isChecked():
            self.assign_mode_ship = None
        else:
            for ship, rb in self.assign_ship_rbs.items():
                if rb.isChecked():
                    self.assign_mode_ship = ship
                    break
        self.update_board_view()

    def _make_cell_handler(self, r: int, c: int):
        def handler():
            self.active_cell = (r, c)
            modifiers = QtWidgets.QApplication.keyboardModifiers()

            if (
                self.assign_mode_ship is not None
                and self.board[r][c] == HIT
                and modifiers == QtCore.Qt.NoModifier
            ):
                self._push_history()
                self._toggle_assignment(self.assign_mode_ship, r, c)
                self.recompute()
                return

            if modifiers & QtCore.Qt.ShiftModifier:
                new_state = HIT
            elif modifiers & QtCore.Qt.AltModifier:
                new_state = MISS
            elif modifiers & QtCore.Qt.ControlModifier:
                new_state = EMPTY
            else:
                current = self.board[r][c]
                if current == EMPTY:
                    new_state = HIT
                elif current == HIT:
                    new_state = MISS
                else:
                    new_state = EMPTY

            self._apply_cell_state(r, c, new_state)

        return handler

    def _make_sunk_handler(self, ship: str):
        def handler(state: int):
            self._push_history()
            if state == QtCore.Qt.Checked:
                self.confirmed_sunk.add(ship)
            else:
                self.confirmed_sunk.discard(ship)
            self.recompute()

        return handler

    def clear_board(self):
        self._push_history()
        self.board = create_board(self.board_size)
        self.confirmed_sunk.clear()
        for cb in self.sunk_checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)
        for ship in self.ship_ids:
            self.assigned_hits[ship].clear()
        self.assign_none_rb.setChecked(True)
        self.game_over = False
        self.active_cell = None
        if self.lock_model_cb.isChecked():
            self.lock_model_cb.blockSignals(True)
            self.lock_model_cb.setChecked(False)
            self.lock_model_cb.blockSignals(False)
        self.locked_model_key = None
        self.recompute()

        # Optional: also wipe any saved in-progress state
        delete_layout_state(self.STATE_PATH, self.layout)

    def _are_all_ships_sunk(self) -> bool:
        # Only trust the user's explicit checkboxes for "game over".
        return len(self.confirmed_sunk) == len(self.ship_ids)

    # In AttackTab class, add the prediction logic:
    def update_win_prediction(self, defense_tab):
        """Simulate Me vs Opponent."""
        if not self.world_masks or not defense_tab.layout_board:
            self.win_prob_label.setText("Win Prob: N/A (Need Defense Layout)")
            return

        # 1. Estimate MY remaining shots
        my_rem_samples = []
        rng = random.Random()
        # Sample up to 10 worlds
        sample_worlds = rng.sample(self.world_masks, min(10, len(self.world_masks)))

        for w_mask in sample_worlds:
            # Quick sim of greedy solver vs this world
            # Copy board
            sim_board = [row[:] for row in self.board]
            shots = 0
            rem_targets = bin(w_mask).count("1") - sum(row.count(HIT) for row in self.board)

            while rem_targets > 0 and shots < self.board_size * self.board_size:
                # Simple greedy selection on sim_board
                # (You can copy the logic from _choose_next_shot_for_strategy here
                # or make a fast helper. For speed, just pick random unknown cell)
                unknown = [
                    (r, c)
                    for r in range(self.board_size)
                    for c in range(self.board_size)
                    if sim_board[r][c] == EMPTY
                ]
                if not unknown:
                    break
                r, c = rng.choice(unknown)  # Fast approximation
                sim_board[r][c] = HIT if (w_mask & (1 << cell_index(r, c, self.board_size))) else MISS
                if sim_board[r][c] == HIT:
                    rem_targets -= 1
                shots += 1
            my_rem_samples.append(shots)

        my_avg_rem = sum(my_rem_samples) / len(my_rem_samples)
        my_total = (sum(row.count(HIT) + row.count(MISS) for row in self.board)) + my_avg_rem

        # 2. Estimate OPPONENT remaining shots
        # Create mask from defense tab layout
        layout_mask = 0
        for r in range(self.board_size):
            for c in range(self.board_size):
                if defense_tab.layout_board[r][c] == HAS_SHIP:
                    layout_mask |= (1 << cell_index(r, c, self.board_size))

        base_heat = [
            build_base_heat(defense_tab.hit_counts_phase[p], defense_tab.miss_counts_phase[p], self.board_size)
            for p in range(4)
        ]

        opp_rem_samples = []
        for _ in range(10):
            # Run simulation starting from current defense board state
            s_taken, _ = simulate_enemy_game_phase(
                layout_mask,
                base_heat,
                defense_tab.disp_counts,
                "seq",
                rng,
                board_size=self.board_size,
                initial_shot_board=defense_tab.shot_board,
            )
            # simulate_enemy returns TOTAL shots, so we don't need to add current
            opp_rem_samples.append(s_taken)

        opp_avg = sum(opp_rem_samples) / len(opp_rem_samples)

        # 3. Calculate Win % (Simple comparison of distributions)
        wins = 0
        total_comps = 0
        for m in my_rem_samples:
            # adjusting my total vs opp total
            m_tot = (sum(row.count(HIT) + row.count(MISS) for row in self.board)) + m
            for o in opp_rem_samples:
                if m_tot < o:
                    wins += 1  # I finish faster
                total_comps += 1

        prob = (wins / total_comps) * 100 if total_comps else 0
        self.win_prob_label.setText(f"Win Probability: {prob:.1f}% (Me: ~{my_total:.1f} vs Opp: ~{opp_avg:.1f})")

    def recompute(self, defense_tab=None):
        # Rebuild world samples and ship-sunk probabilities
        self.world_masks, self.cell_hit_counts, self.ship_sunk_probs, self.num_world_samples = sample_worlds(
            self.board,
            self.placements,
            self.ship_ids,
            self.confirmed_sunk,
            self.assigned_hits,
            board_size=self.board_size,
        )

        N = self.num_world_samples
        if N > 0:
            self.cell_probs = [cnt / N for cnt in self.cell_hit_counts]
        else:
            self.cell_probs = [0.0] * (self.board_size * self.board_size)

        # Detect whether the underlying world model used exact enumeration
        # or Monte Carlo sampling, mirroring the logic in sample_worlds.
        if N > 0:
            # How many ships remain (not confirmed sunk)?
            remaining_ships = [s for s in self.ship_ids if s not in self.confirmed_sunk]
            self.remaining_ship_count = len(remaining_ships)

            # Recompute allowed placements to estimate the search-space size.
            hit_mask, miss_mask = board_masks(self.board, self.board_size)
            allowed = filter_allowed_placements(
                self.placements,
                hit_mask,
                miss_mask,
                self.confirmed_sunk,
                self.assigned_hits,
                self.board_size,
            )

            force_enumeration = (self.remaining_ship_count == 1)

            product = 1
            enumeration = True
            if not force_enumeration:
                for ship in self.ship_ids:
                    n = len(allowed[ship])
                    product *= n
                    if product > ENUMERATION_PRODUCT_LIMIT:
                        enumeration = False
                        break
            else:
                # Endgame: only one ship left -> always enumerate in sample_worlds.
                enumeration = True

            self.enumeration_mode = enumeration
        else:
            # No consistent layouts found - mode doesn't really matter.
            self.enumeration_mode = False
            self.remaining_ship_count = len([s for s in self.ship_ids if s not in self.confirmed_sunk])

        self.game_over = self._are_all_ships_sunk()

        if not self.game_over and N > 0:
            info_vals, two_ply_cells, two_ply_p = self._choose_best_cells_with_2ply()
            self.info_gain_values = info_vals
            self.two_ply_best_cells = two_ply_cells
            self.two_ply_best_prob = two_ply_p
        else:
            self.best_cells = []
            self.best_prob = 0.0
            self.two_ply_best_cells = []
            self.two_ply_best_prob = 0.0
            self.info_gain_values = [0.0] * (self.board_size * self.board_size)

        self.active_phase = classify_phase(
            self.board,
            self.confirmed_sunk,
            self.assigned_hits,
            self.ship_ids,
            board_size=self.board_size,
        )
        model_key, reason, best_overall = self._resolve_active_model(self.active_phase)
        self.active_model_key = model_key
        self.active_model_reason = reason

        unknown_cells = [
            (r, c)
            for r in range(self.board_size)
            for c in range(self.board_size)
            if self.board[r][c] == EMPTY
        ]

        candidates: List[Dict[str, object]] = []
        score_label = "Score"
        higher_better = True
        note = ""
        if not self.game_over and N > 0 and unknown_cells:
            candidates, score_label, higher_better, note = self._rank_cells_for_model(
                model_key, unknown_cells
            )

        if model_key == "two_ply":
            self.best_cells = list(self.two_ply_best_cells)
            self.best_prob = self.two_ply_best_prob
        else:
            if candidates:
                candidates_sorted = sorted(
                    candidates,
                    key=lambda e: (e["score"], e.get("p_hit", 0.0)),
                    reverse=higher_better,
                )
                self.best_cells = [candidates_sorted[0]["cell"]]
                self.best_prob = float(candidates_sorted[0].get("p_hit", 0.0))
            else:
                self.best_cells = []
                self.best_prob = 0.0

        warnings = self._compute_warnings()
        self.warning_label.setText(warnings)

        active_name = self._get_model_name(model_key)
        best_overall_name = self._get_model_name(best_overall) if best_overall else "N/A"
        self.active_model_label.setText(
            f"Active model: {active_name} | Phase: {self.active_phase} | {reason}. "
            f"Best overall: {best_overall_name}"
        )

        self._update_explanation_panel(candidates, score_label, higher_better, note)
        self._update_whatif_preview()

        self.update_board_view()
        self.update_status_view()

        if self.linked_defense_tab:
            self.update_win_prediction(self.linked_defense_tab)

    def _choose_best_cells_with_2ply(self) -> Tuple[List[float], List[Tuple[int, int]], float]:
        if not self.world_masks:
            return [0.0] * (self.board_size * self.board_size), [], 0.0

        info_vals, best_indices, best_p = two_ply_selection(
            self.board,
            self.world_masks,
            self.cell_hit_counts,
            self.board_size,
        )
        best_cells = [(idx // self.board_size, idx % self.board_size) for idx in best_indices]
        return info_vals, best_cells, best_p

    def _get_interpolated_color(self, val: float) -> str:
        """
        val: 0.0 to 1.0
        Returns a hex string or rgb string for background-color.
        Gradient: Dark Slate (#020617) -> Bright Sky Blue (#0ea5e9)
        """
        # Clamp value just in case
        val = max(0.0, min(1.0, val))

        # Start Color (Base background): #020617 -> (2, 6, 23)
        start_r, start_g, start_b = 2, 6, 23

        # End Color (Max heat): #0ea5e9 -> (14, 165, 233)
        end_r, end_g, end_b = 14, 165, 233

        # Linear Interpolation
        r = int(start_r + (end_r - start_r) * val)
        g = int(start_g + (end_g - start_g) * val)
        b = int(start_b + (end_b - start_b) * val)

        return f"rgb({r},{g},{b})"

    # ------------------------------------------------------------------------
    # UPDATED: update_board_view with Heatmap Logic
    # ------------------------------------------------------------------------
    def update_board_view(self):
        overlay_mode = self.overlay_combo.currentIndex()
        show_hit_prob = (overlay_mode == 1 and not self.game_over)
        show_info_gain = (overlay_mode == 2 and not self.game_over)

        best_set = {(r, c) for (r, c) in self.best_cells}

        for r in range(self.board_size):
            for c in range(self.board_size):
                btn = self.cell_buttons[r][c]
                state = self.board[r][c]
                idx = cell_index(r, c, self.board_size)

                # Default empty style
                base_color = Theme.BG_DARK
                text_color = Theme.TEXT_MAIN
                border_style = f"1px solid {Theme.BORDER_EMPTY}"
                text = ""

                if state == EMPTY:
                    # --- HEATMAP LOGIC START ---
                    heat_val = 0.0
                    if show_hit_prob:
                        heat_val = self.cell_probs[idx]
                        if heat_val > 0:
                            text = f"{int(round(heat_val * 100))}"
                    elif show_info_gain:
                        heat_val = self.info_gain_values[idx]
                        if heat_val > 0:
                            text = f"{int(round(heat_val * 100))}"

                    # Apply gradient if we have a value to show
                    if heat_val > 0.0:
                        base_color = self._get_interpolated_color(heat_val)
                        # If the background gets too bright, switch text to black for contrast
                        if heat_val > 0.6:
                            text_color = Theme.TEXT_DARK
                    # --- HEATMAP LOGIC END ---

                elif state == MISS:
                    base_color = Theme.MISS_BG
                    text_color = Theme.MISS_TEXT
                    border_style = f"1px solid {Theme.MISS_BORDER}"
                    text = "M"
                elif state == HIT:
                    base_color = Theme.HIT_BG
                    text_color = Theme.HIT_TEXT
                    border_style = f"1px solid {Theme.HIT_BORDER}"
                    text = "H"
                    assigned = any((r, c) in self.assigned_hits[s] for s in self.ship_ids)
                    if assigned:
                        border_style = f"2px dashed {Theme.ASSIGNED_BORDER}"

                # Construct the final stylesheet
                style_str = (
                    f"background-color: {base_color};"
                    f"color: {text_color};"
                    f"border: {border_style};"
                )

                # Highlight the "Best Guess" recommended by the solver
                if not self.game_over and (r, c) in best_set and state == EMPTY:
                    # Add a bright cyan border to the best moves
                    style_str += f"border: 2px solid {Theme.BORDER_BEST};"

                btn.setStyleSheet(style_str)
                btn.setText(text)

    def update_status_view(self):
        # Best-guess text
        if self.game_over:
            self.best_label.setText(
                "You marked all ships as sunk. Uncheck a ship to resume suggestions."
            )
        else:
            if self.best_cells:
                cells_str = ", ".join(f"({r + 1},{chr(ord('A') + c)})" for (r, c) in self.best_cells)
                model_name = self._get_model_name(self.active_model_key)
                self.best_label.setText(
                    f"Best guess ({model_name}): {cells_str} (p_hit≈{self.best_prob:.3f}, layouts≈{self.num_world_samples})"
                )
            else:
                self.best_label.setText("Best guess: (none)")

        # Summary of current evidence
        known_hits = sum(
            1 for r in range(self.board_size) for c in range(self.board_size) if self.board[r][c] == HIT
        )
        known_misses = sum(
            1 for r in range(self.board_size) for c in range(self.board_size) if self.board[r][c] == MISS
        )
        unknown = self.board_size * self.board_size - known_hits - known_misses
        expected_ship_cells = sum(self.cell_probs)

        self.summary_label.setText(
            f"Known hits: {known_hits}, misses: {known_misses}, unknown: {unknown}\n"
            f"World samples/layouts: {self.num_world_samples}, expected ship cells (total): {expected_ship_cells:.2f}"
        )

        # New: world model mode (exact enumeration vs Monte Carlo)
        if self.num_world_samples == 0:
            self.world_mode_label.setText(
                "World model: no consistent layouts (check hits/misses and sunk flags)."
            )
        else:
            if self.enumeration_mode:
                if self.remaining_ship_count == 1:
                    mode_str = "exact enumeration (endgame: one ship left)"
                else:
                    mode_str = "exact enumeration (small search space)"
            else:
                mode_str = "Monte Carlo sampling"

            self.world_mode_label.setText(
                f"World model: {mode_str}, layouts ≈ {self.num_world_samples}"
            )

        # Per-ship sunk probabilities
        for ship in self.ship_ids:
            lbl = self.ship_status_labels[ship]
            prob = self.ship_sunk_probs.get(ship, 0.0)

            if ship in self.confirmed_sunk:
                status = "SUNK"
                color = Theme.STATUS_SUNK
                prob_display = 1.0
            elif prob >= 0.99:
                status = "SUNK?"
                color = Theme.STATUS_SUNK_MAYBE
                prob_display = prob
            elif prob <= 0.01:
                status = "AFLOAT"
                color = Theme.STATUS_AFLOAT
                prob_display = prob
            else:
                status = "MAYBE"
                color = Theme.STATUS_MAYBE
                prob_display = prob

            lbl.setText(
                f"{self.ship_friendly_names[ship]}: {status} (p≈{prob_display:.2f})"
            )
            lbl.setStyleSheet(f"color: {color};")
