import math
import random
import time
import uuid
from typing import Dict, List, Optional, Sequence, Set, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

from battleship.domain.board import board_masks, cell_index, create_board, make_mask
from battleship.domain.config import (
    EMPTY,
    ENUMERATION_PRODUCT_LIMIT,
    HAS_SHIP,
    HIT,
    MISS,
    PARAM_SPECS,
    SHOT_HIT,
    SHOT_MISS,
    WORLD_SAMPLE_TARGET,
)
from battleship.domain.phase import (
    PHASE_ENDGAME,
    PHASE_HUNT,
    PHASE_TARGET,
    classify_phase,
)
from battleship.domain.worlds import (
    build_placement_index,
    compute_min_expected_worlds_after_one_shot,
    filter_allowed_placements,
    filter_worlds_by_constraints,
    summarize_worlds,
    sample_worlds,
)
from battleship.layouts.cache import LayoutRuntime
from battleship.persistence.layout_state import load_layout_state, save_layout_state
from battleship.persistence.model_selection import load_best_models
from battleship.persistence.stats import StatsTracker
from battleship.sim.defense_sim import build_base_heat, simulate_enemy_game_phase
from battleship.strategies.selection import Posterior, _choose_next_shot_for_strategy, two_ply_selection
from battleship.strategies.registry import model_defs
from battleship.ui.theme import Theme


class AttackTab(QtWidgets.QWidget):
    shot_recorded = QtCore.pyqtSignal()
    state_updated = QtCore.pyqtSignal()
    game_result = QtCore.pyqtSignal(bool)
    opponent_changed = QtCore.pyqtSignal(str)
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

        # Opponent learning (layout priors)
        self.opponents: Dict[str, Dict[str, object]] = {}
        self.active_opponent_id: Optional[str] = None
        self._loading_opponent = False
        self.opponent_cell_counts: List[float] = [0.0] * (self.board_size * self.board_size)
        self.opponent_layouts_recorded: int = 0
        self._using_opponent_prior: bool = False
        self._opponent_prior_confidence: float = 0.0
        self._opponent_prior_total: float = 0.0
        self._world_cache_key = None
        self.world_ship_masks: List[Tuple[int, ...]] = []
        self._placement_index = build_placement_index(self.placements, self.board_size)

        # Lookahead rollouts
        self.rollout_enabled: bool = False
        self.rollout_count: int = 16
        self.rollout_top_k: int = 6

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

        self.opponent_prior_label = QtWidgets.QLabel("")
        self.opponent_prior_label.setWordWrap(True)
        insights_layout.addWidget(self.opponent_prior_label)

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

        opponent_group = QtWidgets.QGroupBox("Opponent learning")
        opponent_layout = QtWidgets.QVBoxLayout(opponent_group)
        opp_row = QtWidgets.QHBoxLayout()
        self.opponent_label = QtWidgets.QLabel("Opponent:")
        opp_row.addWidget(self.opponent_label)
        self.opponent_combo = QtWidgets.QComboBox()
        self.opponent_combo.currentIndexChanged.connect(self._on_opponent_changed)
        opp_row.addWidget(self.opponent_combo, stretch=1)
        self.opponent_add_btn = QtWidgets.QPushButton("Add")
        self.opponent_add_btn.clicked.connect(self._add_opponent)
        opp_row.addWidget(self.opponent_add_btn)
        self.opponent_rename_btn = QtWidgets.QPushButton("Rename")
        self.opponent_rename_btn.clicked.connect(self._rename_opponent)
        opp_row.addWidget(self.opponent_rename_btn)
        self.opponent_delete_btn = QtWidgets.QPushButton("Delete")
        self.opponent_delete_btn.clicked.connect(self._delete_opponent)
        opp_row.addWidget(self.opponent_delete_btn)
        opponent_layout.addLayout(opp_row)

        self.opponent_stats_label = QtWidgets.QLabel("")
        self.opponent_stats_label.setWordWrap(True)
        opponent_layout.addWidget(self.opponent_stats_label)

        self.record_layout_btn = QtWidgets.QPushButton("Record opponent layout")
        self.record_layout_btn.clicked.connect(self._record_opponent_layout)
        opponent_layout.addWidget(self.record_layout_btn)

        models_layout.addWidget(opponent_group)

        rollout_group = QtWidgets.QGroupBox("Lookahead rollouts")
        rollout_layout = QtWidgets.QVBoxLayout(rollout_group)
        self.rollout_enable_cb = QtWidgets.QCheckBox("Enable lookahead (slower)")
        self.rollout_enable_cb.toggled.connect(self._rollout_settings_changed)
        rollout_layout.addWidget(self.rollout_enable_cb)
        roll_row = QtWidgets.QHBoxLayout()
        roll_row.addWidget(QtWidgets.QLabel("Rollouts:"))
        self.rollout_count_spin = QtWidgets.QSpinBox()
        self.rollout_count_spin.setRange(2, 200)
        self.rollout_count_spin.setValue(self.rollout_count)
        self.rollout_count_spin.valueChanged.connect(self._rollout_settings_changed)
        roll_row.addWidget(self.rollout_count_spin)
        roll_row.addWidget(QtWidgets.QLabel("Top cells:"))
        self.rollout_top_spin = QtWidgets.QSpinBox()
        self.rollout_top_spin.setRange(2, 20)
        self.rollout_top_spin.setValue(self.rollout_top_k)
        self.rollout_top_spin.valueChanged.connect(self._rollout_settings_changed)
        roll_row.addWidget(self.rollout_top_spin)
        roll_row.addStretch(1)
        rollout_layout.addLayout(roll_row)

        self.rollout_summary_label = QtWidgets.QLabel("")
        self.rollout_summary_label.setWordWrap(True)
        rollout_layout.addWidget(self.rollout_summary_label)

        models_layout.addWidget(rollout_group)
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

    def _new_opponent_profile(self, name: str) -> Dict[str, object]:
        return {
            "name": name,
            "cell_counts": [0.0] * (self.board_size * self.board_size),
            "layouts_recorded": 0,
        }

    def _export_opponent_profile(self) -> Dict[str, object]:
        return {
            "name": self._current_opponent_name(),
            "cell_counts": list(self.opponent_cell_counts),
            "layouts_recorded": int(self.opponent_layouts_recorded),
        }

    def _import_opponent_profile(self, profile: Dict[str, object]) -> None:
        self._loading_opponent = True
        try:
            counts = profile.get("cell_counts")
            if (
                isinstance(counts, list)
                and len(counts) == self.board_size * self.board_size
                and all(isinstance(v, (int, float)) for v in counts)
            ):
                self.opponent_cell_counts = [float(v) for v in counts]
            else:
                self.opponent_cell_counts = [0.0] * (self.board_size * self.board_size)

            layouts = profile.get("layouts_recorded")
            if isinstance(layouts, int) and layouts >= 0:
                self.opponent_layouts_recorded = layouts
            else:
                self.opponent_layouts_recorded = 0
        finally:
            self._loading_opponent = False
        self._update_opponent_stats_label()

    def _current_opponent_name(self) -> str:
        if self.active_opponent_id and self.active_opponent_id in self.opponents:
            name = self.opponents[self.active_opponent_id].get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
        return "Opponent"

    def _refresh_opponent_combo(self, select_id: Optional[str] = None) -> None:
        if not hasattr(self, "opponent_combo"):
            return
        self.opponent_combo.blockSignals(True)
        self.opponent_combo.clear()
        for oid, profile in self.opponents.items():
            name = profile.get("name") if isinstance(profile, dict) else None
            label = str(name).strip() if isinstance(name, str) and name.strip() else oid
            self.opponent_combo.addItem(label, userData=oid)
        if select_id:
            for i in range(self.opponent_combo.count()):
                if self.opponent_combo.itemData(i) == select_id:
                    self.opponent_combo.setCurrentIndex(i)
                    break
        self.opponent_combo.blockSignals(False)

    def set_opponent_controls_visible(self, visible: bool) -> None:
        for widget in (
            getattr(self, "opponent_label", None),
            getattr(self, "opponent_combo", None),
            getattr(self, "opponent_add_btn", None),
            getattr(self, "opponent_rename_btn", None),
            getattr(self, "opponent_delete_btn", None),
        ):
            if widget is not None:
                widget.setVisible(bool(visible))

    def _ensure_opponents(self) -> None:
        if not self.opponents:
            oid = f"opp_{uuid.uuid4().hex[:8]}"
            self.opponents[oid] = self._new_opponent_profile("Opponent 1")
            self.active_opponent_id = oid
        if self.active_opponent_id not in self.opponents:
            self.active_opponent_id = next(iter(self.opponents))
        self._refresh_opponent_combo(self.active_opponent_id)

    def _find_opponent_id_by_name(self, name: str) -> Optional[str]:
        target = str(name or "").strip().lower()
        if not target:
            return None
        for oid, profile in self.opponents.items():
            if not isinstance(profile, dict):
                continue
            pname = str(profile.get("name") or "").strip().lower()
            if pname == target:
                return oid
        return None

    def _save_current_profile(self) -> None:
        if not self.active_opponent_id:
            return
        self.opponents[self.active_opponent_id] = self._export_opponent_profile()

    def _on_opponent_changed(self, index: int) -> None:
        if self._loading_opponent:
            return
        if index < 0:
            return
        new_id = self.opponent_combo.itemData(index)
        if not isinstance(new_id, str) or not new_id:
            return
        if new_id == self.active_opponent_id:
            return
        self._save_current_profile()
        self.active_opponent_id = new_id
        profile = self.opponents.get(new_id)
        if isinstance(profile, dict):
            self._import_opponent_profile(profile)
        self.save_state()
        self.recompute()
        try:
            self.opponent_changed.emit(self._current_opponent_name())
        except Exception:
            pass

    def _add_opponent(self) -> None:
        name, ok = QtWidgets.QInputDialog.getText(self, "Add Opponent", "Opponent name:")
        if not ok:
            return
        name = str(name).strip() or "Opponent"
        oid = self._ensure_opponent_by_name(name)
        self._refresh_opponent_combo(oid)
        self._import_opponent_profile(self.opponents[oid])
        self.save_state()
        self.recompute()

    def _rename_opponent(self) -> None:
        if not self.active_opponent_id:
            return
        current_name = self._current_opponent_name()
        name, ok = QtWidgets.QInputDialog.getText(self, "Rename Opponent", "Opponent name:", text=current_name)
        if not ok:
            return
        name = str(name).strip() or current_name
        profile = self.opponents.get(self.active_opponent_id, {})
        if isinstance(profile, dict):
            profile["name"] = name
            self.opponents[self.active_opponent_id] = profile
        self._refresh_opponent_combo(self.active_opponent_id)
        self.save_state()

    def _delete_opponent(self) -> None:
        if not self.active_opponent_id:
            return
        if len(self.opponents) <= 1:
            QtWidgets.QMessageBox.information(self, "Cannot delete", "At least one opponent profile is required.")
            return
        name = self._current_opponent_name()
        reply = QtWidgets.QMessageBox.question(
            self,
            "Delete Opponent",
            f"Delete opponent profile '{name}'?\nThis cannot be undone.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        del self.opponents[self.active_opponent_id]
        self.active_opponent_id = next(iter(self.opponents))
        self._refresh_opponent_combo(self.active_opponent_id)
        self._import_opponent_profile(self.opponents[self.active_opponent_id])
        self.save_state()
        self.recompute()

    def _ensure_opponent_by_name(self, name: str) -> str:
        clean = str(name or "").strip() or "Opponent"
        for oid, profile in self.opponents.items():
            if not isinstance(profile, dict):
                continue
            pname = str(profile.get("name") or "").strip()
            if pname.lower() == clean.lower():
                return oid
        oid = f"opp_{uuid.uuid4().hex[:8]}"
        self.opponents[oid] = self._new_opponent_profile(clean)
        self._save_current_profile()
        self.active_opponent_id = oid
        return oid

    def set_active_opponent_by_name(self, name: str) -> None:
        oid = self._ensure_opponent_by_name(name)
        self.active_opponent_id = oid
        self._refresh_opponent_combo(oid)
        self._import_opponent_profile(self.opponents[oid])
        self.save_state()
        self.recompute()

    def get_opponent_names(self) -> List[str]:
        names: List[str] = []
        for profile in self.opponents.values():
            if not isinstance(profile, dict):
                continue
            pname = str(profile.get("name") or "").strip()
            if pname:
                names.append(pname)
        return names

    def opponent_count(self) -> int:
        return len(self.opponents)

    def rename_active_opponent(self, name: str) -> str:
        if not self.active_opponent_id:
            return self._current_opponent_name()
        clean = str(name).strip() or self._current_opponent_name()
        profile = self.opponents.get(self.active_opponent_id, {})
        if isinstance(profile, dict):
            profile["name"] = clean
            self.opponents[self.active_opponent_id] = profile
        self._refresh_opponent_combo(self.active_opponent_id)
        self.save_state()
        return clean

    def delete_opponent_by_name(self, name: str) -> bool:
        if len(self.opponents) <= 1:
            return False
        oid = self._find_opponent_id_by_name(name)
        if not oid:
            return False
        del self.opponents[oid]
        if not self.opponents:
            return False
        if self.active_opponent_id == oid:
            self.active_opponent_id = next(iter(self.opponents))
        self._refresh_opponent_combo(self.active_opponent_id)
        profile = self.opponents.get(self.active_opponent_id)
        if isinstance(profile, dict):
            self._import_opponent_profile(profile)
        self.save_state()
        self.recompute()
        return True

    def _update_opponent_stats_label(self) -> None:
        if not hasattr(self, "opponent_stats_label"):
            return
        total = sum(self.opponent_cell_counts)
        total_i = int(round(total))
        self.opponent_stats_label.setText(
            f"Layouts recorded: {self.opponent_layouts_recorded}. "
            f"Ship cells observed: {total_i}."
        )

    def _record_opponent_layout(self) -> None:
        dialog = OpponentLayoutDialog(self.board_size, self._expected_ship_cells(), parent=self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        cells = dialog.selected_cells()
        if not cells:
            QtWidgets.QMessageBox.information(self, "No cells selected", "No ship cells were selected.")
            return
        expected = self._expected_ship_cells()
        if expected > 0 and len(cells) != expected:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Confirm layout",
                f"Selected {len(cells)} cells, expected {expected}.\nSave anyway?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return
        for r, c in cells:
            idx = cell_index(r, c, self.board_size)
            if 0 <= idx < len(self.opponent_cell_counts):
                self.opponent_cell_counts[idx] += 1.0
        self.opponent_layouts_recorded += 1
        self._update_opponent_stats_label()
        self._save_current_profile()
        self.save_state()
        self.recompute()

    def _compute_opponent_prior(self) -> Optional[List[float]]:
        counts = self.opponent_cell_counts
        if not counts:
            self._using_opponent_prior = False
            self._opponent_prior_confidence = 0.0
            self._opponent_prior_total = 0.0
            return None

        total = float(sum(counts))
        self._opponent_prior_total = total
        if total <= 0.0:
            self._using_opponent_prior = False
            self._opponent_prior_confidence = 0.0
            return None

        total_cells = self.board_size * self.board_size
        smoothing = 1.0
        uniform = 1.0 / total_cells
        denom = total + smoothing * total_cells
        normalized = [(float(c) + smoothing) / denom for c in counts]

        # Confidence rises with more observed ship cells.
        blend_k = max(1.0, float(self._expected_ship_cells()) * 6.0)
        confidence = total / (total + blend_k)
        confidence = max(0.0, min(1.0, confidence))
        self._opponent_prior_confidence = confidence
        self._using_opponent_prior = True

        if confidence <= 0.0:
            return None
        if confidence >= 1.0:
            return normalized
        return [confidence * normalized[i] + (1.0 - confidence) * uniform for i in range(total_cells)]

    def _rollout_settings_changed(self) -> None:
        self.rollout_enabled = bool(self.rollout_enable_cb.isChecked())
        self.rollout_count = int(self.rollout_count_spin.value())
        self.rollout_top_k = int(self.rollout_top_spin.value())
        self.save_state()
        self.recompute()

    def _rollout_budget(self, unknown_cells_count: int) -> Dict[str, object]:
        total_cells = self.board_size * self.board_size
        if total_cells <= 0:
            return {"enabled": False, "note": "Lookahead skipped (invalid board size)."}

        unknown_ratio = unknown_cells_count / total_cells
        if unknown_ratio >= 0.85:
            return {
                "enabled": False,
                "note": f"Lookahead throttled (early game: {int(round(unknown_ratio * 100))}% unknown).",
            }

        if unknown_ratio >= 0.70:
            scale = 0.25
            time_limit = 0.35
        elif unknown_ratio >= 0.55:
            scale = 0.5
            time_limit = 0.6
        elif unknown_ratio >= 0.40:
            scale = 0.75
            time_limit = 0.9
        else:
            scale = 1.0
            time_limit = 1.2

        rollouts = max(2, int(round(self.rollout_count * scale)))
        top_k = max(2, int(round(self.rollout_top_k * scale)))

        max_shots = int(total_cells * (0.35 + 0.65 * (1.0 - unknown_ratio)))
        max_shots = max(12, min(total_cells, max_shots))

        note = ""
        if scale < 1.0:
            note = "Lookahead throttled for responsiveness."
        return {
            "enabled": True,
            "rollouts": rollouts,
            "top_k": top_k,
            "max_shots": max_shots,
            "time_limit": time_limit,
            "note": note,
        }

    def _simulate_rollout_after_shot(
        self,
        base_board: List[List[str]],
        world_mask: int,
        first_shot: Tuple[int, int],
        model_key: str,
        rng: random.Random,
        max_shots: Optional[int] = None,
    ) -> int:
        sim_board = [row[:] for row in base_board]
        total_targets = int(bin(world_mask).count("1"))
        hits = 0
        for r in range(self.board_size):
            for c in range(self.board_size):
                if sim_board[r][c] == HIT:
                    idx = cell_index(r, c, self.board_size)
                    if (world_mask >> idx) & 1:
                        hits += 1
        remaining = max(0, total_targets - hits)

        shots = 0
        fr, fc = first_shot
        if sim_board[fr][fc] == EMPTY:
            idx = cell_index(fr, fc, self.board_size)
            is_hit = (world_mask >> idx) & 1
            sim_board[fr][fc] = HIT if is_hit else MISS
            shots += 1
            if is_hit:
                remaining -= 1

        if max_shots is None:
            max_shots = self.board_size * self.board_size
        while remaining > 0 and shots < max_shots:
            r, c = _choose_next_shot_for_strategy(
                model_key,
                sim_board,
                self.placements,
                rng,
                self.ship_ids,
                board_size=self.board_size,
            )
            if sim_board[r][c] != EMPTY:
                # Safety: pick a random unknown if strategy returns a known cell.
                unknown = [
                    (rr, cc)
                    for rr in range(self.board_size)
                    for cc in range(self.board_size)
                    if sim_board[rr][cc] == EMPTY
                ]
                if not unknown:
                    break
                r, c = rng.choice(unknown)
            idx = cell_index(r, c, self.board_size)
            is_hit = (world_mask >> idx) & 1
            sim_board[r][c] = HIT if is_hit else MISS
            shots += 1
            if is_hit:
                remaining -= 1
        if remaining > 0 and shots >= max_shots:
            shots += remaining
        return shots

    def _apply_rollout_lookahead(
        self,
        candidates: List[Dict[str, object]],
        higher_better: bool,
        model_key: str,
        rollouts: int,
        top_k: int,
        max_shots: int,
        time_limit: float,
    ) -> Optional[Tuple[Tuple[int, int], float, bool]]:
        if not self.rollout_enabled:
            return None
        if not candidates or not self.world_masks:
            return None

        candidates_sorted = sorted(
            candidates,
            key=lambda e: (e["score"], e.get("p_hit", 0.0)),
            reverse=higher_better,
        )
        eval_cells = [entry["cell"] for entry in candidates_sorted[:top_k]]
        if not eval_cells:
            return None

        rng = random.Random()
        if len(self.world_masks) <= rollouts:
            world_samples = list(self.world_masks)
        else:
            world_samples = rng.sample(self.world_masks, rollouts)

        if not world_samples:
            return None

        best_cell = None
        best_avg = None

        start = time.time()
        timed_out = False
        for cell in eval_cells:
            total = 0.0
            for wmask in world_samples:
                if time.time() - start > time_limit:
                    timed_out = True
                    break
                total += self._simulate_rollout_after_shot(
                    self.board, wmask, cell, model_key, rng, max_shots=max_shots
                )
            if time.time() - start > time_limit:
                timed_out = True
                break
            avg = total / len(world_samples)
            if best_avg is None or avg < best_avg:
                best_avg = avg
                best_cell = cell
        if best_cell is None:
            return None

        return best_cell, float(best_avg), timed_out

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
        if current == EMPTY and new_state in (HIT, MISS):
            try:
                self.shot_recorded.emit()
            except Exception:
                pass
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

        if model_key in {
            "assigned_target_marginal",
            "minlen_parity_entropy",
            "ewa1_pruned",
            "placement_factorized",
            "endgame_exact_combo",
            "meta_ucb_hybrid",
        }:
            score_label = "p(hit)"
            hit_mask = 0
            miss_mask = 0
            for r in range(self.board_size):
                for c in range(self.board_size):
                    idx = cell_index(r, c, self.board_size)
                    if self.board[r][c] == HIT:
                        hit_mask |= 1 << idx
                    elif self.board[r][c] == MISS:
                        miss_mask |= 1 << idx
            posterior = Posterior(self.world_masks, self.cell_hit_counts, self.num_world_samples)
            rng = random.Random(0)
            r, c = _choose_next_shot_for_strategy(
                model_key,
                self.board,
                self.placements,
                rng,
                self.ship_ids,
                board_size=self.board_size,
                known_sunk=self.confirmed_sunk,
                known_assigned=self.assigned_hits,
                posterior=posterior,
                unknown_cells=unknown_cells,
                has_any_hit=bool(hit_mask),
                hit_mask=hit_mask,
                miss_mask=miss_mask,
            )
            if self.board[r][c] != EMPTY:
                r, c = random.choice(unknown_cells)
            candidates.append({"cell": (r, c), "p_hit": p_hit(r, c), "score": p_hit(r, c)})
            note = "Selection engine"
            return candidates, score_label, higher_better, note

        if model_key == "entropy1":
            score_label = "Info (1-ply)"

            def score_entropy(r: int, c: int) -> float:
                idx = cell_index(r, c, self.board_size)
                n_hit = self.cell_hit_counts[idx]
                if n_hit <= 0 or n_hit >= N:
                    return 0.0
                p_h = n_hit / N
                p_m = 1.0 - p_h
                return -(p_h * math.log2(p_h) + p_m * math.log2(p_m))

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

        if model_key == "ucb_explore":
            score_label = "UCB"
            c_bonus = 0.35
            specs = PARAM_SPECS.get("ucb_explore", [])
            if specs:
                c_bonus = float(specs[0].get("default", c_bonus))
            for r, c in unknown_cells:
                p = p_hit(r, c)
                bonus = c_bonus * math.sqrt(max(0.0, p * (1.0 - p)))
                candidates.append({"cell": (r, c), "p_hit": p, "score": p + bonus})
            note = f"Exploration bonus c={c_bonus:.2f}"
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

        if model_key == "rollout_mcts":
            score_label = "Base"
            if is_target_mode:
                for r, c in unknown_cells:
                    candidates.append({"cell": (r, c), "p_hit": p_hit(r, c), "score": p_hit(r, c)})
                note = "Rollout lookahead (target mode baseline)"
            else:
                for r, c in unknown_cells:
                    idx = cell_index(r, c, self.board_size)
                    n_hit = self.cell_hit_counts[idx]
                    n_miss = N - n_hit
                    score = N - (n_hit * n_hit + n_miss * n_miss) / N
                    candidates.append({"cell": (r, c), "p_hit": p_hit(r, c), "score": score})
                note = "Rollout lookahead (hunt mode baseline)"
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
        self._save_current_profile()

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
            "opponents": self.opponents,
            "active_opponent_id": self.active_opponent_id,
            "rollout_enabled": bool(self.rollout_enabled),
            "rollout_count": int(self.rollout_count),
            "rollout_top_k": int(self.rollout_top_k),
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
            self._ensure_opponents()
            profile = self.opponents.get(self.active_opponent_id, {})
            if isinstance(profile, dict):
                self._import_opponent_profile(profile)
            return

        opponents = data.get("opponents")
        if isinstance(opponents, dict) and opponents:
            self.opponents = opponents
            active = data.get("active_opponent_id")
            if isinstance(active, str) and active in self.opponents:
                self.active_opponent_id = active
            else:
                self.active_opponent_id = next(iter(self.opponents))
            self._ensure_opponents()
            profile = self.opponents.get(self.active_opponent_id, {})
            if isinstance(profile, dict):
                self._import_opponent_profile(profile)
        else:
            self._ensure_opponents()
            profile = self.opponents.get(self.active_opponent_id, {})
            if isinstance(profile, dict):
                self._import_opponent_profile(profile)

        rollout_enabled = data.get("rollout_enabled")
        if isinstance(rollout_enabled, bool):
            self.rollout_enabled = rollout_enabled
        rollout_count = data.get("rollout_count")
        if isinstance(rollout_count, int):
            self.rollout_count = max(2, rollout_count)
        rollout_top_k = data.get("rollout_top_k")
        if isinstance(rollout_top_k, int):
            self.rollout_top_k = max(2, rollout_top_k)

        if hasattr(self, "rollout_enable_cb"):
            self.rollout_enable_cb.blockSignals(True)
            self.rollout_enable_cb.setChecked(self.rollout_enabled)
            self.rollout_enable_cb.blockSignals(False)
        if hasattr(self, "rollout_count_spin"):
            self.rollout_count_spin.blockSignals(True)
            self.rollout_count_spin.setValue(self.rollout_count)
            self.rollout_count_spin.blockSignals(False)
        if hasattr(self, "rollout_top_spin"):
            self.rollout_top_spin.blockSignals(True)
            self.rollout_top_spin.setValue(self.rollout_top_k)
            self.rollout_top_spin.blockSignals(False)

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

        # If the last saved game was already marked as over,
        # treat that as a finished game and do NOT restore it.
        if data.get("game_over", False):
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

        self.undo_stack.clear()
        self.redo_stack.clear()
        self._update_history_buttons()

    def _record_game_result(self, win: bool):
        self.stats.record_game(win)
        if hasattr(self, "stats_label"):
            self.stats_label.setText(self.stats.summary_text())
        try:
            self.game_result.emit(bool(win))
        except Exception:
            pass
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

        # Persist cleared board without dropping opponent history.
        self.save_state()

    def _are_all_ships_sunk(self) -> bool:
        # Only trust the user's explicit checkboxes for "game over".
        return len(self.confirmed_sunk) == len(self.ship_ids)

    # In AttackTab class, add the prediction logic:
    def update_win_prediction(self, defense_tab):
        """Simulate Me vs Opponent."""
        if not hasattr(self, "win_prob_label"):
            return
        result = self.compute_win_prediction(defense_tab)
        if result is None:
            self.win_prob_label.setText("Win Prob: N/A (Need Defense Layout)")
            return
        prob, my_total, opp_avg = result
        self.win_prob_label.setText(
            f"Win Probability: {prob:.1f}% (Me: ~{my_total:.1f} vs Opp: ~{opp_avg:.1f})"
        )

    def compute_win_prediction(self, defense_tab) -> Optional[Tuple[float, float, float]]:
        """Return (win_pct, my_total, opp_avg) or None if unavailable."""
        if not self.world_masks or not getattr(defense_tab, "layout_board", None):
            return None

        # 1. Estimate MY remaining shots using a fast, static ordering based on current probabilities.
        my_rem_samples = []
        rng = random.Random()
        sample_worlds = rng.sample(self.world_masks, min(6, len(self.world_masks)))

        unknown_cells = []
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r][c] == EMPTY:
                    unknown_cells.append((r, c))
        if not unknown_cells:
            return None

        if not getattr(self, "cell_probs", None):
            return None
        if len(self.cell_probs) != self.board_size * self.board_size:
            return None

        known_hit_mask = 0
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r][c] == HIT:
                    known_hit_mask |= 1 << cell_index(r, c, self.board_size)

        def pick_best_unknown(sim_board, candidates=None):
            best = None
            best_p = -1.0
            if candidates is None:
                for rr in range(self.board_size):
                    for cc in range(self.board_size):
                        if sim_board[rr][cc] != EMPTY:
                            continue
                        idx = cell_index(rr, cc, self.board_size)
                        p = self.cell_probs[idx]
                        if p > best_p + 1e-12:
                            best_p = p
                            best = (rr, cc)
                        elif best is not None and abs(p - best_p) <= 1e-12 and rng.random() < 0.5:
                            best = (rr, cc)
            else:
                for rr, cc in candidates:
                    if sim_board[rr][cc] != EMPTY:
                        continue
                    idx = cell_index(rr, cc, self.board_size)
                    p = self.cell_probs[idx]
                    if p > best_p + 1e-12:
                        best_p = p
                        best = (rr, cc)
                    elif best is not None and abs(p - best_p) <= 1e-12 and rng.random() < 0.5:
                        best = (rr, cc)
            return best

        def simulate_fast(w_mask: int) -> int:
            sim_board = [row[:] for row in self.board]
            remaining = bin(w_mask & ~known_hit_mask).count("1")
            if remaining <= 0:
                return 0

            frontier = set()
            for rr in range(self.board_size):
                for cc in range(self.board_size):
                    if sim_board[rr][cc] != HIT:
                        continue
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr = rr + dr
                        nc = cc + dc
                        if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                            if sim_board[nr][nc] == EMPTY:
                                frontier.add((nr, nc))

            shots = 0
            max_shots = len(unknown_cells)
            while remaining > 0 and shots < max_shots:
                cell = pick_best_unknown(sim_board, frontier if frontier else None)
                if cell is None:
                    break
                r, c = cell
                idx = cell_index(r, c, self.board_size)
                is_hit = (w_mask >> idx) & 1
                sim_board[r][c] = HIT if is_hit else MISS
                shots += 1
                if (r, c) in frontier:
                    frontier.discard((r, c))
                if is_hit:
                    remaining -= 1
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr = r + dr
                        nc = c + dc
                        if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                            if sim_board[nr][nc] == EMPTY:
                                frontier.add((nr, nc))
            if remaining > 0 and shots >= max_shots:
                shots += remaining
            return shots

        for w_mask in sample_worlds:
            my_rem_samples.append(simulate_fast(w_mask))

        if not my_rem_samples:
            return None
        my_avg_rem = sum(my_rem_samples) / len(my_rem_samples)
        my_total = (sum(row.count(HIT) + row.count(MISS) for row in self.board)) + my_avg_rem

        # 2. Estimate OPPONENT remaining shots
        layout_mask = 0
        for r in range(self.board_size):
            for c in range(self.board_size):
                if defense_tab.layout_board[r][c] == HAS_SHIP:
                    layout_mask |= (1 << cell_index(r, c, self.board_size))

        if hasattr(defense_tab, "_build_base_heat_phase"):
            base_heat, _pattern = defense_tab._build_base_heat_phase()
        else:
            base_heat = [
                build_base_heat(defense_tab.hit_counts_phase[p], defense_tab.miss_counts_phase[p], self.board_size)
                for p in range(4)
            ]

        opp_rem_samples = []
        for _ in range(10):
            s_taken, _ = simulate_enemy_game_phase(
                layout_mask,
                base_heat,
                defense_tab.disp_counts,
                "seq",
                rng,
                board_size=self.board_size,
                initial_shot_board=defense_tab.shot_board,
            )
            opp_rem_samples.append(s_taken)

        if not opp_rem_samples:
            return None
        opp_avg = sum(opp_rem_samples) / len(opp_rem_samples)

        wins = 0
        total_comps = 0
        for m in my_rem_samples:
            m_tot = (sum(row.count(HIT) + row.count(MISS) for row in self.board)) + m
            for o in opp_rem_samples:
                if m_tot < o:
                    wins += 1
                total_comps += 1

        prob = (wins / total_comps) * 100 if total_comps else 0

        my_shots = sum(row.count(HIT) + row.count(MISS) for row in self.board)
        opp_shots = sum(
            row.count(SHOT_HIT) + row.count(SHOT_MISS) for row in defense_tab.shot_board
        )
        shot_diff = my_shots - opp_shots
        baseline = 50.0 - (4.0 * shot_diff)
        baseline = max(5.0, min(95.0, baseline))
        total_shots = my_shots + opp_shots
        denom = max(10.0, float(self.board_size) * 1.5)
        weight = min(1.0, total_shots / denom)
        adj_prob = baseline + weight * (prob - baseline)
        adj_prob = max(0.0, min(100.0, adj_prob))

        return float(adj_prob), float(my_total), float(opp_avg)

    def recompute(self, defense_tab=None):
        # Rebuild world samples and ship-sunk probabilities
        cell_prior = self._compute_opponent_prior()
        hit_mask, miss_mask = board_masks(self.board, self.board_size)
        assigned_masks = tuple(
            make_mask(list(self.assigned_hits.get(ship, set())), self.board_size)
            if self.assigned_hits.get(ship)
            else 0
            for ship in self.ship_ids
        )
        confirmed_key = tuple(sorted(self.confirmed_sunk))
        prior_key = None
        if cell_prior is not None:
            prior_key = tuple(int(v) for v in self.opponent_cell_counts)
        target_worlds = WORLD_SAMPLE_TARGET
        cache_key = (hit_mask, miss_mask, confirmed_key, assigned_masks, prior_key, target_worlds)
        reuse_prior = False
        can_reuse = False
        if self._world_cache_key is not None and len(self._world_cache_key) >= 6:
            reuse_prior = (self._world_cache_key[4] == prior_key)
            try:
                prev_hit, prev_miss, prev_confirmed, prev_assigned, _prev_prior, _prev_target = (
                    self._world_cache_key
                )
                prev_confirmed_set = set(prev_confirmed)
                new_confirmed_set = set(confirmed_key)
                hits_stricter = (hit_mask | prev_hit) == hit_mask
                misses_stricter = (miss_mask | prev_miss) == miss_mask
                confirmed_stricter = prev_confirmed_set.issubset(new_confirmed_set)
                assigned_stricter = True
                if isinstance(prev_assigned, tuple) and len(prev_assigned) == len(assigned_masks):
                    for prev_mask, new_mask in zip(prev_assigned, assigned_masks):
                        if prev_mask & ~new_mask:
                            assigned_stricter = False
                            break
                else:
                    assigned_stricter = False
                can_reuse = reuse_prior and hits_stricter and misses_stricter and confirmed_stricter and assigned_stricter
            except Exception:
                can_reuse = False

        filtered_union: Optional[List[int]] = None
        filtered_masks: Optional[List[Tuple[int, ...]]] = None
        if can_reuse and self.world_ship_masks:
            filtered_union, filtered_masks = filter_worlds_by_constraints(
                self.world_masks,
                self.world_ship_masks,
                self.ship_ids,
                hit_mask,
                miss_mask,
                self.confirmed_sunk,
                assigned_masks,
            )

        if cache_key != self._world_cache_key:
            if filtered_union is not None and filtered_masks is not None and len(filtered_union) >= target_worlds:
                self.world_masks = filtered_union[:target_worlds]
                self.world_ship_masks = filtered_masks[:target_worlds]
                self.cell_hit_counts, self.ship_sunk_probs = summarize_worlds(
                    self.world_masks,
                    self.world_ship_masks,
                    self.ship_ids,
                    hit_mask,
                    self.board_size,
                    confirmed_sunk=self.confirmed_sunk,
                )
                self.num_world_samples = len(self.world_masks)
            else:
                existing_union = filtered_union if filtered_union is not None else None
                existing_masks = filtered_masks if filtered_masks is not None else None
                (
                    self.world_masks,
                    self.world_ship_masks,
                    self.cell_hit_counts,
                    self.ship_sunk_probs,
                    self.num_world_samples,
                ) = sample_worlds(
                    self.board,
                    self.placements,
                    self.ship_ids,
                    self.confirmed_sunk,
                    self.assigned_hits,
                    board_size=self.board_size,
                    cell_prior=cell_prior,
                    target_worlds=target_worlds,
                    existing_union=existing_union,
                    existing_ship_masks=existing_masks,
                    return_ship_masks=True,
                    placement_index=self._placement_index,
                )
            self._world_cache_key = cache_key

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
            allowed = filter_allowed_placements(
                self.placements,
                hit_mask,
                miss_mask,
                self.confirmed_sunk,
                self.assigned_hits,
                self.board_size,
                placement_index=self._placement_index,
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

            if cell_prior is not None:
                enumeration = False
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

        rollout_note = ""
        rollout_active = self.rollout_enabled or model_key == "rollout_mcts"
        if rollout_active and not self.game_over and N > 0 and candidates:
            budget = self._rollout_budget(len(unknown_cells))
            if not budget.get("enabled", False):
                rollout_note = str(budget.get("note", "")).strip()
            else:
                rollout_result = self._apply_rollout_lookahead(
                    candidates,
                    higher_better,
                    model_key,
                    rollouts=int(budget.get("rollouts", self.rollout_count)),
                    top_k=int(budget.get("top_k", self.rollout_top_k)),
                    max_shots=int(budget.get("max_shots", self.board_size * self.board_size)),
                    time_limit=float(budget.get("time_limit", 1.0)),
                )
                if rollout_result is not None:
                    best_cell, avg_shots, timed_out = rollout_result
                    self.best_cells = [best_cell]
                    self.best_prob = self.cell_probs[cell_index(best_cell[0], best_cell[1], self.board_size)]
                    rollout_note = f"Lookahead: expected remaining shots ~{avg_shots:.1f}"
                    if timed_out:
                        rollout_note += " (throttled)"
                else:
                    rollout_note = str(budget.get("note", "")).strip()
            if model_key == "rollout_mcts" and not self.rollout_enabled:
                if budget.get("enabled", False):
                    if rollout_note:
                        rollout_note += " (model forces lookahead)"
                    else:
                        rollout_note = "Lookahead active (model forces lookahead)."
                else:
                    if rollout_note:
                        rollout_note += " (model throttled)"
                    else:
                        rollout_note = "Lookahead throttled (model selected)."

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
        if hasattr(self, "rollout_summary_label"):
            self.rollout_summary_label.setText(rollout_note)

        self.update_board_view()
        self.update_status_view()

        if self.linked_defense_tab:
            self.update_win_prediction(self.linked_defense_tab)
        try:
            self.state_updated.emit()
        except Exception:
            pass

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

        if hasattr(self, "opponent_prior_label"):
            if not self._using_opponent_prior or self._opponent_prior_total <= 0.0:
                self.opponent_prior_label.setText(
                    "Opponent prior: none (record layouts to bias sampling)."
                )
            else:
                conf_pct = int(round(self._opponent_prior_confidence * 100))
                self.opponent_prior_label.setText(
                    f"Opponent prior: {self._current_opponent_name()} | "
                    f"layouts recorded: {self.opponent_layouts_recorded}, confidence: {conf_pct}%"
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


class OpponentLayoutDialog(QtWidgets.QDialog):
    def __init__(self, board_size: int, expected_cells: int, parent=None):
        super().__init__(parent)
        self.board_size = int(board_size)
        self.expected_cells = int(expected_cells)
        self._selected = [[False for _ in range(self.board_size)] for _ in range(self.board_size)]
        self._buttons: List[List[QtWidgets.QToolButton]] = []

        self.setWindowTitle("Record opponent layout")
        self.setModal(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        hint = QtWidgets.QLabel(
            "Click the cells where the opponent placed ships.\n"
            "This is saved only on your device."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {Theme.TEXT_LABEL};")
        layout.addWidget(hint)

        self.count_label = QtWidgets.QLabel("")
        self.count_label.setWordWrap(True)
        layout.addWidget(self.count_label)

        board_container = QtWidgets.QWidget()
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

        cell = max(18, min(30, int(320 / max(1, self.board_size))))
        for r in range(self.board_size):
            row = []
            for c in range(self.board_size):
                btn = QtWidgets.QToolButton()
                btn.setCheckable(True)
                btn.setFixedSize(cell, cell)
                btn.clicked.connect(self._make_toggle_handler(r, c))
                self._apply_cell_style(btn, False)
                board_layout.addWidget(btn, r + 1, c + 1)
                row.append(btn)
            self._buttons.append(row)

        layout.addWidget(board_container)

        action_row = QtWidgets.QHBoxLayout()
        clear_btn = QtWidgets.QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_cells)
        action_row.addWidget(clear_btn)
        action_row.addStretch(1)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        action_row.addWidget(buttons)

        layout.addLayout(action_row)

        self._update_count_label()

    def _make_toggle_handler(self, r: int, c: int):
        def handler():
            btn = self._buttons[r][c]
            checked = btn.isChecked()
            self._selected[r][c] = checked
            self._apply_cell_style(btn, checked)
            self._update_count_label()
        return handler

    def _apply_cell_style(self, btn: QtWidgets.QToolButton, checked: bool) -> None:
        if checked:
            btn.setStyleSheet(
                f"background-color: {Theme.LAYOUT_SHIP_BG};"
                f"color: {Theme.LAYOUT_SHIP_TEXT};"
                f"border: 1px solid {Theme.LAYOUT_SHIP_BORDER};"
            )
        else:
            btn.setStyleSheet(
                f"background-color: {Theme.BG_DARK};"
                f"color: {Theme.TEXT_MAIN};"
                f"border: 1px solid {Theme.BORDER_EMPTY};"
            )

    def _clear_cells(self) -> None:
        for r in range(self.board_size):
            for c in range(self.board_size):
                self._selected[r][c] = False
                btn = self._buttons[r][c]
                btn.setChecked(False)
                self._apply_cell_style(btn, False)
        self._update_count_label()

    def _update_count_label(self) -> None:
        count = sum(1 for r in range(self.board_size) for c in range(self.board_size) if self._selected[r][c])
        if self.expected_cells > 0:
            self.count_label.setText(f"Selected ship cells: {count} / {self.expected_cells}")
        else:
            self.count_label.setText(f"Selected ship cells: {count}")

    def selected_cells(self) -> List[Tuple[int, int]]:
        cells: List[Tuple[int, int]] = []
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self._selected[r][c]:
                    cells.append((r, c))
        return cells
