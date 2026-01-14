import random
import uuid
from typing import Any, Dict, List, Optional, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

from battleship.domain.board import cell_index
from battleship.domain.config import DISP_RADIUS, HAS_SHIP, NO_SHIP, NO_SHOT, SHOT_HIT, SHOT_MISS
from battleship.layouts.cache import LayoutRuntime
from battleship.persistence.layout_state import load_layout_state, save_layout_state
from battleship.sim.defense_sim import (
    build_base_heat,
    recommend_layout_ga,
    recommend_layout_phase,
    simulate_enemy_game_phase,
)
from battleship.ui.theme import Theme


class DefenseTab(QtWidgets.QWidget):
    shot_recorded = QtCore.pyqtSignal()
    state_updated = QtCore.pyqtSignal()
    opponent_changed = QtCore.pyqtSignal(str)
    STATE_PATH = "battleship_defense_state.json"

    def __init__(self, layout_runtime: LayoutRuntime, parent=None):
        super().__init__(parent)
        self.layout_runtime = layout_runtime
        self.layout = layout_runtime.definition
        self.board_size = self.layout.board_size
        self.ship_ids = list(self.layout.ship_ids())
        self.expected_ship_cells = sum(
            int(spec.length or 0) if spec.kind == "line" else len(spec.cells or [])
            for spec in self.layout.ships
        )
        self.placements = layout_runtime.placements
        self.layout_board = [[NO_SHIP for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.shot_board = [[NO_SHOT for _ in range(self.board_size)] for _ in range(self.board_size)]

        self.hit_counts_phase = [
            [[0.0 for _ in range(self.board_size)] for _ in range(self.board_size)]
            for _ in range(4)
        ]
        self.miss_counts_phase = [
            [[0.0 for _ in range(self.board_size)] for _ in range(self.board_size)]
            for _ in range(4)
        ]
        self.disp_counts = [
            [
                [
                    [0.0 for _ in range(2 * DISP_RADIUS + 1)]
                    for _ in range(2 * DISP_RADIUS + 1)
                ]
                for _ in range(2)
            ]
            for _ in range(4)
        ]
        self.start_counts: List[float] = [0.0, 0.0, 0.0, 0.0]

        self.history_events: List[Tuple[int, int, bool, int]] = []
        self.last_shot_for_sequence: Optional[Tuple[int, int, int, bool]] = None

        self.recommended_layout = None
        self.recommended_mask = 0
        self.recommended_robust = 0.0
        self.recommended_heat = 0.0
        self.recommended_seq = 0.0
        self.eval_games_per_layout = 10
        self.eval_random_layouts = 12
        self.eval_results: Dict[str, object] = {}
        self._eval_worker: Optional["DefenseEvalWorker"] = None
        self._eval_progress: Optional[QtWidgets.QProgressDialog] = None
        self.heatmap_checkbox: Optional[QtWidgets.QCheckBox] = None

        self.opponents: Dict[str, Dict[str, object]] = {}
        self.active_opponent_id: Optional[str] = None
        self._loading_opponent = False

        self._build_ui()
        self.load_state()
        self.update_board_view()
        self.update_summary_labels()

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
        self.cell_buttons_def = []
        for r in range(self.board_size):
            row = []
            for c in range(self.board_size):
                btn = QtWidgets.QPushButton("")
                btn.setFixedSize(cell_size, cell_size)
                btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
                btn.clicked.connect(self._make_cell_handler(r, c))
                row.append(btn)
                board_layout.addWidget(btn, r + 1, c + 1)
            self.cell_buttons_def.append(row)

        self.board_container = board_container
        self.board_layout = board_layout
        board_container.installEventFilter(self)
        splitter.addWidget(board_container)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        header = QtWidgets.QLabel("Defense assistant")
        f = header.font()
        f.setPointSize(f.pointSize() + 1)
        f.setBold(True)
        header.setFont(f)
        right_layout.addWidget(header)

        desc = QtWidgets.QLabel(
            "Place your ships, record opponent shots, and evaluate layouts against attack models."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {Theme.TEXT_LABEL};")
        right_layout.addWidget(desc)

        opponent_row = QtWidgets.QHBoxLayout()
        self.opponent_label = QtWidgets.QLabel("Opponent:")
        opponent_row.addWidget(self.opponent_label)
        self.opponent_combo = QtWidgets.QComboBox()
        self.opponent_combo.currentIndexChanged.connect(self._on_opponent_changed)
        opponent_row.addWidget(self.opponent_combo, stretch=1)
        self.opponent_add_btn = QtWidgets.QPushButton("Add")
        self.opponent_add_btn.clicked.connect(self._add_opponent)
        opponent_row.addWidget(self.opponent_add_btn)
        self.opponent_rename_btn = QtWidgets.QPushButton("Rename")
        self.opponent_rename_btn.clicked.connect(self._rename_opponent)
        opponent_row.addWidget(self.opponent_rename_btn)
        self.opponent_delete_btn = QtWidgets.QPushButton("Delete")
        self.opponent_delete_btn.clicked.connect(self._delete_opponent)
        opponent_row.addWidget(self.opponent_delete_btn)
        right_layout.addLayout(opponent_row)

        self.mode_tabs = QtWidgets.QTabWidget()
        self.mode_tabs.currentChanged.connect(self._on_mode_changed)
        right_layout.addWidget(self.mode_tabs, stretch=1)

        place_container = QtWidgets.QWidget()
        place_layout = QtWidgets.QVBoxLayout(place_container)
        place_ctrl_group = QtWidgets.QGroupBox("Layout controls")
        place_ctrl_layout = QtWidgets.QVBoxLayout(place_ctrl_group)

        self.clear_layout_btn = QtWidgets.QPushButton("Clear layout")
        self.clear_layout_btn.clicked.connect(self.clear_layout)
        place_ctrl_layout.addWidget(self.clear_layout_btn)
        place_hint = QtWidgets.QLabel("Use Analyze tab to apply suggested layouts.")
        place_hint.setWordWrap(True)
        place_hint.setStyleSheet(f"color: {Theme.TEXT_MUTED};")
        place_ctrl_layout.addWidget(place_hint)

        place_layout.addWidget(place_ctrl_group)

        self.place_summary_label = QtWidgets.QLabel("")
        self.place_summary_label.setWordWrap(True)
        place_layout.addWidget(self.place_summary_label)
        place_layout.addStretch(1)

        self.place_tab = wrap_scroll(place_container)
        self.mode_tabs.addTab(self.place_tab, "Place")

        analyze_container = QtWidgets.QWidget()
        analyze_layout = QtWidgets.QVBoxLayout(analyze_container)
        shots_group = QtWidgets.QGroupBox("Opponent shots")
        shots_layout = QtWidgets.QVBoxLayout(shots_group)

        self.clear_shots_btn = QtWidgets.QPushButton("New game (clear board shots)")
        self.clear_shots_btn.clicked.connect(self.clear_shots)
        shots_layout.addWidget(self.clear_shots_btn)

        self.reset_model_btn = QtWidgets.QPushButton("Reset learning (all history)")
        self.reset_model_btn.clicked.connect(self.reset_model)
        shots_layout.addWidget(self.reset_model_btn)

        analyze_layout.addWidget(shots_group)

        suggest_group = QtWidgets.QGroupBox("Recommendations")
        suggest_layout = QtWidgets.QVBoxLayout(suggest_group)

        self.suggest_layout_btn = QtWidgets.QPushButton("Suggest layout from history")
        self.suggest_layout_btn.clicked.connect(self.compute_suggested_layout)
        suggest_layout.addWidget(self.suggest_layout_btn)
        self.apply_suggestion_btn = QtWidgets.QPushButton("Apply suggested layout")
        self.apply_suggested_layout_btn = self.apply_suggestion_btn
        self.apply_suggestion_btn.clicked.connect(self.apply_suggested_layout)
        suggest_layout.addWidget(self.apply_suggestion_btn)

        analyze_layout.addWidget(suggest_group)

        eval_group = QtWidgets.QGroupBox("Evaluation harness")
        eval_layout = QtWidgets.QVBoxLayout(eval_group)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Games per layout:"))
        self.eval_games_spin = QtWidgets.QSpinBox()
        self.eval_games_spin.setRange(1, 200)
        self.eval_games_spin.setValue(self.eval_games_per_layout)
        row.addWidget(self.eval_games_spin)
        eval_layout.addLayout(row)

        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel("Random layouts (baseline):"))
        self.eval_random_spin = QtWidgets.QSpinBox()
        self.eval_random_spin.setRange(0, 200)
        self.eval_random_spin.setValue(self.eval_random_layouts)
        row2.addWidget(self.eval_random_spin)
        eval_layout.addLayout(row2)

        self.eval_button = QtWidgets.QPushButton("Evaluate layout")
        self.eval_button.clicked.connect(self.run_evaluation)
        eval_layout.addWidget(self.eval_button)

        self.eval_result_label = QtWidgets.QLabel("No evaluation run yet.")
        self.eval_result_label.setWordWrap(True)
        eval_layout.addWidget(self.eval_result_label)

        analyze_layout.addWidget(eval_group)

        self.heatmap_checkbox = QtWidgets.QCheckBox("Show total shot counts (heatmap overlay)")
        self.heatmap_checkbox.stateChanged.connect(self.update_board_view)
        analyze_layout.addWidget(self.heatmap_checkbox)

        self.summary_label = QtWidgets.QLabel("")
        self.summary_label.setWordWrap(True)
        analyze_layout.addWidget(self.summary_label)

        self.pattern_label = QtWidgets.QLabel("")
        self.pattern_label.setWordWrap(True)
        analyze_layout.addWidget(self.pattern_label)

        self.recommendation_label = QtWidgets.QLabel(
            "No layout suggestion yet.\nRecord opponent shots, then click 'Suggest layout'."
        )
        self.recommendation_label.setWordWrap(True)
        analyze_layout.addWidget(self.recommendation_label)
        analyze_layout.addStretch(1)

        self.analyze_tab = wrap_scroll(analyze_container)
        self.mode_tabs.addTab(self.analyze_tab, "Analyze")

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([800, 450])

        self._resize_board()

    def _blank_board(self, fill_value: int) -> List[List[int]]:
        return [[fill_value for _ in range(self.board_size)] for _ in range(self.board_size)]

    def _blank_phase_counts(self) -> List[List[List[float]]]:
        return [
            [[0.0 for _ in range(self.board_size)] for _ in range(self.board_size)]
            for _ in range(4)
        ]

    def _blank_disp_counts(self) -> List[List[List[List[float]]]]:
        return [
            [
                [
                    [0.0 for _ in range(2 * DISP_RADIUS + 1)]
                    for _ in range(2 * DISP_RADIUS + 1)
                ]
                for _ in range(2)
            ]
            for _ in range(4)
        ]

    def _blank_start_counts(self) -> List[float]:
        return [0.0, 0.0, 0.0, 0.0]

    def _new_opponent_profile(self, name: str) -> Dict[str, object]:
        return {
            "name": name,
            "layout_board": self._blank_board(NO_SHIP),
            "shot_board": self._blank_board(NO_SHOT),
            "hit_counts_phase": self._blank_phase_counts(),
            "miss_counts_phase": self._blank_phase_counts(),
            "disp_counts": self._blank_disp_counts(),
            "start_counts": self._blank_start_counts(),
            "history_events": [],
            "last_shot_for_sequence": None,
            "mode_tab": 0,
            "eval_games_per_layout": int(self.eval_games_per_layout),
            "eval_random_layouts": int(self.eval_random_layouts),
            "eval_results": {},
        }

    def _export_profile(self) -> Dict[str, object]:
        return {
            "name": self._current_opponent_name(),
            "layout_board": self.layout_board,
            "shot_board": self.shot_board,
            "hit_counts_phase": self.hit_counts_phase,
            "miss_counts_phase": self.miss_counts_phase,
            "disp_counts": self.disp_counts,
            "start_counts": list(self.start_counts),
            "history_events": [
                [int(r), int(c), bool(was_hit), int(phase)]
                for (r, c, was_hit, phase) in self.history_events
            ],
            "last_shot_for_sequence": (
                [int(self.last_shot_for_sequence[0]),
                 int(self.last_shot_for_sequence[1]),
                 int(self.last_shot_for_sequence[2]),
                 bool(self.last_shot_for_sequence[3])]
                if self.last_shot_for_sequence is not None
                else None
            ),
            "mode_tab": int(self.mode_tabs.currentIndex()) if hasattr(self, "mode_tabs") else 0,
            "eval_games_per_layout": int(self.eval_games_spin.value()) if hasattr(self, "eval_games_spin") else 0,
            "eval_random_layouts": int(self.eval_random_spin.value()) if hasattr(self, "eval_random_spin") else 0,
            "eval_results": self.eval_results,
        }

    def _import_profile(self, profile: Dict[str, object]) -> None:
        self._loading_opponent = True
        try:
            self.recommended_layout = None
            self.recommended_mask = 0
            self.recommended_robust = 0.0
            self.recommended_heat = 0.0
            self.recommended_seq = 0.0

            lb = profile.get("layout_board")
            if (
                isinstance(lb, list)
                and len(lb) == self.board_size
                and all(isinstance(row, list) and len(row) == self.board_size for row in lb)
            ):
                self.layout_board = lb
            else:
                self.layout_board = self._blank_board(NO_SHIP)

            sb = profile.get("shot_board")
            if (
                isinstance(sb, list)
                and len(sb) == self.board_size
                and all(isinstance(row, list) and len(row) == self.board_size for row in sb)
            ):
                self.shot_board = sb
            else:
                self.shot_board = self._blank_board(NO_SHOT)

            hc = profile.get("hit_counts_phase")
            mc = profile.get("miss_counts_phase")
            if isinstance(hc, list) and isinstance(mc, list) and len(hc) == 4 and len(mc) == 4:
                self.hit_counts_phase = hc
                self.miss_counts_phase = mc
            else:
                self.hit_counts_phase = self._blank_phase_counts()
                self.miss_counts_phase = self._blank_phase_counts()

            dc = profile.get("disp_counts")
            if isinstance(dc, list) and len(dc) == 4:
                self.disp_counts = dc
            else:
                self.disp_counts = self._blank_disp_counts()

            sc = profile.get("start_counts")
            if isinstance(sc, list) and len(sc) == 4 and all(isinstance(v, (int, float)) for v in sc):
                self.start_counts = [float(v) for v in sc]
            else:
                self.start_counts = self._blank_start_counts()

            self.history_events = []
            he = profile.get("history_events")
            if isinstance(he, list):
                for item in he:
                    if (
                        isinstance(item, list)
                        and len(item) == 4
                        and isinstance(item[0], int)
                        and isinstance(item[1], int)
                        and isinstance(item[3], int)
                    ):
                        r, c, was_hit, phase = item
                        if 0 <= r < self.board_size and 0 <= c < self.board_size and 0 <= phase <= 3:
                            self.history_events.append((r, c, bool(was_hit), phase))

            lss = profile.get("last_shot_for_sequence")
            if (
                isinstance(lss, list)
                and len(lss) == 4
                and isinstance(lss[0], int)
                and isinstance(lss[1], int)
                and isinstance(lss[2], int)
            ):
                lr, lc, phase, was_hit = lss
                if 0 <= lr < self.board_size and 0 <= lc < self.board_size and 0 <= phase <= 3:
                    self.last_shot_for_sequence = (lr, lc, phase, bool(was_hit))
                else:
                    self.last_shot_for_sequence = None
            else:
                self.last_shot_for_sequence = None

            mode_tab = profile.get("mode_tab")
            if isinstance(mode_tab, int) and 0 <= mode_tab < self.mode_tabs.count():
                self.mode_tabs.setCurrentIndex(mode_tab)

            eval_games = profile.get("eval_games_per_layout")
            if isinstance(eval_games, int):
                self.eval_games_per_layout = eval_games
                self.eval_games_spin.setValue(eval_games)
            eval_random = profile.get("eval_random_layouts")
            if isinstance(eval_random, int):
                self.eval_random_layouts = eval_random
                self.eval_random_spin.setValue(eval_random)

            eval_results = profile.get("eval_results")
            if isinstance(eval_results, dict):
                self.eval_results = eval_results
                self._update_eval_results_label()
            else:
                self.eval_results = {}
                self._update_eval_results_label()
        finally:
            self._loading_opponent = False
        self.update_board_view()
        self.update_summary_labels()

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
        self.opponents[self.active_opponent_id] = self._export_profile()

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
            self._import_profile(profile)
        self.save_state()
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
        self._import_profile(self.opponents[oid])
        self.save_state()

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
        self._import_profile(self.opponents[self.active_opponent_id])
        self.save_state()

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
        self._import_profile(self.opponents[oid])
        self.save_state()

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
            self._import_profile(profile)
        self.save_state()
        return True

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
        for row in self.cell_buttons_def:
            for btn in row:
                btn.setFixedSize(cell, cell)

    def save_state(self, path: Optional[str] = None):
        if path is None:
            path = self.STATE_PATH
        self._save_current_profile()

        state = {
            "layout_id": self.layout.layout_id,
            "layout_version": self.layout.layout_version,
            "layout_hash": self.layout.layout_hash,
            "active_opponent_id": self.active_opponent_id,
            "opponents": self.opponents,
        }
        save_layout_state(path, self.layout, state)

    def load_state(self, path: Optional[str] = None):
        if path is None:
            path = self.STATE_PATH
        data, _raw = load_layout_state(path, self.layout)
        if not data:
            self._ensure_opponents()
            profile = self.opponents.get(self.active_opponent_id, {})
            if isinstance(profile, dict):
                self._import_profile(profile)
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
                self._import_profile(profile)
            return

        # Legacy single-opponent payload migration
        oid = f"opp_{uuid.uuid4().hex[:8]}"
        profile = self._new_opponent_profile("Opponent 1")

        for key in (
            "hit_counts_phase",
            "miss_counts_phase",
            "disp_counts",
            "layout_board",
            "shot_board",
            "history_events",
            "last_shot_for_sequence",
            "mode_tab",
            "eval_games_per_layout",
            "eval_random_layouts",
            "eval_results",
        ):
            if key in data:
                profile[key] = data.get(key)

        self.opponents = {oid: profile}
        self.active_opponent_id = oid
        self._ensure_opponents()
        self._import_profile(profile)

    def _is_place_mode(self) -> bool:
        return self.mode_tabs.currentIndex() == 0

    def _on_mode_changed(self, _index: int):
        self.update_board_view()

    def _make_cell_handler(self, r: int, c: int):
        def handler():
            if self._is_place_mode():
                self.toggle_ship_cell(r, c)
            else:
                self.record_shot_at(r, c)

        return handler

    def toggle_ship_cell(self, r: int, c: int):
        self.layout_board[r][c] = HAS_SHIP if self.layout_board[r][c] == NO_SHIP else NO_SHIP
        self.update_board_view()
        self.update_summary_labels()
        try:
            self.state_updated.emit()
        except Exception:
            pass

    def clear_layout(self):
        for r in range(self.board_size):
            for c in range(self.board_size):
                self.layout_board[r][c] = NO_SHIP
        self.update_board_view()
        self.update_summary_labels()

    def _decay_counts(self, factor: float = 0.97):
        """
        Exponentially decay all learned statistics so that recent games
        have more influence than very old ones.

        Counts are stored as floats; we deliberately *do not* truncate to
        integers here, otherwise low-frequency patterns vanish too fast.
        """
        for p in range(4):
            # Per-cell hit/miss counts
            for r in range(self.board_size):
                for c in range(self.board_size):
                    self.hit_counts_phase[p][r][c] *= factor
                    self.miss_counts_phase[p][r][c] *= factor

            # Shot-displacement counts (sequence model)
            for hm in range(2):
                for dr in range(2 * DISP_RADIUS + 1):
                    for dc in range(2 * DISP_RADIUS + 1):
                        self.disp_counts[p][hm][dr][dc] *= factor

        for i in range(len(self.start_counts)):
            self.start_counts[i] *= factor

    def _compute_sunk_ship_count(self) -> int:
        visited = [[False for _ in range(self.board_size)] for _ in range(self.board_size)]
        sunk_count = 0
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.layout_board[r][c] != HAS_SHIP or visited[r][c]:
                    continue
                stack = [(r, c)]
                visited[r][c] = True
                cells = []
                while stack:
                    cr, cc = stack.pop()
                    cells.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        rr = cr + dr
                        cc2 = cc + dc
                        if 0 <= rr < self.board_size and 0 <= cc2 < self.board_size and not visited[rr][cc2] and \
                                self.layout_board[rr][cc2] == HAS_SHIP:
                            visited[rr][cc2] = True
                            stack.append((rr, cc2))
                sunk = True
                for cr, cc in cells:
                    if self.shot_board[cr][cc] != SHOT_HIT:
                        sunk = False
                        break
                if sunk:
                    sunk_count += 1
        return sunk_count

    def _nearest_corner_index(self, r: int, c: int) -> int:
        corners = [
            (0, 0),
            (0, self.board_size - 1),
            (self.board_size - 1, 0),
            (self.board_size - 1, self.board_size - 1),
        ]
        best_idx = 0
        best_dist = None
        for idx, (cr, cc) in enumerate(corners):
            dist = abs(r - cr) + abs(c - cc)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx

    def record_shot_at(self, r: int, c: int):
        has_ship = self.layout_board[r][c] == HAS_SHIP
        current = self.shot_board[r][c]
        if current == NO_SHOT:
            phase = self._compute_sunk_ship_count()
            if phase > 3:
                phase = 3
            if has_ship:
                self.shot_board[r][c] = SHOT_HIT
                self.hit_counts_phase[phase][r][c] += 1
                was_hit = True
            else:
                self.shot_board[r][c] = SHOT_MISS
                self.miss_counts_phase[phase][r][c] += 1
                was_hit = False
            if not self.history_events:
                corner_idx = self._nearest_corner_index(r, c)
                if 0 <= corner_idx < len(self.start_counts):
                    self.start_counts[corner_idx] += 1.0
            self.history_events.append((r, c, was_hit, phase))
            if self.last_shot_for_sequence is not None:
                lr, lc, last_phase, last_hit = self.last_shot_for_sequence
                dr = r - lr
                dc = c - lc
                if abs(dr) <= DISP_RADIUS and abs(dc) <= DISP_RADIUS:
                    phase_idx = min(last_phase, 3)
                    type_idx = 1 if last_hit else 0
                    self.disp_counts[phase_idx][type_idx][dr + DISP_RADIUS][dc + DISP_RADIUS] += 1
            self.last_shot_for_sequence = (r, c, phase, was_hit)
            try:
                self.shot_recorded.emit()
            except Exception:
                pass
        else:
            # undo
            for i in range(len(self.history_events) - 1, -1, -1):
                er, ec, was_hit, phase = self.history_events[i]
                if er == r and ec == c:
                    self.history_events.pop(i)
                    if i == 0:
                        corner_idx = self._nearest_corner_index(er, ec)
                        if 0 <= corner_idx < len(self.start_counts) and self.start_counts[corner_idx] > 0:
                            self.start_counts[corner_idx] -= 1.0
                    if was_hit:
                        if self.hit_counts_phase[phase][r][c] > 0:
                            self.hit_counts_phase[phase][r][c] -= 1
                    else:
                        if self.miss_counts_phase[phase][r][c] > 0:
                            self.miss_counts_phase[phase][r][c] -= 1
                    break
            self.shot_board[r][c] = NO_SHOT
            if self.last_shot_for_sequence is not None:
                lr, lc, _, _ = self.last_shot_for_sequence
                if lr == r and lc == c:
                    self.last_shot_for_sequence = None
        self.update_board_view()
        self.update_summary_labels()
        try:
            self.state_updated.emit()
        except Exception:
            pass

    def clear_shots(self):
        # Treat this as "end of game": decay historical stats so that
        # patterns from very old games matter less than recent ones.
        self._decay_counts()

        for r in range(self.board_size):
            for c in range(self.board_size):
                self.shot_board[r][c] = NO_SHOT

        self.history_events.clear()
        self.last_shot_for_sequence = None
        self.update_board_view()
        self.update_summary_labels()
        try:
            self.state_updated.emit()
        except Exception:
            pass

    def reset_model(self):
        for p in range(4):
            for r in range(self.board_size):
                for c in range(self.board_size):
                    self.hit_counts_phase[p][r][c] = 0
                    self.miss_counts_phase[p][r][c] = 0
        for p in range(4):
            for hm in range(2):
                for dr in range(2 * DISP_RADIUS + 1):
                    for dc in range(2 * DISP_RADIUS + 1):
                        self.disp_counts[p][hm][dr][dc] = 0
        self.start_counts = self._blank_start_counts()
        self.history_events.clear()
        self.last_shot_for_sequence = None
        for r in range(self.board_size):
            for c in range(self.board_size):
                self.shot_board[r][c] = NO_SHOT
        self.recommended_layout = None
        self.recommended_mask = 0
        self.recommended_robust = 0.0
        self.recommended_heat = 0.0
        self.recommended_seq = 0.0
        self.update_board_view()
        self.update_summary_labels()
        self.recommendation_label.setText(
            "Learning reset.\nRecord opponent shots, then click 'Suggest layout'."
        )
        self.save_state()
        try:
            self.state_updated.emit()
        except Exception:
            pass

    def compute_suggested_layout(self):
        base_heat_phase, _pattern = self._build_base_heat_phase()
        layout, mask, robust, avg_heat, avg_seq = recommend_layout_ga(
            self.hit_counts_phase,
            self.miss_counts_phase,
            self.disp_counts,
            self.placements,
            ship_ids=self.ship_ids,
            board_size=self.board_size,
            generations=18,
            population_size=36,
            sim_games_per_layout=8,
            base_heat_phase=base_heat_phase,
        )
        if layout is None:
            layout, mask, robust, avg_heat, avg_seq = recommend_layout_phase(
                self.hit_counts_phase,
                self.miss_counts_phase,
                self.disp_counts,
                self.placements,
                ship_ids=self.ship_ids,
                board_size=self.board_size,
                n_iter=250,
                sim_games_per_layout=10,
                base_heat_phase=base_heat_phase,
            )
        if layout is None:
            self.recommended_layout = None
            self.recommended_mask = 0
            self.recommended_robust = 0.0
            self.recommended_heat = 0.0
            self.recommended_seq = 0.0
            self.recommendation_label.setText(
                "Could not compute a layout suggestion (no valid combination)."
            )
        else:
            self.recommended_layout = layout
            self.recommended_mask = mask
            self.recommended_robust = robust
            self.recommended_heat = avg_heat
            self.recommended_seq = avg_seq
            total_hits, total_misses = self._total_hits_misses()
            total_shots = total_hits + total_misses
            total_hits_i = int(round(total_hits))
            total_misses_i = int(round(total_misses))
            total_shots_i = int(round(total_shots))
            if total_shots == 0:
                extra = "No history yet; layout is effectively random."
            else:
                extra = (
                    f"Using {total_shots_i} recorded shots ({total_hits_i} hits, {total_misses_i} misses).\n"
                    f"Estimated shots to sink all ships:\n"
                    f"  Heatmap model: {avg_heat:.1f}\n"
                    f"  Sequence-aware model: {avg_seq:.1f}\n"
                    f"Robust score (min of the two): {robust:.1f}"
                )
            self.recommendation_label.setText(
                "Suggested layout ready.\n" + extra
            )
        self.update_board_view()

    def apply_suggested_layout(self):
        if not self.recommended_layout:
            QtWidgets.QMessageBox.information(
                self,
                "No suggestion",
                "No suggested layout available.\nRecord some shots and click 'Suggest layout' first.",
            )
            return
        for r in range(self.board_size):
            for c in range(self.board_size):
                self.layout_board[r][c] = NO_SHIP
        for p in self.recommended_layout.values():
            for r, c in p.cells:
                self.layout_board[r][c] = HAS_SHIP
        # clear preview
        self.recommended_mask = 0
        self.recommended_layout = None
        self.recommended_robust = 0.0
        self.recommended_heat = 0.0
        self.recommended_seq = 0.0
        self.recommendation_label.setText(
            "Suggested layout applied to current board."
        )
        self.update_board_view()
        self.update_summary_labels()

    def _layout_mask_from_board(self) -> int:
        mask = 0
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.layout_board[r][c] == HAS_SHIP:
                    idx = cell_index(r, c, self.board_size)
                    mask |= 1 << idx
        return mask

    def run_evaluation(self):
        if self._eval_worker is not None:
            QtWidgets.QMessageBox.information(self, "Busy", "Evaluation is already running.")
            return

        current_mask = self._layout_mask_from_board()
        suggested_mask = self.recommended_mask if self.recommended_mask else 0
        games_per_layout = int(self.eval_games_spin.value())
        random_layouts = int(self.eval_random_spin.value())

        self.eval_games_per_layout = games_per_layout
        self.eval_random_layouts = random_layouts

        if current_mask == 0 and suggested_mask == 0 and random_layouts == 0:
            QtWidgets.QMessageBox.information(
                self,
                "Nothing to evaluate",
                "Place some ships or enable a random baseline before running evaluation.",
            )
            return

        self._eval_progress = QtWidgets.QProgressDialog("Evaluating layouts...", "Cancel", 0, 100, self)
        self._eval_progress.setWindowModality(QtCore.Qt.WindowModal)
        self._eval_progress.setMinimumDuration(0)
        self._eval_progress.setValue(0)
        self._eval_progress.setAutoClose(False)
        self._eval_progress.setAutoReset(False)

        base_heat_phase, _pattern = self._build_base_heat_phase()
        self._eval_worker = DefenseEvalWorker(
            current_mask=current_mask,
            suggested_mask=suggested_mask,
            placements=self.placements,
            ship_ids=self.ship_ids,
            board_size=self.board_size,
            hit_counts_phase=self.hit_counts_phase,
            miss_counts_phase=self.miss_counts_phase,
            disp_counts=self.disp_counts,
            base_heat_phase=base_heat_phase,
            games_per_layout=games_per_layout,
            random_layouts=random_layouts,
        )
        self._eval_worker.progress.connect(self._on_eval_progress)
        self._eval_worker.finished.connect(self._on_eval_finished)
        self._eval_worker.error.connect(self._on_eval_error)
        self._eval_progress.canceled.connect(self._eval_worker.cancel)
        self._eval_worker.start()

    def _on_eval_progress(self, done: int, total: int):
        if not self._eval_progress:
            return
        self._eval_progress.setMaximum(max(1, total))
        self._eval_progress.setValue(done)

    def _on_eval_finished(self, results: Dict[str, object]):
        self.eval_results = results or {}
        if self._eval_progress is not None:
            self._eval_progress.setValue(self._eval_progress.maximum())
            self._eval_progress.close()
        self._eval_progress = None
        if self._eval_worker is not None:
            self._eval_worker = None
        self._update_eval_results_label()
        self.save_state()

    def _on_eval_error(self, message: str):
        if self._eval_progress is not None:
            self._eval_progress.close()
        self._eval_progress = None
        self._eval_worker = None
        QtWidgets.QMessageBox.critical(self, "Evaluation error", message)

    def _update_eval_results_label(self):
        if not self.eval_results:
            self.eval_result_label.setText("No evaluation run yet.")
            return

        def fmt_block(title: str, data: Dict[str, object]) -> str:
            avg_heat = data.get("avg_heat")
            avg_seq = data.get("avg_seq")
            robust = data.get("robust")
            games = data.get("games", 0)
            layouts = data.get("layouts", None)
            parts = [f"{title} (games={games})"]
            if layouts is not None:
                parts[0] += f", layouts={layouts}"
            parts.append(f"Heat: {avg_heat:.1f}" if isinstance(avg_heat, (int, float)) else "Heat: N/A")
            parts.append(f"Seq: {avg_seq:.1f}" if isinstance(avg_seq, (int, float)) else "Seq: N/A")
            parts.append(f"Robust: {robust:.1f}" if isinstance(robust, (int, float)) else "Robust: N/A")
            return " | ".join(parts)

        lines = []
        current = self.eval_results.get("current")
        if isinstance(current, dict):
            lines.append(fmt_block("Current layout", current))
        suggested = self.eval_results.get("suggested")
        if isinstance(suggested, dict):
            lines.append(fmt_block("Suggested layout", suggested))
        baseline = self.eval_results.get("baseline_random")
        if isinstance(baseline, dict):
            lines.append(fmt_block("Random baseline", baseline))

        self.eval_result_label.setText("\n".join(lines))

    def _total_hits_misses(self) -> Tuple[int, int]:
        total_hits = 0
        total_misses = 0
        for p in range(4):
            for r in range(self.board_size):
                for c in range(self.board_size):
                    total_hits += self.hit_counts_phase[p][r][c]
                    total_misses += self.miss_counts_phase[p][r][c]
        return total_hits, total_misses

    def _total_counts_cell(self, r: int, c: int) -> Tuple[int, int]:
        hits = 0
        misses = 0
        for p in range(4):
            hits += self.hit_counts_phase[p][r][c]
            misses += self.miss_counts_phase[p][r][c]
        return hits, misses

    def _total_disp_samples(self) -> int:
        total = 0
        for p in range(4):
            for hm in range(2):
                for dr in range(2 * DISP_RADIUS + 1):
                    for dc in range(2 * DISP_RADIUS + 1):
                        total += self.disp_counts[p][hm][dr][dc]
        return total

    def _quadrant_index(self, r: int, c: int) -> int:
        mid = self.board_size // 2
        top = r < mid
        left = c < mid
        if top and left:
            return 0  # NW
        if top and not left:
            return 1  # NE
        if not top and left:
            return 2  # SW
        return 3  # SE

    def _compute_pattern_bias(self) -> Dict[str, object]:
        total = 0.0
        even = 0.0
        quad_counts = [0.0, 0.0, 0.0, 0.0]
        for r in range(self.board_size):
            for c in range(self.board_size):
                h, m = self._total_counts_cell(r, c)
                v = float(h + m)
                if v <= 0:
                    continue
                total += v
                if (r + c) % 2 == 0:
                    even += v
                q = self._quadrant_index(r, c)
                quad_counts[q] += v

        pattern: Dict[str, object] = {"total_shots": total}
        if total <= 0.0:
            return pattern

        odd = total - even
        parity_bias = (even - odd) / total if total > 0 else 0.0
        parity_strength = min(0.5, abs(parity_bias) * 1.2)
        if abs(parity_bias) >= 0.08:
            pattern["parity_pref"] = "even" if parity_bias >= 0 else "odd"
            pattern["parity_strength"] = parity_strength
            pattern["parity_bias"] = parity_bias

        quad_shares = [q / total for q in quad_counts]
        max_share = max(quad_shares)
        max_idx = quad_shares.index(max_share)
        quad_delta = max_share - 0.25
        quad_strength = min(0.5, max(0.0, quad_delta * 3.0))
        if quad_delta >= 0.06:
            pattern["quadrant_pref"] = max_idx
            pattern["quadrant_strength"] = quad_strength
            pattern["quadrant_share"] = max_share

        start_total = float(sum(self.start_counts))
        pattern["start_total"] = start_total
        if start_total > 0:
            start_shares = [c / start_total for c in self.start_counts]
            start_max = max(start_shares)
            start_idx = start_shares.index(start_max)
            start_delta = start_max - 0.25
            start_strength = min(0.4, max(0.0, start_delta * 3.0))
            if start_delta >= 0.10:
                pattern["start_corner_pref"] = start_idx
                pattern["start_corner_strength"] = start_strength
                pattern["start_corner_share"] = start_max
        return pattern

    def _apply_pattern_bias(self, heat, pattern: Dict[str, object]):
        if not pattern:
            return heat

        parity_pref = pattern.get("parity_pref")
        parity_strength = float(pattern.get("parity_strength", 0.0) or 0.0)
        quad_pref = pattern.get("quadrant_pref")
        quad_strength = float(pattern.get("quadrant_strength", 0.0) or 0.0)
        corner_pref = pattern.get("start_corner_pref")
        corner_strength = float(pattern.get("start_corner_strength", 0.0) or 0.0)

        if not parity_pref and quad_pref is None and corner_pref is None:
            return heat

        corners = [
            (0, 0),
            (0, self.board_size - 1),
            (self.board_size - 1, 0),
            (self.board_size - 1, self.board_size - 1),
        ]
        denom = max(1, (self.board_size - 1) * 2)

        weighted = [[0.0 for _ in range(self.board_size)] for _ in range(self.board_size)]
        total = 0.0
        for r in range(self.board_size):
            for c in range(self.board_size):
                factor = 1.0
                if parity_pref:
                    is_even = (r + c) % 2 == 0
                    if (parity_pref == "even") == is_even:
                        factor *= 1.0 + parity_strength
                if quad_pref is not None:
                    if self._quadrant_index(r, c) == quad_pref:
                        factor *= 1.0 + quad_strength
                if corner_pref is not None:
                    cr, cc = corners[int(corner_pref)]
                    dist = abs(r - cr) + abs(c - cc)
                    decay = max(0.0, 1.0 - (dist / denom))
                    factor *= 1.0 + corner_strength * decay
                val = heat[r][c] * factor
                weighted[r][c] = val
                total += val

        if total <= 0:
            return heat
        for r in range(self.board_size):
            for c in range(self.board_size):
                weighted[r][c] /= total
        return weighted

    def _build_base_heat_phase(self):
        base_heat_phase = [
            build_base_heat(self.hit_counts_phase[p], self.miss_counts_phase[p], self.board_size)
            for p in range(4)
        ]
        pattern = self._compute_pattern_bias()
        for i in range(len(base_heat_phase)):
            base_heat_phase[i] = self._apply_pattern_bias(base_heat_phase[i], pattern)
        return base_heat_phase, pattern

    def _pattern_summary_text(self, pattern: Dict[str, object]) -> str:
        total = float(pattern.get("total_shots", 0.0) or 0.0)
        if total <= 0:
            return "Pattern bias: none yet (record more opponent shots)."

        parts: List[str] = []
        parity_pref = pattern.get("parity_pref")
        if isinstance(parity_pref, str):
            bias = float(pattern.get("parity_bias", 0.0) or 0.0)
            parts.append(f"Parity: {parity_pref} ({bias:+.2f})")

        quad_pref = pattern.get("quadrant_pref")
        if isinstance(quad_pref, int):
            names = ["NW", "NE", "SW", "SE"]
            share = float(pattern.get("quadrant_share", 0.0) or 0.0)
            label = names[quad_pref] if 0 <= quad_pref < len(names) else str(quad_pref)
            parts.append(f"Quadrant: {label} ({share * 100:.0f}%)")

        corner_pref = pattern.get("start_corner_pref")
        if isinstance(corner_pref, int):
            names = ["NW", "NE", "SW", "SE"]
            share = float(pattern.get("start_corner_share", 0.0) or 0.0)
            label = names[corner_pref] if 0 <= corner_pref < len(names) else str(corner_pref)
            parts.append(f"Start corner: {label} ({share * 100:.0f}%)")

        if not parts:
            return "Pattern bias: none detected yet."
        return "Pattern bias: " + " | ".join(parts)

    def update_summary_labels(self):
        total_hits, total_misses = self._total_hits_misses()
        total_shots = total_hits + total_misses
        total_hits_i = int(round(total_hits))
        total_misses_i = int(round(total_misses))
        total_shots_i = int(round(total_shots))
        total_layout_cells = sum(
            self.layout_board[r][c] == HAS_SHIP
            for r in range(self.board_size)
            for c in range(self.board_size)
        )
        sunk_now = self._compute_sunk_ship_count()
        disp_samples = self._total_disp_samples()
        disp_samples_i = int(round(disp_samples))
        self.summary_label.setText(
            f"Total recorded opponent shots (all games, all phases): {total_shots_i}\n"
            f"Hits: {total_hits_i}, Misses: {total_misses_i}\n"
            f"Current layout: {total_layout_cells} ship cells (ideal is ~{self.expected_ship_cells}).\n"
            f"Ships sunk this game (approx via clusters): {sunk_now}\n"
            f"Sequence samples (displacements between shots): {disp_samples_i}"
        )
        self.place_summary_label.setText(
            f"Current layout: {total_layout_cells} ship cells (ideal is ~{self.expected_ship_cells})."
        )
        if hasattr(self, "pattern_label"):
            pattern = self._compute_pattern_bias()
            self.pattern_label.setText(self._pattern_summary_text(pattern))

    def update_board_view(self):
        show_heat = bool(self.heatmap_checkbox.isChecked()) if self.heatmap_checkbox is not None else False

        # First pass: max total count for normalization
        max_count = 0.0
        for r in range(self.board_size):
            for c in range(self.board_size):
                h, m = self._total_counts_cell(r, c)
                cc = h + m
                if cc > max_count:
                    max_count = cc

        for r in range(self.board_size):
            for c in range(self.board_size):
                btn = self.cell_buttons_def[r][c]
                has_ship = (self.layout_board[r][c] == HAS_SHIP)
                shot_state = self.shot_board[r][c]
                if self._is_place_mode():
                    shot_state = NO_SHOT

                hcount, mcount = self._total_counts_cell(r, c)
                total_count = hcount + mcount

                base_style_parts: List[str] = []
                text = ""

                if shot_state == NO_SHOT:
                    if has_ship:
                        # Our own ship that hasn't been shot yet
                        base_style_parts.append(f"background-color: {Theme.LAYOUT_SHIP_BG};")
                        base_style_parts.append(f"color: {Theme.LAYOUT_SHIP_TEXT};")
                        base_style_parts.append(f"border: 1px solid {Theme.LAYOUT_SHIP_BORDER};")
                    else:
                        # Empty water, no shot yet
                        base_style_parts.append(f"background-color: {Theme.BG_DARK};")
                        base_style_parts.append(f"color: {Theme.TEXT_MAIN};")
                        base_style_parts.append(f"border: 1px solid {Theme.BORDER_EMPTY};")
                elif shot_state == SHOT_MISS:
                    base_style_parts.append(f"background-color: {Theme.MISS_BG};")
                    base_style_parts.append(f"color: {Theme.MISS_TEXT};")
                    base_style_parts.append(f"border: 1px solid {Theme.MISS_BORDER};")
                    text = "M"
                elif shot_state == SHOT_HIT:
                    base_style_parts.append(f"background-color: {Theme.HIT_BG};")
                    base_style_parts.append(f"color: {Theme.HIT_TEXT};")
                    base_style_parts.append(f"border: 1px solid {Theme.HIT_BORDER};")
                    text = "H"

                # Optional heat overlay (only on unknown cells)
                if show_heat and shot_state == NO_SHOT and total_count > 0:
                    text = str(int(round(total_count)))
                    if max_count > 0:
                        alpha = total_count / max_count
                        border_w = 1 + int(round(alpha * 2))
                        base_style_parts = [
                            f"background-color: {Theme.BG_DARK};",
                            f"color: {Theme.HEAT_TEXT};",
                            f"border: {border_w}px solid {Theme.HEAT_BORDER};",
                        ]

                # Recommended layout highlight
                if self.recommended_mask:
                    idx = cell_index(r, c, self.board_size)
                    if (self.recommended_mask >> idx) & 1:
                        base_style_parts.append(f"border: 2px solid {Theme.HIGHLIGHT};")

                btn.setStyleSheet("".join(base_style_parts))
                btn.setText(text)


class DefenseEvalWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int)
    finished = QtCore.pyqtSignal(dict)
    error = QtCore.pyqtSignal(str)

    def __init__(
        self,
        current_mask: int,
        suggested_mask: int,
        placements: Dict[str, List[object]],
        ship_ids: List[str],
        board_size: int,
        hit_counts_phase,
        miss_counts_phase,
        disp_counts,
        games_per_layout: int,
        random_layouts: int,
        base_heat_phase=None,
    ):
        super().__init__()
        self.current_mask = int(current_mask)
        self.suggested_mask = int(suggested_mask)
        self.placements = placements
        self.ship_ids = list(ship_ids)
        self.board_size = int(board_size)
        self.hit_counts_phase = hit_counts_phase
        self.miss_counts_phase = miss_counts_phase
        self.disp_counts = disp_counts
        self.base_heat_phase = base_heat_phase
        self.games_per_layout = int(games_per_layout)
        self.random_layouts = int(random_layouts)
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def _random_layout_mask(self, rng: random.Random) -> Optional[int]:
        used_mask = 0
        ship_order = list(self.ship_ids)
        rng.shuffle(ship_order)
        for ship in ship_order:
            options = self.placements.get(ship, [])
            if not options:
                return None
            placed = False
            for _ in range(50):
                p = rng.choice(options)
                if p.mask & used_mask:
                    continue
                used_mask |= p.mask
                placed = True
                break
            if not placed:
                return None
        return used_mask

    def _eval_mask(self, label: str, mask: int, base_heat_phase, rng: random.Random, total_jobs: int, done: int):
        total_heat = 0.0
        total_seq = 0.0
        for _ in range(self.games_per_layout):
            if self._cancelled:
                break
            sh, _ = simulate_enemy_game_phase(
                mask,
                base_heat_phase,
                self.disp_counts,
                "heat",
                rng,
                board_size=self.board_size,
            )
            total_heat += sh
            done += 1
            self.progress.emit(done, total_jobs)

            if self._cancelled:
                break
            ss, _ = simulate_enemy_game_phase(
                mask,
                base_heat_phase,
                self.disp_counts,
                "seq",
                rng,
                board_size=self.board_size,
            )
            total_seq += ss
            done += 1
            self.progress.emit(done, total_jobs)
        if self._cancelled:
            return None, done
        avg_heat = total_heat / self.games_per_layout if self.games_per_layout else 0.0
        avg_seq = total_seq / self.games_per_layout if self.games_per_layout else 0.0
        return {
            "label": label,
            "games": self.games_per_layout,
            "avg_heat": avg_heat,
            "avg_seq": avg_seq,
            "robust": min(avg_heat, avg_seq),
        }, done

    def run(self):
        try:
            rng = random.Random()
            base_heat_phase = self.base_heat_phase
            if base_heat_phase is None:
                base_heat_phase = [
                    build_base_heat(self.hit_counts_phase[p], self.miss_counts_phase[p], self.board_size)
                    for p in range(4)
                ]

            eval_targets = 0
            if self.current_mask:
                eval_targets += 1
            if self.suggested_mask:
                eval_targets += 1
            eval_targets += max(0, self.random_layouts)
            total_jobs = eval_targets * self.games_per_layout * 2
            done = 0

            results: Dict[str, object] = {}
            if self.current_mask:
                res, done = self._eval_mask("current", self.current_mask, base_heat_phase, rng, total_jobs, done)
                if res:
                    results["current"] = res

            if self.suggested_mask:
                res, done = self._eval_mask("suggested", self.suggested_mask, base_heat_phase, rng, total_jobs, done)
                if res:
                    results["suggested"] = res

            if self.random_layouts > 0 and not self._cancelled:
                total_heat = 0.0
                total_seq = 0.0
                layouts_used = 0
                for _ in range(self.random_layouts):
                    if self._cancelled:
                        break
                    mask = self._random_layout_mask(rng)
                    if mask is None:
                        continue
                    layouts_used += 1
                    for _ in range(self.games_per_layout):
                        if self._cancelled:
                            break
                        sh, _ = simulate_enemy_game_phase(
                            mask,
                            base_heat_phase,
                            self.disp_counts,
                            "heat",
                            rng,
                            board_size=self.board_size,
                        )
                        total_heat += sh
                        done += 1
                        self.progress.emit(done, total_jobs)
                        if self._cancelled:
                            break
                        ss, _ = simulate_enemy_game_phase(
                            mask,
                            base_heat_phase,
                            self.disp_counts,
                            "seq",
                            rng,
                            board_size=self.board_size,
                        )
                        total_seq += ss
                        done += 1
                        self.progress.emit(done, total_jobs)
                if layouts_used > 0 and not self._cancelled:
                    games = layouts_used * self.games_per_layout
                    avg_heat = total_heat / games if games else 0.0
                    avg_seq = total_seq / games if games else 0.0
                    results["baseline_random"] = {
                        "label": "baseline_random",
                        "games": games,
                        "layouts": layouts_used,
                        "avg_heat": avg_heat,
                        "avg_seq": avg_seq,
                        "robust": min(avg_heat, avg_seq),
                    }

            self.finished.emit(results)
        except Exception:
            import traceback as _tb
            self.error.emit(_tb.format_exc())
