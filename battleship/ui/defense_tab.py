import random
from typing import Dict, List, Optional, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

from battleship.domain.board import cell_index
from battleship.domain.config import DISP_RADIUS, HAS_SHIP, NO_SHIP, NO_SHOT, SHOT_HIT, SHOT_MISS
from battleship.layouts.cache import LayoutRuntime
from battleship.persistence.layout_state import delete_layout_state, load_layout_state, save_layout_state
from battleship.sim.defense_sim import build_base_heat, recommend_layout_phase, simulate_enemy_game_phase
from battleship.ui.theme import Theme


class DefenseTab(QtWidgets.QWidget):
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

        self._build_ui()
        self.load_state()
        self.update_board_view()
        self.update_summary_labels()

    def _build_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(24)

        board_container = QtWidgets.QWidget()
        board_layout = QtWidgets.QGridLayout(board_container)
        board_layout.setSpacing(2)
        board_layout.setContentsMargins(24, 24, 24, 24)

        for c in range(self.board_size):
            lbl = QtWidgets.QLabel(chr(ord("A") + c))
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setStyleSheet(f"color: {Theme.TEXT_LABEL};")
            board_layout.addWidget(lbl, 0, c + 1)
        for r in range(self.board_size):
            lbl = QtWidgets.QLabel(str(r + 1))
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setStyleSheet(f"color: {Theme.TEXT_LABEL};")
            board_layout.addWidget(lbl, r + 1, 0)

        self.cell_buttons_def: List[List[QtWidgets.QPushButton]] = []
        for r in range(self.board_size):
            row = []
            for c in range(self.board_size):
                btn = QtWidgets.QPushButton("")
                btn.setFixedSize(48, 48)
                btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
                btn.clicked.connect(self._make_cell_handler(r, c))
                row.append(btn)
                board_layout.addWidget(btn, r + 1, c + 1)
            self.cell_buttons_def.append(row)

        main_layout.addWidget(board_container, stretch=0, alignment=QtCore.Qt.AlignCenter)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setSpacing(8)

        header = QtWidgets.QLabel("Tab 2: Opponent attacks you")
        f = header.font()
        f.setPointSize(f.pointSize() + 1)
        f.setBold(True)
        header.setFont(f)
        right_layout.addWidget(header)

        desc = QtWidgets.QLabel(
            "Paint your ships in 'Edit layout' mode.\n"
            "Switch to 'Record opponent shots' and click where they shoot.\n"
            "The app learns phase-specific heatmaps and shot sequences, and "
            "then suggests layouts that are robust to both behaviours."
        )
        desc.setWordWrap(True)
        right_layout.addWidget(desc)

        self.mode_tabs = QtWidgets.QTabWidget()
        self.place_tab = QtWidgets.QWidget()
        self.analyze_tab = QtWidgets.QWidget()
        self.mode_tabs.addTab(self.place_tab, "Place")
        self.mode_tabs.addTab(self.analyze_tab, "Analyze")
        self.mode_tabs.currentChanged.connect(self._on_mode_changed)
        right_layout.addWidget(self.mode_tabs)

        place_layout = QtWidgets.QVBoxLayout(self.place_tab)
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

        analyze_layout = QtWidgets.QVBoxLayout(self.analyze_tab)
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

        self.recommendation_label = QtWidgets.QLabel(
            "No layout suggestion yet.\nRecord opponent shots, then click 'Suggest layout'."
        )
        self.recommendation_label.setWordWrap(True)
        analyze_layout.addWidget(self.recommendation_label)

        right_layout.addStretch(1)
        main_layout.addWidget(right_panel, stretch=1)

    def save_state(self, path: Optional[str] = None):
        if path is None:
            path = self.STATE_PATH

        # history_events is List[Tuple[int,int,bool,int]]
        history_serializable = [
            [int(r), int(c), bool(was_hit), int(phase)]
            for (r, c, was_hit, phase) in self.history_events
        ]

        # last_shot_for_sequence is Optional[Tuple[int,int,int,bool]]
        if self.last_shot_for_sequence is None:
            last_seq = None
        else:
            lr, lc, phase, was_hit = self.last_shot_for_sequence
            last_seq = [int(lr), int(lc), int(phase), bool(was_hit)]

        state = {
            "layout_id": self.layout.layout_id,
            "layout_version": self.layout.layout_version,
            "layout_hash": self.layout.layout_hash,
            # long-term learning stats
            "hit_counts_phase": self.hit_counts_phase,
            "miss_counts_phase": self.miss_counts_phase,
            "disp_counts": self.disp_counts,

            # current game state
            "layout_board": self.layout_board,
            "shot_board": self.shot_board,
            "history_events": history_serializable,
            "last_shot_for_sequence": last_seq,
            "mode_tab": self.mode_tabs.currentIndex(),
            "eval_games_per_layout": int(self.eval_games_spin.value()),
            "eval_random_layouts": int(self.eval_random_spin.value()),
            "eval_results": self.eval_results,
        }
        save_layout_state(path, self.layout, state)

    def load_state(self, path: Optional[str] = None):
        if path is None:
            path = self.STATE_PATH
        data, _raw = load_layout_state(path, self.layout)
        if not data:
            return

        # --- long-term learning stats (backwards compatible) ---
        hc = data.get("hit_counts_phase")
        mc = data.get("miss_counts_phase")
        dc = data.get("disp_counts")
        if isinstance(hc, list) and isinstance(mc, list) and len(hc) == 4 and len(mc) == 4:
            self.hit_counts_phase = hc
            self.miss_counts_phase = mc
        if isinstance(dc, list) and len(dc) == 4:
            self.disp_counts = dc

        # --- current layout board (optional, for mid-game resume) ---
        lb = data.get("layout_board")
        if (
            isinstance(lb, list)
            and len(lb) == self.board_size
            and all(isinstance(row, list) and len(row) == self.board_size for row in lb)
        ):
            self.layout_board = lb

        # --- current shot board (optional, for mid-game resume) ---
        sb = data.get("shot_board")
        if (
            isinstance(sb, list)
            and len(sb) == self.board_size
            and all(isinstance(row, list) and len(row) == self.board_size for row in sb)
        ):
            self.shot_board = sb

        # --- shot history for correct undo behaviour ---
        he = data.get("history_events")
        new_history: List[Tuple[int, int, bool, int]] = []
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
                        new_history.append((r, c, bool(was_hit), phase))
        self.history_events = new_history

        # --- last shot for sequence model (optional) ---
        lss = data.get("last_shot_for_sequence")
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

        mode_tab = data.get("mode_tab")
        if isinstance(mode_tab, int) and 0 <= mode_tab < self.mode_tabs.count():
            self.mode_tabs.setCurrentIndex(mode_tab)

        eval_games = data.get("eval_games_per_layout")
        if isinstance(eval_games, int):
            self.eval_games_per_layout = eval_games
            self.eval_games_spin.setValue(eval_games)
        eval_random = data.get("eval_random_layouts")
        if isinstance(eval_random, int):
            self.eval_random_layouts = eval_random
            self.eval_random_spin.setValue(eval_random)

        eval_results = data.get("eval_results")
        if isinstance(eval_results, dict):
            self.eval_results = eval_results
            self._update_eval_results_label()

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
        else:
            # undo
            for i in range(len(self.history_events) - 1, -1, -1):
                er, ec, was_hit, phase = self.history_events[i]
                if er == r and ec == c:
                    self.history_events.pop(i)
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
        delete_layout_state(self.STATE_PATH, self.layout)

    def compute_suggested_layout(self):
        layout, mask, robust, avg_heat, avg_seq = recommend_layout_phase(
            self.hit_counts_phase,
            self.miss_counts_phase,
            self.disp_counts,
            self.placements,
            ship_ids=self.ship_ids,
            board_size=self.board_size,
            n_iter=250,
            sim_games_per_layout=10,
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

        self._eval_worker = DefenseEvalWorker(
            current_mask=current_mask,
            suggested_mask=suggested_mask,
            placements=self.placements,
            ship_ids=self.ship_ids,
            board_size=self.board_size,
            hit_counts_phase=self.hit_counts_phase,
            miss_counts_phase=self.miss_counts_phase,
            disp_counts=self.disp_counts,
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

    def update_board_view(self):
        show_heat = self.heatmap_checkbox.isChecked()

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
