import os
import sys

from PyQt5 import QtCore, QtGui, QtWidgets

from battleship.domain.config import HIT, MISS, SHOT_HIT, SHOT_MISS
from battleship.layouts import DEFAULT_PLACEMENT_CACHE, LayoutDefinition, builtin_layouts, classic_layout
from battleship.persistence.app_state import load_match_state, load_selected_layout, save_match_state, save_selected_layout
from battleship.persistence.layouts_store import is_custom_layout_id, load_custom_layouts
from battleship.persistence.stats import StatsTracker
from battleship.ui.attack_tab import AttackTab
from battleship.ui.defense_tab import DefenseTab
from battleship.ui.layouts_tab import LayoutsTab
from battleship.ui.model_stats_tab import ModelStatsTab
from battleship.ui.check_style import CheckboxRadioStyle
from battleship.ui.theme import Theme
from battleship.utils import debug


def apply_dark_palette(app: QtWidgets.QApplication):
    """Apply a consistent dark theme using the Theme color palette."""
    QtWidgets.QApplication.setStyle("Fusion")
    palette = QtGui.QPalette()

    # Core backgrounds
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(Theme.BG_DARK))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(Theme.BG_DARK))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(Theme.BG_PANEL))

    # Text colors
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(Theme.TEXT_MAIN))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor(Theme.TEXT_MAIN))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(Theme.TEXT_MAIN))

    # Buttons
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(Theme.BG_BUTTON))

    # Links & selection
    palette.setColor(QtGui.QPalette.Link, QtGui.QColor(Theme.LINK))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(Theme.HIGHLIGHT))
    palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)

    # Tooltips / bright text
    palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)

    app.setPalette(palette)
    app.setStyle(CheckboxRadioStyle(app.style()))


class MainWindow(QtWidgets.QMainWindow):
    APP_STATE_PATH = "battleship_app_state.json"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Battleship Solver – 2-ply Attack & Learned Defense")
        self.resize(1250, 700)

        self.stats = StatsTracker()
        self.custom_layouts = load_custom_layouts()
        self.layouts = list(builtin_layouts()) + list(self.custom_layouts)
        self.layout_definition = self._resolve_layout_selection(self.layouts)
        self.layout_runtime = DEFAULT_PLACEMENT_CACHE.get(self.layout_definition)
        self.match_state = load_match_state(self.APP_STATE_PATH)
        self._syncing_opponent = False

        central = QtWidgets.QWidget()
        root_layout = QtWidgets.QVBoxLayout(central)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(8)

        selector_row = QtWidgets.QHBoxLayout()
        selector_row.setContentsMargins(0, 0, 0, 0)
        selector_label = QtWidgets.QLabel("Layout:")
        self.layout_combo = QtWidgets.QComboBox()
        for layout in self.layouts:
            self.layout_combo.addItem(self._layout_label(layout), userData=layout)
        selector_row.addWidget(selector_label)
        selector_row.addWidget(self.layout_combo)
        selector_row.addStretch(1)
        root_layout.addLayout(selector_row)

        self._build_match_panel(root_layout)

        self.tabs = QtWidgets.QTabWidget()
        root_layout.addWidget(self.tabs, stretch=1)
        self.setCentralWidget(central)

        index = self._layout_index(self.layout_definition)
        if index >= 0:
            self.layout_combo.setCurrentIndex(index)
        self.layout_combo.currentIndexChanged.connect(self._on_layout_changed)

        self._build_tabs()
        self._connect_live_match()

    # We need to hook up the signal so AttackTab can see DefenseTab
    # Override the recompute connection in AttackTab
    def _connect_tabs(self):
        # When Attack tab recomputes, pass Defense tab
        self.attack_tab.recompute_button.clicked.disconnect()
        self.attack_tab.recompute_button.clicked.connect(
            lambda: self.attack_tab.recompute(self.defense_tab)
        )
        # Also trigger it on cell clicks?
        # A simpler way is to just set a reference:
        self.attack_tab.linked_defense_tab = self.defense_tab

    def _layout_label(self, layout: LayoutDefinition) -> str:
        ship_count = len(layout.ships)
        tag = "Custom" if is_custom_layout_id(layout.layout_id) else "Built-in"
        return f"{layout.name} ({layout.board_size}x{layout.board_size}, {ship_count} ships) [{tag}]"

    def _layout_index(self, layout: LayoutDefinition) -> int:
        for i, candidate in enumerate(self.layouts):
            if (
                candidate.layout_id == layout.layout_id
                and candidate.layout_version == layout.layout_version
                and candidate.layout_hash == layout.layout_hash
            ):
                return i
        for i, candidate in enumerate(self.layouts):
            if candidate.layout_id == layout.layout_id:
                return i
        return -1

    def _resolve_layout_selection(self, layouts):
        selected = load_selected_layout(self.APP_STATE_PATH)
        if selected:
            for layout in layouts:
                if (
                    layout.layout_id == selected.get("layout_id")
                    and layout.layout_version == selected.get("layout_version")
                    and layout.layout_hash == selected.get("layout_hash")
                ):
                    return layout
            for layout in layouts:
                if layout.layout_id == selected.get("layout_id"):
                    return layout
        classic_id = classic_layout().layout_id
        for layout in layouts:
            if layout.layout_id == classic_id:
                return layout
        return layouts[0]

    def _build_tabs(self):
        self.attack_tab = AttackTab(self.stats, self.layout_runtime)
        self.defense_tab = DefenseTab(self.layout_runtime)
        self.model_tab = ModelStatsTab(self.layout_runtime)
        self.layouts_tab = LayoutsTab(
            get_active_layout=lambda: self.layout_definition,
            on_layouts_updated=self.refresh_layouts,
            on_layout_selected=self._set_layout,
        )
        self.attack_tab.linked_defense_tab = self.defense_tab

        self.tabs.addTab(self.attack_tab, "You attack")
        self.tabs.addTab(self.defense_tab, "Opponent attacks you")
        self.tabs.addTab(self.model_tab, "Model stats")
        self.tabs.addTab(self.layouts_tab, "Layouts")

    def _build_match_panel(self, root_layout: QtWidgets.QVBoxLayout) -> None:
        self.match_group = QtWidgets.QGroupBox("Live match")
        match_layout = QtWidgets.QGridLayout(self.match_group)
        match_layout.setContentsMargins(12, 10, 12, 10)
        match_layout.setHorizontalSpacing(12)
        match_layout.setVerticalSpacing(6)

        opponent_label = QtWidgets.QLabel("Opponent:")
        self.match_opponent_combo = QtWidgets.QComboBox()
        self.match_opponent_combo.setMinimumWidth(220)
        self.match_opponent_combo.currentIndexChanged.connect(self._on_match_opponent_changed)

        self.match_record_label = QtWidgets.QLabel("Record: 0-0")
        self.match_record_label.setStyleSheet(f"color: {Theme.TEXT_LABEL};")

        self.match_win_label = QtWidgets.QLabel("Live win%: N/A")
        self.match_win_label.setStyleSheet(f"color: {Theme.HIGHLIGHT}; font-weight: bold;")

        self.match_shots_label = QtWidgets.QLabel("Shots: You 0 | Opp 0")
        self.match_turn_label = QtWidgets.QLabel("Next shot: You")

        self.match_new_game_btn = QtWidgets.QPushButton("New game (clear both)")
        self.match_new_game_btn.clicked.connect(self._clear_match_boards)
        self.match_record_win_btn = QtWidgets.QPushButton("Record win")
        self.match_record_win_btn.clicked.connect(lambda: self._record_match_result(True))
        self.match_record_loss_btn = QtWidgets.QPushButton("Record loss")
        self.match_record_loss_btn.clicked.connect(lambda: self._record_match_result(False))

        match_layout.addWidget(opponent_label, 0, 0)
        match_layout.addWidget(self.match_opponent_combo, 0, 1, 1, 2)
        match_layout.addWidget(self.match_record_label, 0, 3)

        match_layout.addWidget(self.match_win_label, 1, 0, 1, 2)
        match_layout.addWidget(self.match_shots_label, 1, 2)
        match_layout.addWidget(self.match_turn_label, 1, 3)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(8)
        btn_layout.addWidget(self.match_new_game_btn)
        btn_layout.addWidget(self.match_record_win_btn)
        btn_layout.addWidget(self.match_record_loss_btn)
        btn_layout.addStretch(1)
        match_layout.addLayout(btn_layout, 0, 4, 2, 1)

        root_layout.addWidget(self.match_group)

    def _connect_live_match(self) -> None:
        self.attack_tab.opponent_changed.connect(self._on_tab_opponent_changed)
        self.defense_tab.opponent_changed.connect(self._on_tab_opponent_changed)
        self.attack_tab.state_updated.connect(self._update_match_status)
        self.defense_tab.state_updated.connect(self._update_match_status)
        self.attack_tab.game_result.connect(self._on_attack_game_result)

        self._refresh_match_opponents()
        active = self.match_state.get("active_opponent") if isinstance(self.match_state, dict) else None
        if isinstance(active, str) and active.strip():
            self._set_active_match_opponent(active.strip(), persist=False)
        else:
            self._set_active_match_opponent(self.attack_tab._current_opponent_name(), persist=False)
        self._update_match_status()

    def _refresh_match_opponents(self) -> None:
        names = []
        names.extend(self.attack_tab.get_opponent_names())
        names.extend(self.defense_tab.get_opponent_names())
        # Preserve insertion order, case-insensitive uniqueness.
        seen = set()
        unique = []
        for name in names:
            key = name.strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(name.strip())
        if not unique:
            unique = ["Opponent"]

        current = self.match_opponent_combo.currentText().strip()
        active = self.match_state.get("active_opponent") if isinstance(self.match_state, dict) else None
        target = (active or current).strip() if isinstance(active, str) or current else ""
        if target and target.lower() not in {n.lower() for n in unique}:
            unique.append(target)

        self.match_opponent_combo.blockSignals(True)
        self.match_opponent_combo.clear()
        for name in unique:
            self.match_opponent_combo.addItem(name)
        if target:
            idx = next((i for i, n in enumerate(unique) if n.lower() == target.lower()), 0)
            self.match_opponent_combo.setCurrentIndex(idx)
        self.match_opponent_combo.blockSignals(False)

    def _set_active_match_opponent(self, name: str, persist: bool = True) -> None:
        if not name:
            return
        if self._syncing_opponent:
            return
        self._syncing_opponent = True
        try:
            self.match_state["active_opponent"] = name
            self.attack_tab.set_active_opponent_by_name(name)
            self.defense_tab.set_active_opponent_by_name(name)
            self._refresh_match_opponents()
            if persist:
                save_match_state(self.APP_STATE_PATH, self.match_state)
        finally:
            self._syncing_opponent = False

    def _on_match_opponent_changed(self, index: int) -> None:
        if self._syncing_opponent:
            return
        if index < 0:
            return
        name = self.match_opponent_combo.itemText(index).strip()
        if name:
            self._set_active_match_opponent(name, persist=True)
            self._update_match_status()

    def _on_tab_opponent_changed(self, name: str) -> None:
        if self._syncing_opponent:
            return
        if not name:
            return
        self._set_active_match_opponent(name, persist=True)
        self._update_match_status()

    def _record_match_result(self, win: bool) -> None:
        if hasattr(self, "attack_tab"):
            self.attack_tab._record_game_result(win)

    def _on_attack_game_result(self, win: bool) -> None:
        name = self.match_opponent_combo.currentText().strip() or "Opponent"
        records = self.match_state.setdefault("records", {})
        record = records.setdefault(name, {"wins": 0, "losses": 0})
        if win:
            record["wins"] = int(record.get("wins", 0)) + 1
        else:
            record["losses"] = int(record.get("losses", 0)) + 1
        records[name] = record
        self.match_state["records"] = records
        save_match_state(self.APP_STATE_PATH, self.match_state)
        if hasattr(self, "defense_tab"):
            self.defense_tab.clear_shots()
        self._update_match_status()

    def _clear_match_boards(self) -> None:
        if hasattr(self, "attack_tab"):
            self.attack_tab.clear_board()
        if hasattr(self, "defense_tab"):
            self.defense_tab.clear_shots()
        self._update_match_status()

    def _update_match_status(self) -> None:
        self._refresh_match_opponents()
        name = self.match_opponent_combo.currentText().strip() or "Opponent"
        records = {}
        if isinstance(self.match_state, dict):
            records = self.match_state.get("records", {}) or {}
        record = records.get(name, {"wins": 0, "losses": 0})
        wins = int(record.get("wins", 0))
        losses = int(record.get("losses", 0))
        total = wins + losses
        if total > 0:
            rate = (wins / total) * 100
            self.match_record_label.setText(f"Record: {wins}-{losses} ({rate:.1f}%)")
        else:
            self.match_record_label.setText("Record: 0-0")

        if hasattr(self, "attack_tab") and hasattr(self, "defense_tab"):
            result = self.attack_tab.compute_win_prediction(self.defense_tab)
        else:
            result = None
        if result is None:
            self.match_win_label.setText("Live win%: N/A")
        else:
            win_pct, my_total, opp_avg = result
            self.match_win_label.setText(
                f"Live win%: {win_pct:.1f}% (You: ~{my_total:.1f} vs Opp: ~{opp_avg:.1f})"
            )

        my_shots = 0
        opp_shots = 0
        if hasattr(self, "attack_tab"):
            my_shots = sum(row.count(HIT) + row.count(MISS) for row in self.attack_tab.board)
        if hasattr(self, "defense_tab"):
            opp_shots = sum(row.count(SHOT_HIT) + row.count(SHOT_MISS) for row in self.defense_tab.shot_board)
        self.match_shots_label.setText(f"Shots: You {my_shots} | Opp {opp_shots}")
        next_turn = "You" if my_shots <= opp_shots else "Opponent"
        self.match_turn_label.setText(f"Next shot: {next_turn}")

    def _save_tabs_state(self):
        try:
            self.attack_tab.save_state()
        except Exception as e:
            print("Error saving attack state:", e)

        try:
            self.defense_tab.save_state()
        except Exception as e:
            print("Error saving defense state:", e)

        try:
            self.model_tab.save_state()
        except Exception as e:
            print("Error saving model stats:", e)

    def _set_layout(self, layout: LayoutDefinition, persist: bool = True):
        if (
            layout.layout_id == self.layout_definition.layout_id
            and layout.layout_version == self.layout_definition.layout_version
            and layout.layout_hash == self.layout_definition.layout_hash
        ):
            return

        self._save_tabs_state()
        self.layout_definition = layout
        self.layout_runtime = DEFAULT_PLACEMENT_CACHE.get(layout)

        while self.tabs.count():
            widget = self.tabs.widget(0)
            self.tabs.removeTab(0)
            if widget is not None:
                widget.deleteLater()
        self._build_tabs()
        self._connect_live_match()

        if persist:
            save_selected_layout(self.APP_STATE_PATH, layout)

    def _on_layout_changed(self, index: int):
        layout = self.layout_combo.itemData(index)
        if isinstance(layout, LayoutDefinition):
            self._set_layout(layout, persist=True)

    def refresh_layouts(self):
        current_layout = self.layout_definition
        self.custom_layouts = load_custom_layouts()
        self.layouts = list(builtin_layouts()) + list(self.custom_layouts)
        self.layout_combo.blockSignals(True)
        self.layout_combo.clear()
        for layout in self.layouts:
            self.layout_combo.addItem(self._layout_label(layout), userData=layout)
        idx = self._layout_index(current_layout)
        if idx < 0 and self.layouts:
            idx = 0
        if idx >= 0:
            current_layout = self.layouts[idx]
            self.layout_combo.setCurrentIndex(idx)
        self.layout_combo.blockSignals(False)
        if (
            current_layout.layout_id != self.layout_definition.layout_id
            or current_layout.layout_version != self.layout_definition.layout_version
            or current_layout.layout_hash != self.layout_definition.layout_hash
        ):
            self._set_layout(current_layout, persist=True)

    def closeEvent(self, event: QtGui.QCloseEvent):
        self._save_tabs_state()

        try:
            save_selected_layout(self.APP_STATE_PATH, self.layout_definition)
        except Exception as e:
            print("Error saving app state:", e)

        try:
            self.stats.save()
        except Exception as e:
            print("Error saving stats:", e)

        event.accept()


def main():
    # Enable debug via flag or env var (BATTLESHIP_DEBUG=1)
    try:
        argv = list(sys.argv)
        if "--debug" in argv:
            debug.DEBUG_ENABLED = True
            argv.remove("--debug")
        if os.environ.get("BATTLESHIP_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}:
            debug.DEBUG_ENABLED = True
    except Exception:
        argv = sys.argv

    app = QtWidgets.QApplication(argv)
    apply_dark_palette(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
