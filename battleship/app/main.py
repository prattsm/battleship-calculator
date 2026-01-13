import os
import sys

from PyQt5 import QtCore, QtGui, QtWidgets

from battleship.layouts import DEFAULT_PLACEMENT_CACHE, LayoutDefinition, builtin_layouts, classic_layout
from battleship.persistence.app_state import load_selected_layout, save_selected_layout
from battleship.persistence.layouts_store import is_custom_layout_id, load_custom_layouts
from battleship.persistence.stats import StatsTracker
from battleship.ui.attack_tab import AttackTab
from battleship.ui.defense_tab import DefenseTab
from battleship.ui.layouts_tab import LayoutsTab
from battleship.ui.model_stats_tab import ModelStatsTab
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

        self.tabs = QtWidgets.QTabWidget()
        root_layout.addWidget(self.tabs, stretch=1)
        self.setCentralWidget(central)

        index = self._layout_index(self.layout_definition)
        if index >= 0:
            self.layout_combo.setCurrentIndex(index)
        self.layout_combo.currentIndexChanged.connect(self._on_layout_changed)

        self._build_tabs()

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
