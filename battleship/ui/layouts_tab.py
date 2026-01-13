import re
from typing import Dict, List, Optional, Tuple

from PyQt5 import QtCore, QtWidgets

from battleship.layouts.definition import LayoutDefinition, ShipSpec, normalize_shape_cells
from battleship.layouts.placements import generate_ship_placements
from battleship.layouts.validation import validate_layout
from battleship.persistence.layouts_store import (
    CUSTOM_LAYOUTS_PATH,
    is_custom_layout_id,
    load_custom_layouts,
    new_custom_layout_id,
    save_custom_layouts,
)
from battleship.ui.theme import Theme


def _parse_cells(text: str) -> Tuple[Tuple[int, int], ...]:
    tokens = re.split(r"[;\n\r\t ]+", text.strip())
    cells = []
    for tok in tokens:
        if not tok:
            continue
        cleaned = tok.strip().strip("()")
        if not cleaned:
            continue
        parts = cleaned.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid cell token: {tok}")
        r = int(parts[0].strip())
        c = int(parts[1].strip())
        cells.append((r, c))
    if not cells:
        return tuple()
    norm = normalize_shape_cells(cells)
    return tuple(dict.fromkeys(norm))


class ShipDialog(QtWidgets.QDialog):
    def __init__(self, ship: Optional[ShipSpec] = None, parent=None):
        super().__init__(parent)
        self.ship = ship
        self.setWindowTitle("Ship Editor")
        self.resize(420, 360)
        self._build_ui()
        if ship is not None:
            self._load_ship(ship)

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        self.id_input = QtWidgets.QLineEdit()
        form.addRow("Instance ID", self.id_input)

        self.name_input = QtWidgets.QLineEdit()
        form.addRow("Display name", self.name_input)

        self.kind_combo = QtWidgets.QComboBox()
        self.kind_combo.addItems(["line", "shape"])
        self.kind_combo.currentIndexChanged.connect(self._on_kind_changed)
        form.addRow("Kind", self.kind_combo)

        self.length_spin = QtWidgets.QSpinBox()
        self.length_spin.setRange(1, 20)
        form.addRow("Length", self.length_spin)

        self.cells_edit = QtWidgets.QPlainTextEdit()
        self.cells_edit.setPlaceholderText("Example:\n0,0\n0,1\n1,0")
        form.addRow("Shape cells", self.cells_edit)

        self.rotate_cb = QtWidgets.QCheckBox("Allow rotations")
        self.rotate_cb.setChecked(True)
        form.addRow("", self.rotate_cb)

        layout.addLayout(form)

        self.error_label = QtWidgets.QLabel("")
        self.error_label.setStyleSheet("color: #ff6b6b;")
        self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        self._on_kind_changed()

    def _on_kind_changed(self):
        is_line = self.kind_combo.currentText() == "line"
        self.length_spin.setEnabled(is_line)
        self.cells_edit.setEnabled(not is_line)

    def _load_ship(self, ship: ShipSpec):
        self.id_input.setText(ship.instance_id)
        self.name_input.setText(ship.name or "")
        self.kind_combo.setCurrentText(ship.kind)
        self.length_spin.setValue(int(ship.length or 1))
        if ship.cells:
            cells = "\n".join(f"{r},{c}" for r, c in ship.cells)
            self.cells_edit.setPlainText(cells)
        self.rotate_cb.setChecked(bool(ship.allow_rotations))
        self._on_kind_changed()

    def _validate(self) -> Optional[ShipSpec]:
        instance_id = self.id_input.text().strip()
        if not instance_id:
            self.error_label.setText("Instance ID is required.")
            return None

        kind = self.kind_combo.currentText()
        name = self.name_input.text().strip() or None
        allow_rotations = self.rotate_cb.isChecked()

        if kind == "line":
            length = int(self.length_spin.value())
            return ShipSpec(
                instance_id=instance_id,
                kind="line",
                length=length,
                allow_rotations=allow_rotations,
                name=name,
            )

        try:
            cells = _parse_cells(self.cells_edit.toPlainText())
        except Exception as exc:
            self.error_label.setText(str(exc))
            return None
        if not cells:
            self.error_label.setText("Shape ships must define at least one cell.")
            return None
        return ShipSpec(
            instance_id=instance_id,
            kind="shape",
            cells=cells,
            allow_rotations=allow_rotations,
            name=name,
        )

    def accept(self):
        spec = self._validate()
        if spec is None:
            return
        self.ship = spec
        super().accept()


class LayoutsTab(QtWidgets.QWidget):
    def __init__(
        self,
        get_active_layout,
        on_layouts_updated,
        on_layout_selected,
        parent=None,
    ):
        super().__init__(parent)
        self.get_active_layout = get_active_layout
        self.on_layouts_updated = on_layouts_updated
        self.on_layout_selected = on_layout_selected

        self.custom_layouts: List[LayoutDefinition] = load_custom_layouts(CUSTOM_LAYOUTS_PATH)
        self.current_layout_id: Optional[str] = None
        self.ship_specs: List[ShipSpec] = []

        self._build_ui()
        self._refresh_layout_list(select_first=True)

    def _build_ui(self):
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.addWidget(QtWidgets.QLabel("Custom layouts"))

        self.layout_list = QtWidgets.QListWidget()
        self.layout_list.currentRowChanged.connect(self._on_layout_selected)
        left_layout.addWidget(self.layout_list)

        btn_row = QtWidgets.QHBoxLayout()
        self.new_btn = QtWidgets.QPushButton("New")
        self.new_btn.clicked.connect(self.new_layout)
        btn_row.addWidget(self.new_btn)

        self.dup_btn = QtWidgets.QPushButton("Duplicate")
        self.dup_btn.clicked.connect(self.duplicate_layout)
        btn_row.addWidget(self.dup_btn)
        left_layout.addLayout(btn_row)

        btn_row2 = QtWidgets.QHBoxLayout()
        self.copy_active_btn = QtWidgets.QPushButton("Copy active")
        self.copy_active_btn.clicked.connect(self.copy_active_layout)
        btn_row2.addWidget(self.copy_active_btn)

        self.delete_btn = QtWidgets.QPushButton("Delete")
        self.delete_btn.clicked.connect(self.delete_layout)
        btn_row2.addWidget(self.delete_btn)
        left_layout.addLayout(btn_row2)

        layout.addWidget(left_panel, stretch=0)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)

        form = QtWidgets.QFormLayout()
        self.name_input = QtWidgets.QLineEdit()
        form.addRow("Name", self.name_input)
        self.board_spin = QtWidgets.QSpinBox()
        self.board_spin.setRange(4, 20)
        form.addRow("Board size", self.board_spin)
        self.touch_cb = QtWidgets.QCheckBox("Allow ships to touch")
        self.touch_cb.setChecked(True)
        form.addRow("", self.touch_cb)
        right_layout.addLayout(form)

        ship_group = QtWidgets.QGroupBox("Ships")
        ship_layout = QtWidgets.QVBoxLayout(ship_group)
        self.ship_table = QtWidgets.QTableWidget()
        self.ship_table.setColumnCount(5)
        self.ship_table.setHorizontalHeaderLabels(["ID", "Name", "Kind", "Size/Cells", "Rotations"])
        self.ship_table.horizontalHeader().setStretchLastSection(True)
        self.ship_table.verticalHeader().setVisible(False)
        self.ship_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.ship_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        ship_layout.addWidget(self.ship_table)

        ship_btns = QtWidgets.QHBoxLayout()
        self.add_line_btn = QtWidgets.QPushButton("Add line")
        self.add_line_btn.clicked.connect(lambda: self._add_ship("line"))
        ship_btns.addWidget(self.add_line_btn)
        self.add_shape_btn = QtWidgets.QPushButton("Add shape")
        self.add_shape_btn.clicked.connect(lambda: self._add_ship("shape"))
        ship_btns.addWidget(self.add_shape_btn)
        self.edit_ship_btn = QtWidgets.QPushButton("Edit")
        self.edit_ship_btn.clicked.connect(self._edit_ship)
        ship_btns.addWidget(self.edit_ship_btn)
        self.remove_ship_btn = QtWidgets.QPushButton("Remove")
        self.remove_ship_btn.clicked.connect(self._remove_ship)
        ship_btns.addWidget(self.remove_ship_btn)
        ship_layout.addLayout(ship_btns)
        right_layout.addWidget(ship_group)

        action_row = QtWidgets.QHBoxLayout()
        self.save_btn = QtWidgets.QPushButton("Save layout")
        self.save_btn.clicked.connect(self.save_layout)
        action_row.addWidget(self.save_btn)
        self.activate_btn = QtWidgets.QPushButton("Set as active")
        self.activate_btn.clicked.connect(self.set_active_layout)
        action_row.addWidget(self.activate_btn)
        action_row.addStretch(1)
        right_layout.addLayout(action_row)

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setStyleSheet(f"color: {Theme.TEXT_MUTED};")
        self.status_label.setWordWrap(True)
        right_layout.addWidget(self.status_label)

        layout.addWidget(right_panel, stretch=1)

    def _refresh_layout_list(self, select_first: bool = False, select_id: Optional[str] = None):
        self.custom_layouts = load_custom_layouts(CUSTOM_LAYOUTS_PATH)
        self.layout_list.blockSignals(True)
        self.layout_list.clear()
        for layout in self.custom_layouts:
            label = f"{layout.name} ({layout.board_size}x{layout.board_size})"
            self.layout_list.addItem(label)
        self.layout_list.blockSignals(False)

        if select_id:
            for i, layout in enumerate(self.custom_layouts):
                if layout.layout_id == select_id:
                    self.layout_list.setCurrentRow(i)
                    return
        if select_first and self.custom_layouts:
            self.layout_list.setCurrentRow(0)
        elif not self.custom_layouts:
            self.current_layout_id = None
            self.ship_specs = []
            self._clear_editor()

    def _clear_editor(self):
        self.name_input.setText("")
        self.board_spin.setValue(10)
        self.touch_cb.setChecked(True)
        self.ship_specs = []
        self._refresh_ship_table()
        self.status_label.setText("Create a new layout or copy the active one.")

    def _refresh_ship_table(self):
        self.ship_table.setRowCount(len(self.ship_specs))
        for row, spec in enumerate(self.ship_specs):
            size = ""
            if spec.kind == "line":
                size = str(int(spec.length or 0))
            else:
                size = ", ".join(f"{r},{c}" for r, c in (spec.cells or []))
            vals = [
                spec.instance_id,
                spec.name or "",
                spec.kind,
                size,
                "yes" if spec.allow_rotations else "no",
            ]
            for col, val in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(val)
                self.ship_table.setItem(row, col, it)

    def _on_layout_selected(self, row: int):
        if row < 0 or row >= len(self.custom_layouts):
            return
        layout = self.custom_layouts[row]
        self.current_layout_id = layout.layout_id
        self.name_input.setText(layout.name)
        self.board_spin.setValue(layout.board_size)
        self.touch_cb.setChecked(layout.allow_touching)
        self.ship_specs = list(layout.ships)
        self._refresh_ship_table()
        self.status_label.setText(f"Editing {layout.name} (v{layout.layout_version})")

    def _add_ship(self, kind: str):
        dialog = ShipDialog(parent=self)
        dialog.kind_combo.setCurrentText(kind)
        dialog._on_kind_changed()
        if dialog.exec_() == QtWidgets.QDialog.Accepted and dialog.ship is not None:
            self.ship_specs.append(dialog.ship)
            self._refresh_ship_table()

    def _edit_ship(self):
        row = self.ship_table.currentRow()
        if row < 0 or row >= len(self.ship_specs):
            return
        dialog = ShipDialog(self.ship_specs[row], parent=self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted and dialog.ship is not None:
            self.ship_specs[row] = dialog.ship
            self._refresh_ship_table()

    def _remove_ship(self):
        row = self.ship_table.currentRow()
        if row < 0 or row >= len(self.ship_specs):
            return
        self.ship_specs.pop(row)
        self._refresh_ship_table()

    def new_layout(self):
        self.current_layout_id = None
        self.ship_specs = []
        self._clear_editor()

    def duplicate_layout(self):
        row = self.layout_list.currentRow()
        if row < 0 or row >= len(self.custom_layouts):
            return
        layout = self.custom_layouts[row]
        self.current_layout_id = None
        self.name_input.setText(f"{layout.name} Copy")
        self.board_spin.setValue(layout.board_size)
        self.touch_cb.setChecked(layout.allow_touching)
        self.ship_specs = list(layout.ships)
        self._refresh_ship_table()
        self.status_label.setText("Duplicated layout. Save to create a new custom layout.")

    def copy_active_layout(self):
        active = self.get_active_layout() if self.get_active_layout else None
        if active is None:
            return
        self.current_layout_id = None
        self.name_input.setText(f"{active.name} Custom")
        self.board_spin.setValue(active.board_size)
        self.touch_cb.setChecked(active.allow_touching)
        self.ship_specs = list(active.ships)
        self._refresh_ship_table()
        self.status_label.setText("Copied active layout. Save to create a new custom layout.")

    def delete_layout(self):
        row = self.layout_list.currentRow()
        if row < 0 or row >= len(self.custom_layouts):
            return
        layout = self.custom_layouts[row]
        if not is_custom_layout_id(layout.layout_id):
            QtWidgets.QMessageBox.information(self, "Cannot delete", "Built-in layouts cannot be deleted.")
            return
        confirm = QtWidgets.QMessageBox.question(
            self,
            "Delete layout",
            f"Delete '{layout.name}'?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if confirm != QtWidgets.QMessageBox.Yes:
            return
        self.custom_layouts.pop(row)
        save_custom_layouts(self.custom_layouts, CUSTOM_LAYOUTS_PATH)
        self._refresh_layout_list(select_first=True)
        if self.on_layouts_updated:
            self.on_layouts_updated()

    def _validate_layout(self, layout: LayoutDefinition) -> List[str]:
        errors = validate_layout(layout)
        for spec in layout.ships:
            placements = generate_ship_placements(spec, layout.board_size, layout.allow_touching)
            if not placements:
                errors.append(f"Ship {spec.instance_id} has no valid placements.")
        total_cells = sum(
            int(spec.length or 0) if spec.kind == "line" else len(spec.cells or [])
            for spec in layout.ships
        )
        if total_cells > layout.board_size * layout.board_size:
            errors.append("Total ship cells exceed board size.")
        return errors

    def save_layout(self):
        name = self.name_input.text().strip()
        if not name:
            QtWidgets.QMessageBox.information(self, "Missing name", "Layout name is required.")
            return
        board_size = int(self.board_spin.value())
        allow_touching = self.touch_cb.isChecked()

        ships = tuple(self.ship_specs)
        layout_id = self.current_layout_id or new_custom_layout_id()
        layout_version = 1
        old_layout = None
        for layout in self.custom_layouts:
            if layout.layout_id == layout_id:
                old_layout = layout
                layout_version = layout.layout_version
                break

        candidate = LayoutDefinition(
            layout_id=layout_id,
            name=name,
            board_size=board_size,
            ships=ships,
            allow_touching=allow_touching,
            layout_version=layout_version,
        )

        errors = self._validate_layout(candidate)
        if errors:
            QtWidgets.QMessageBox.critical(self, "Layout errors", "\n".join(errors))
            return

        if old_layout is not None and old_layout.layout_hash != candidate.layout_hash:
            candidate = LayoutDefinition(
                layout_id=layout_id,
                name=name,
                board_size=board_size,
                ships=ships,
                allow_touching=allow_touching,
                layout_version=layout_version + 1,
            )

        updated = False
        new_layouts: List[LayoutDefinition] = []
        for layout in self.custom_layouts:
            if layout.layout_id == candidate.layout_id:
                new_layouts.append(candidate)
                updated = True
            else:
                new_layouts.append(layout)
        if not updated:
            new_layouts.append(candidate)

        save_custom_layouts(new_layouts, CUSTOM_LAYOUTS_PATH)
        self.current_layout_id = candidate.layout_id
        self.custom_layouts = new_layouts
        self._refresh_layout_list(select_id=candidate.layout_id)
        self.status_label.setText(f"Saved layout {candidate.name} (v{candidate.layout_version}).")
        if self.on_layouts_updated:
            self.on_layouts_updated()

    def set_active_layout(self):
        if self.current_layout_id is None:
            return
        for layout in self.custom_layouts:
            if layout.layout_id == self.current_layout_id:
                if self.on_layout_selected:
                    self.on_layout_selected(layout)
                return
