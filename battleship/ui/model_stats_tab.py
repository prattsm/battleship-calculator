import itertools
import json
import math
import os
import random
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

from battleship.domain.config import PARAM_SPECS
from battleship.layouts.cache import LayoutRuntime
from battleship.persistence.layout_state import find_layout_versions, load_layout_state, save_layout_state
from battleship.sim.attack_sim import SimProfiler, simulate_model_game, simulate_model_game_with_phases
from battleship.strategies.registry import model_defs
from battleship.domain.phase import PHASE_ENDGAME, PHASE_HUNT, PHASE_TARGET
from battleship.ui.theme import Theme
from battleship.utils.debug import debug_event


PHASES = [PHASE_HUNT, PHASE_TARGET, PHASE_ENDGAME]


def _sim_profile_enabled() -> bool:
    raw = os.getenv("SIM_PROFILE")
    if raw is None:
        return False
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _hist_percentile(hist: List[int], pct: float) -> int:
    total = sum(hist)
    if total <= 0:
        return 0
    threshold = pct * total
    running = 0
    for idx, count in enumerate(hist):
        running += int(count)
        if running >= threshold:
            return idx
    return max(0, len(hist) - 1)


def _compute_basic_stats(stats: Dict[str, object]) -> Dict[str, float]:
    games = int(stats.get("total_games", 0))
    total = float(stats.get("total_shots", 0.0))
    sq = float(stats.get("sum_sq_shots", 0.0))
    hist = stats.get("hist")
    if not isinstance(hist, list):
        hist = []
    mean = total / games if games > 0 else 0.0
    var = max(0.0, (sq / games) - mean * mean) if games > 0 else 0.0
    std = math.sqrt(var)
    median = _hist_percentile(hist, 0.50) if games > 0 else 0
    p90 = _hist_percentile(hist, 0.90) if games > 0 else 0
    p95 = _hist_percentile(hist, 0.95) if games > 0 else 0
    return {
        "games": games,
        "mean": mean,
        "std": std,
        "median": float(median),
        "p90": float(p90),
        "p95": float(p95),
    }


def _compute_phase_avg(stats: Dict[str, object], phase: str) -> Optional[float]:
    phase_block = stats.get("phase", {})
    if not isinstance(phase_block, dict):
        return None
    entry = phase_block.get(phase)
    if not isinstance(entry, dict):
        return None
    games = int(entry.get("total_games", 0))
    if games <= 0:
        return None
    total = float(entry.get("total_shots", 0.0))
    return total / games


def _set_tooltip_palette(app: QtWidgets.QApplication) -> None:
    palette = app.palette()
    palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(Theme.BG_PANEL))
    palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(Theme.TEXT_MAIN))
    app.setPalette(palette)


class ModelStatsTab(QtWidgets.QWidget):
    STATE_PATH = "battleship_model_stats.json"

    def __init__(self, layout_runtime: LayoutRuntime, parent=None):
        super().__init__(parent)
        self.layout_runtime = layout_runtime
        self.layout = layout_runtime.definition
        self.board_size = self.layout.board_size
        self.total_cells = self.board_size * self.board_size
        self.ship_ids = list(self.layout.ship_ids())
        self.placements = layout_runtime.placements

        # Strategy definitions
        self._base_model_defs = [dict(md) for md in model_defs()]
        self.model_defs = [dict(md) for md in self._base_model_defs]
        self.model_overrides: Dict[str, Dict[str, str]] = {}

        self.model_stats: Dict[str, Dict[str, object]] = {}
        self.param_sweeps: Dict[str, List[Dict[str, object]]] = {}
        self._ensure_all_models()
        self.best_model_overall: Optional[str] = None
        self.best_model_by_phase: Dict[str, str] = {}

        self._sim_thread: Optional[QtCore.QThread] = None
        self._sim_worker: Optional[SimulationWorker] = None
        self._last_checkpoint_save = 0.0
        self._last_persist_log: Dict[str, float] = {}
        self._persist_log_interval = 1.0
        self._stats_lock = QtCore.QMutex()
        self._active_sim_keys: List[str] = []

        self._build_ui()
        self.load_state()
        self._apply_model_overrides()
        self._refresh_model_combo()
        self.refresh_table()
        self.update_summary_label()
        try:
            app = QtWidgets.QApplication.instance()
            if app is not None:
                _set_tooltip_palette(app)
        except Exception:
            pass

    # ---------------- UI ----------------

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        controls_widget = QtWidgets.QWidget()
        controls = QtWidgets.QGridLayout(controls_widget)
        controls.setHorizontalSpacing(12)
        controls.setVerticalSpacing(8)

        self.model_combo = QtWidgets.QComboBox()
        self._refresh_model_combo()
        controls.addWidget(QtWidgets.QLabel("Model:"), 0, 0)
        controls.addWidget(self.model_combo, 0, 1)

        self.games_spin = QtWidgets.QSpinBox()
        self.games_spin.setRange(10, 100000)
        self.games_spin.setValue(500)
        self.games_spin.setSingleStep(100)
        controls.addWidget(QtWidgets.QLabel("Games per model:"), 0, 2)
        controls.addWidget(self.games_spin, 0, 3)
        controls.setColumnStretch(4, 1)

        self.run_button = QtWidgets.QPushButton("Run / Resume")
        self.run_button.clicked.connect(self.run_simulations)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_simulations)
        self.report_button = QtWidgets.QPushButton("Report Card")
        self.report_button.clicked.connect(self.open_report_card)

        actions = QtWidgets.QHBoxLayout()
        actions.addWidget(self.run_button)
        actions.addWidget(self.cancel_button)
        actions.addWidget(self.report_button)
        actions.addStretch(1)
        controls.addLayout(actions, 1, 0, 1, 4)

        layout.addWidget(controls_widget)

        # Progress bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Table
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(10)
        self.table.setHorizontalHeaderLabels(
            [
                "Model",
                "Games",
                "Avg Shots",
                "Median",
                "Std Dev",
                "P90",
                "P95",
                "Min",
                "Max",
                "Notes",
            ]
        )
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.cellDoubleClicked.connect(self.open_model_details)
        layout.addWidget(self.table)

        # Summary label
        self.summary_label = QtWidgets.QLabel()
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.stale_label = QtWidgets.QLabel("")
        self.stale_label.setStyleSheet("color: #ffb84d; font-weight: bold;")
        self.stale_label.setWordWrap(True)
        layout.addWidget(self.stale_label)

    # ---------------- State helpers ----------------

    def _phase_template(self) -> Dict[str, Dict[str, object]]:
        return {
            phase: {
                "total_games": 0,
                "total_shots": 0.0,
                "sum_sq_shots": 0.0,
                "hist": [0] * (self.total_cells + 1),
            }
            for phase in PHASES
        }

    def _ensure_all_models(self):
        for md in self.model_defs:
            key = md["key"]
            if key not in self.model_stats:
                self.model_stats[key] = {
                    "total_games": 0,
                    "total_shots": 0.0,
                    "sum_sq_shots": 0.0,
                    "min_shots": 0,
                    "max_shots": 0,
                    "hist": [0] * (self.total_cells + 1),
                    "phase": self._phase_template(),
                }
            else:
                cur = self.model_stats[key]
                if "phase" not in cur or not isinstance(cur.get("phase"), dict):
                    cur["phase"] = self._phase_template()
                else:
                    phase_block = cur.get("phase", {})
                    for phase in PHASES:
                        entry = phase_block.get(phase)
                        if not isinstance(entry, dict):
                            phase_block[phase] = {
                                "total_games": 0,
                                "total_shots": 0.0,
                                "sum_sq_shots": 0.0,
                                "hist": [0] * (self.total_cells + 1),
                            }
                        else:
                            if "hist" not in entry or not isinstance(entry.get("hist"), list):
                                entry["hist"] = [0] * (self.total_cells + 1)
                            if len(entry["hist"]) < self.total_cells + 1:
                                entry["hist"] = entry["hist"] + [0] * ((self.total_cells + 1) - len(entry["hist"]))
                            entry.setdefault("total_games", 0)
                            entry.setdefault("total_shots", 0.0)
                            entry.setdefault("sum_sq_shots", 0.0)
                    cur["phase"] = phase_block

    def save_state(self, path: Optional[str] = None):
        if path is None:
            path = self.STATE_PATH
        self._compute_best_models()
        state = {
            "layout_id": self.layout.layout_id,
            "layout_version": self.layout.layout_version,
            "layout_hash": self.layout.layout_hash,
            "model_stats": self.model_stats,
            "param_sweeps": self.param_sweeps,
            "best_model_overall": self.best_model_overall,
            "best_model_by_phase": self.best_model_by_phase,
            "model_overrides": self.model_overrides,
        }
        save_layout_state(path, self.layout, state)

    def _disk_games_for_key(self, key: str) -> int:
        data, _raw = load_layout_state(self.STATE_PATH, self.layout)
        if not data:
            return 0
        stats = data.get("model_stats")
        if not isinstance(stats, dict):
            return 0
        entry = stats.get(key, {})
        if not isinstance(entry, dict):
            return 0
        return int(entry.get("total_games", 0))

    def _should_log_persist(self, key: str) -> bool:
        now = time.time()
        last = self._last_persist_log.get(key, 0.0)
        if now - last >= self._persist_log_interval:
            self._last_persist_log[key] = now
            return True
        return False

    def _log_persist(
        self,
        key: str,
        stage: str,
        disk_before: int,
        mem_before: int,
        mem_after: int,
        mem_at_save: Optional[int],
        disk_after: Optional[int],
    ) -> None:
        self._last_persist_log[key] = time.time()
        msg = (
            f"model={key} stage={stage} "
            f"disk_before={disk_before} "
            f"mem_before={mem_before} "
            f"mem_after={mem_after}"
        )
        if mem_at_save is not None:
            msg += f" mem_at_save={mem_at_save}"
        if disk_after is not None:
            msg += f" disk_after={disk_after}"
        debug_event(self, "ModelStats Persist", msg)

    def _save_state_with_logging(
        self,
        keys: List[str],
        merge_snapshot: Dict[str, Tuple[int, int]],
        stage: str,
        *,
        force_log: bool = False,
    ) -> None:
        if not keys:
            self.save_state()
            return
        disk_before: Dict[str, int] = {}
        for key in keys:
            if force_log or self._should_log_persist(key):
                disk_before[key] = self._disk_games_for_key(key)

        mem_at_save: Dict[str, int] = {}
        for key in keys:
            stats = self.model_stats.get(key, {})
            mem_at_save[key] = int(stats.get("total_games", 0)) if isinstance(stats, dict) else 0

        self.save_state()

        disk_after: Dict[str, int] = {}
        for key in keys:
            if key in disk_before:
                disk_after[key] = self._disk_games_for_key(key)

        for key in keys:
            if key not in disk_before:
                continue
            mem_before, mem_after = merge_snapshot.get(key, (mem_at_save.get(key, 0), mem_at_save.get(key, 0)))
            self._log_persist(
                key,
                stage,
                disk_before[key],
                mem_before,
                mem_after,
                mem_at_save.get(key),
                disk_after.get(key),
            )

    def add_param_sweep(self, model_key: str, sweep_record: Dict[str, object]) -> None:
        self.param_sweeps.setdefault(model_key, [])
        self.param_sweeps[model_key].insert(0, sweep_record)
        self.save_state()

    def get_param_sweeps(self, model_key: str) -> List[Dict[str, object]]:
        sweeps = self.param_sweeps.get(model_key, [])
        return sweeps if isinstance(sweeps, list) else []

    def load_state(self, path: Optional[str] = None):
        if path is None:
            path = self.STATE_PATH
        data, _raw = load_layout_state(path, self.layout)
        if not data:
            stale = find_layout_versions(path, self.layout.layout_id)
            if stale:
                versions = ", ".join(sorted({f"v{v}" for v, _ in stale}))
                self.stale_label.setText(
                    f"Stats exist for previous layout versions ({versions}) and are now stale. Run new sims."
                )
            else:
                self.stale_label.setText("")
            return

        stats = data.get("model_stats")
        if isinstance(stats, dict):
            self.model_stats = stats
        sweeps = data.get("param_sweeps")
        if isinstance(sweeps, dict):
            self.param_sweeps = sweeps
        best_overall = data.get("best_model_overall")
        if isinstance(best_overall, str):
            self.best_model_overall = best_overall
        best_by_phase = data.get("best_model_by_phase")
        if isinstance(best_by_phase, dict):
            self.best_model_by_phase = {
                k: v for k, v in best_by_phase.items() if isinstance(k, str) and isinstance(v, str)
            }
        overrides = data.get("model_overrides")
        overrides_updated = False
        if isinstance(overrides, dict):
            cleaned: Dict[str, Dict[str, str]] = {}
            for key, payload in overrides.items():
                if not isinstance(key, str) or not isinstance(payload, dict):
                    continue
                name = payload.get("name")
                notes = payload.get("notes")
                entry: Dict[str, str] = {}
                if isinstance(name, str) and name.strip():
                    entry["name"] = name.strip()
                if isinstance(notes, str) and notes.strip():
                    entry["notes"] = notes.strip()
                if entry:
                    cleaned[key] = entry
            self.model_overrides = cleaned

            # Migrate stale notes for newly added models if overrides match old defaults.
            old_notes = {
                "ucb_explore": {
                    "Data-free exploration that prefers cells with high posterior uncertainty. Tunable via the UCB bonus parameter.",
                    "Exploration-leaning variant of Greedy. Helps early when the posterior is flat; too much bonus can be wasteful.",
                    "A controlled-exploration variant of Greedy. Helps early when the posterior is flat; too much bonus can be wasteful.",
                },
                "rollout_mcts": {
                    "Monte Carlo lookahead with a fast rollout policy. Data-free but heavier to simulate; keep rollouts small in Model Stats.",
                    "Monte Carlo lookahead with a fast rollout policy. Slower but can reduce total shots when the posterior is ambiguous.",
                },
            }
            default_notes = {md.get("key"): md.get("notes") for md in self._base_model_defs}
            for key, legacy in old_notes.items():
                current = cleaned.get(key, {})
                note = current.get("notes")
                if isinstance(note, str) and note.strip() in legacy:
                    new_note = default_notes.get(key)
                    if isinstance(new_note, str) and new_note.strip():
                        cleaned[key]["notes"] = new_note.strip()
                        overrides_updated = True
            self.model_overrides = cleaned
        self._ensure_all_models()
        self.stale_label.setText("")
        if overrides_updated:
            try:
                self.save_state()
            except Exception:
                pass

    # ---------------- Stats merge + table ----------------

    def _apply_model_overrides(self) -> None:
        self.model_defs = [dict(md) for md in self._base_model_defs]
        for md in self.model_defs:
            key = md.get("key")
            if not key or key not in self.model_overrides:
                continue
            override = self.model_overrides.get(key, {})
            name = override.get("name")
            notes = override.get("notes")
            if isinstance(name, str) and name.strip():
                md["name"] = name.strip()
            if isinstance(notes, str) and notes.strip():
                md["notes"] = notes.strip()

    def _refresh_model_combo(self) -> None:
        try:
            self.model_combo.blockSignals(True)
            self.model_combo.clear()
            self.model_combo.addItem("All models")
            for md in self.model_defs:
                self.model_combo.addItem(md.get("name", md.get("key", "Model")))
        finally:
            self.model_combo.blockSignals(False)

    def update_model_override(self, model_key: str, name: str, notes: str) -> None:
        key = str(model_key or "").strip()
        if not key:
            return
        payload: Dict[str, str] = {}
        if isinstance(name, str) and name.strip():
            payload["name"] = name.strip()
        if isinstance(notes, str) and notes.strip():
            payload["notes"] = notes.strip()
        if payload:
            self.model_overrides[key] = payload
        else:
            self.model_overrides.pop(key, None)
        self._apply_model_overrides()
        self._refresh_model_combo()
        self.refresh_table()
        self.save_state()

    def _merge_model_stats(self, key: str, delta: Dict[str, object]):
        self._ensure_all_models()
        cur = self.model_stats[key]

        dg = int(delta.get("total_games", 0))
        if dg <= 0:
            return

        cur_g = int(cur.get("total_games", 0))
        cur_s = float(cur.get("total_shots", 0.0))
        cur_sq = float(cur.get("sum_sq_shots", 0.0))
        cur_min = int(cur.get("min_shots", 0))
        cur_max = int(cur.get("max_shots", 0))
        cur_hist = cur.get("hist")
        if not isinstance(cur_hist, list) or len(cur_hist) < self.total_cells + 1:
            cur_hist = [0] * (self.total_cells + 1)

        dg_s = float(delta.get("total_shots", 0.0))
        dg_sq = float(delta.get("sum_sq_shots", 0.0))
        dg_min = int(delta.get("min_shots", cur_min))
        dg_max = int(delta.get("max_shots", cur_max))
        dg_hist = delta.get("hist") or [0] * (self.total_cells + 1)

        new_g = cur_g + dg
        new_s = cur_s + dg_s
        new_sq = cur_sq + dg_sq

        if cur_g == 0:
            new_min = dg_min
            new_max = dg_max
        else:
            new_min = min(cur_min, dg_min)
            new_max = max(cur_max, dg_max)

        new_hist = [0] * max(len(cur_hist), len(dg_hist))
        for i in range(len(new_hist)):
            c = cur_hist[i] if i < len(cur_hist) else 0
            d = dg_hist[i] if i < len(dg_hist) else 0
            new_hist[i] = c + d

        cur["total_games"] = new_g
        cur["total_shots"] = new_s
        cur["sum_sq_shots"] = new_sq
        cur["min_shots"] = new_min
        cur["max_shots"] = new_max
        cur["hist"] = new_hist

        # Merge phase stats if present
        delta_phase = delta.get("phase")
        if isinstance(delta_phase, dict):
            cur_phase = cur.get("phase")
            if not isinstance(cur_phase, dict):
                cur_phase = self._phase_template()
            for phase in PHASES:
                d = delta_phase.get(phase)
                if not isinstance(d, dict):
                    continue
                c = cur_phase.get(phase, {})
                if not isinstance(c, dict):
                    c = {"total_games": 0, "total_shots": 0.0, "sum_sq_shots": 0.0, "hist": [0] * (self.total_cells + 1)}

                c_games = int(c.get("total_games", 0))
                c_total = float(c.get("total_shots", 0.0))
                c_sq = float(c.get("sum_sq_shots", 0.0))
                c_hist = c.get("hist") or [0] * (self.total_cells + 1)

                d_games = int(d.get("total_games", 0))
                d_total = float(d.get("total_shots", 0.0))
                d_sq = float(d.get("sum_sq_shots", 0.0))
                d_hist = d.get("hist") or [0] * (self.total_cells + 1)

                merged_hist = [0] * max(len(c_hist), len(d_hist))
                for i in range(len(merged_hist)):
                    merged_hist[i] = (c_hist[i] if i < len(c_hist) else 0) + (d_hist[i] if i < len(d_hist) else 0)

                c["total_games"] = c_games + d_games
                c["total_shots"] = c_total + d_total
                c["sum_sq_shots"] = c_sq + d_sq
                c["hist"] = merged_hist
                cur_phase[phase] = c

            cur["phase"] = cur_phase

    def refresh_table(self):
        self.table.setRowCount(len(self.model_defs))

        for row, md in enumerate(self.model_defs):
            key = md["key"]
            stats = self.model_stats.get(key, {})

            metrics = _compute_basic_stats(stats)
            games = int(metrics["games"])
            mean = float(metrics["mean"])
            std = float(metrics["std"])
            median = float(metrics["median"])
            p90 = float(metrics["p90"])
            p95 = float(metrics["p95"])

            # Column 0: model name (store key in UserRole so lookups never depend on visible text)
            item_name = QtWidgets.QTableWidgetItem(md["name"])
            item_name.setData(QtCore.Qt.UserRole, key)
            self.table.setItem(row, 0, item_name)

            # Numeric columns
            vals = [
                str(games),
                f"{mean:.2f}" if games > 0 else "-",
                f"{median:.0f}" if games > 0 else "-",
                f"{std:.2f}" if games > 0 else "-",
                f"{p90:.0f}" if games > 0 else "-",
                f"{p95:.0f}" if games > 0 else "-",
                str(stats.get("min_shots", "-")),
                str(stats.get("max_shots", "-")),
            ]
            for c, v in enumerate(vals, start=1):
                it = QtWidgets.QTableWidgetItem(v)
                it.setTextAlignment(QtCore.Qt.AlignCenter)
                self.table.setItem(row, c, it)

            # Notes column (short)
            note = md.get("notes") or md.get("description") or ""
            if len(note) > 120:
                note = note[:117] + "..."
            it_note = QtWidgets.QTableWidgetItem(note)
            it_note.setTextAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            self.table.setItem(row, 9, it_note)

        self.update_summary_label()

    def open_model_details(self, row, col):
        try:
            item = self.table.item(row, 0)
            if item is None:
                return

            key = item.data(QtCore.Qt.UserRole)
            if not key:
                return

            model_def = next((md for md in self.model_defs if md.get("key") == key), None)
            if not model_def:
                return

            dlg = ModelDetailDialog(
                model_def,
                self.model_stats.get(key, {}),
                self.placements,
                stats_tab=self,
                parent=self,
            )
            dlg.exec_()
        except Exception:
            import traceback
            tb = traceback.format_exc()
            try:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error",
                    "An error occurred while opening model details.\n\n" + tb,
                )
            except Exception:
                # If the GUI is in a bad state, fall back to stderr
                import sys
                print(tb, file=sys.stderr)

    def update_summary_label(self):
        self._compute_best_models()
        best_overall_name = None
        best_overall_mean = None

        if self.best_model_overall:
            for md in self.model_defs:
                if md["key"] == self.best_model_overall:
                    best_overall_name = md["name"]
                    stats = self.model_stats.get(md["key"], {})
                    games = int(stats.get("total_games", 0))
                    total_shots = float(stats.get("total_shots", 0.0))
                    best_overall_mean = (total_shots / games) if games > 0 else None
                    break

        if not best_overall_name:
            self.summary_label.setText(
                "Run some simulations to compare model efficiency (shots to sink all ships)."
            )
            return

        phase_bits = []
        for phase in PHASES:
            key = self.best_model_by_phase.get(phase)
            if not key:
                continue
            name = next((md["name"] for md in self.model_defs if md["key"] == key), key)
            phase_bits.append(f"{phase}: {name}")

        phase_text = " | ".join(phase_bits) if phase_bits else "Phase bests: N/A"
        mean_text = f"{best_overall_mean:.2f}" if best_overall_mean is not None else "N/A"
        self.summary_label.setText(
            f"Best overall: <b>{best_overall_name}</b> (~{mean_text} shots). {phase_text}"
        )

    def _compute_best_models(self) -> None:
        best_overall = None
        best_overall_mean = None

        for md in self.model_defs:
            key = md["key"]
            stats = self.model_stats.get(key, {})
            games = int(stats.get("total_games", 0))
            if games <= 0:
                continue
            total_shots = float(stats.get("total_shots", 0.0))
            mean = total_shots / games
            if best_overall_mean is None or mean < best_overall_mean:
                best_overall_mean = mean
                best_overall = key

        best_by_phase: Dict[str, str] = {}
        for phase in PHASES:
            best_key = None
            best_avg = None
            for md in self.model_defs:
                key = md["key"]
                stats = self.model_stats.get(key, {})
                avg = _compute_phase_avg(stats, phase)
                if avg is None:
                    continue
                if best_avg is None or avg < best_avg:
                    best_avg = avg
                    best_key = key
            if best_key:
                best_by_phase[phase] = best_key

        self.best_model_overall = best_overall
        self.best_model_by_phase = best_by_phase

    def open_report_card(self):
        dlg = ReportCardDialog(self.model_defs, self.model_stats, parent=self)
        dlg.exec_()

    # ---------------- Simulation control (threaded) ----------------

    def run_simulations(self):
        if self._sim_thread is not None:
            QtWidgets.QMessageBox.information(self, "Busy", "Simulations running.")
            return

        idx = self.model_combo.currentIndex()
        target_games = self.games_spin.value()  # This is the "Goal" number
        # Determine which models to run
        if idx == 0:
            candidates = [md["key"] for md in self.model_defs]
        else:
            candidates = [self.model_defs[idx - 1]["key"]]

        # Calculate actual work needed (top-up to target)
        work_order = {}  # key -> games_to_run
        total_jobs = 0

        for key in candidates:
            current_stats = self.model_stats.get(key, {})
            current_games = int(current_stats.get("total_games", 0))

            if current_games >= target_games:
                continue
            needed = target_games - current_games
            if needed > 0:
                work_order[key] = needed
                total_jobs += needed

        self._active_sim_keys = list(work_order.keys())

        if total_jobs == 0:
            QtWidgets.QMessageBox.information(
                self,
                "Done",
                f"All selected models already have at least {target_games} games simulated.",
            )
            return

        self.progress_bar.setRange(0, total_jobs)
        self.progress_bar.setValue(0)

        # Disable controls
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.model_combo.setEnabled(False)
        self.games_spin.setEnabled(False)

        self._sim_thread = QtCore.QThread(self)
        # Pass the calculated work_order instead of raw numbers
        self._sim_worker = SimulationWorker(work_order, self.placements, self.ship_ids, self.board_size)
        self._sim_worker.moveToThread(self._sim_thread)

        self._sim_thread.started.connect(self._sim_worker.run)
        self._sim_worker.progress.connect(self._on_worker_progress)
        self._sim_worker.checkpoint.connect(self._on_worker_checkpoint)
        self._sim_worker.finished.connect(self._on_worker_finished)
        self._sim_worker.finished.connect(self._sim_thread.quit)
        self._sim_worker.finished.connect(self._sim_worker.deleteLater)
        self._sim_thread.finished.connect(self._on_thread_finished)
        self._sim_thread.finished.connect(self._sim_thread.deleteLater)

        self._sim_thread.start()

    def cancel_simulations(self):
        if self._sim_worker is not None:
            self._sim_worker.cancel()

    @QtCore.pyqtSlot(int, int)
    def _on_worker_progress(self, done: int, total: int):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(done)

    @QtCore.pyqtSlot(dict, int, int)
    def _on_worker_checkpoint(self, delta_stats: Dict[str, Dict[str, object]], done: int, total: int):
        if not delta_stats:
            return
        locker = QtCore.QMutexLocker(self._stats_lock)
        merge_snapshot: Dict[str, Tuple[int, int]] = {}
        for key, delta in delta_stats.items():
            stats = self.model_stats.get(key, {})
            mem_before = int(stats.get("total_games", 0)) if isinstance(stats, dict) else 0
            disk_before = None
            if self._should_log_persist(key):
                disk_before = self._disk_games_for_key(key)
            self._merge_model_stats(key, delta)
            stats_after = self.model_stats.get(key, {})
            mem_after = int(stats_after.get("total_games", 0)) if isinstance(stats_after, dict) else mem_before
            merge_snapshot[key] = (mem_before, mem_after)
            if disk_before is not None:
                self._log_persist(
                    key,
                    "merge",
                    disk_before,
                    mem_before,
                    mem_after,
                    None,
                    None,
                )

        now = time.time()
        if now - self._last_checkpoint_save >= 2.0:
            self._save_state_with_logging(list(delta_stats.keys()), merge_snapshot, "checkpoint_save", force_log=True)
            self._last_checkpoint_save = now
        del locker

    @QtCore.pyqtSlot(dict)
    def _on_worker_finished(self, delta_stats: Dict[str, Dict[str, object]]):
        locker = QtCore.QMutexLocker(self._stats_lock)
        merge_snapshot: Dict[str, Tuple[int, int]] = {}
        for key, delta in delta_stats.items():
            stats = self.model_stats.get(key, {})
            mem_before = int(stats.get("total_games", 0)) if isinstance(stats, dict) else 0
            disk_before = None
            if self._should_log_persist(key):
                disk_before = self._disk_games_for_key(key)
            self._merge_model_stats(key, delta)
            stats_after = self.model_stats.get(key, {})
            mem_after = int(stats_after.get("total_games", 0)) if isinstance(stats_after, dict) else mem_before
            merge_snapshot[key] = (mem_before, mem_after)
            if disk_before is not None:
                self._log_persist(
                    key,
                    "merge_final",
                    disk_before,
                    mem_before,
                    mem_after,
                    None,
                    None,
                )
        keys = list(delta_stats.keys()) or list(self._active_sim_keys)
        self._save_state_with_logging(keys, merge_snapshot, "final_save", force_log=True)
        for key in keys:
            stats = self.model_stats.get(key, {})
            games = int(stats.get("total_games", 0)) if isinstance(stats, dict) else 0
            debug_event(self, "ModelStats Persist", f"model={key} stage=final_totals mem_total={games}")
        self._active_sim_keys = []
        del locker
        self.refresh_table()
        self.update_summary_label()

    @QtCore.pyqtSlot()
    def _on_thread_finished(self):
        self._sim_thread = None
        self._sim_worker = None
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.model_combo.setEnabled(True)
        self.games_spin.setEnabled(True)
        # leave progress bar at its final max
        self.progress_bar.setValue(self.progress_bar.maximum())
        if self._active_sim_keys:
            locker = QtCore.QMutexLocker(self._stats_lock)
            self._save_state_with_logging(self._active_sim_keys, {}, "thread_finished_save", force_log=True)
            for key in self._active_sim_keys:
                stats = self.model_stats.get(key, {})
                games = int(stats.get("total_games", 0)) if isinstance(stats, dict) else 0
                debug_event(self, "ModelStats Persist", f"model={key} stage=thread_finished mem_total={games}")
            self._active_sim_keys = []
            del locker


class SimulationWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(int, int)  # done, total
    checkpoint = QtCore.pyqtSignal(dict, int, int)  # delta stats, done, total
    finished = QtCore.pyqtSignal(dict)  # key -> delta stats

    def __init__(
        self,
        work_order: Dict[str, int],  # Map: "greedy" -> 500 more games
        placements: Dict[str, List[object]],
        ship_ids: List[str],
        board_size: int,
    ):
        super().__init__()
        self.work_order = work_order
        self.placements = placements
        self.ship_ids = list(ship_ids)
        self.board_size = int(board_size)
        self.total_cells = self.board_size * self.board_size
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def _blank_stats(self) -> Dict[str, object]:
        return {
            "total_games": 0,
            "total_shots": 0.0,
            "sum_sq_shots": 0.0,
            "min_shots": 0,
            "max_shots": 0,
            "hist": [0] * (self.total_cells + 1),
            "phase": {
                phase: {
                    "total_games": 0,
                    "total_shots": 0.0,
                    "sum_sq_shots": 0.0,
                    "hist": [0] * (self.total_cells + 1),
                }
                for phase in PHASES
            },
        }

    @QtCore.pyqtSlot()
    def run(self):
        rng = random.Random()
        total_jobs = sum(self.work_order.values())
        if total_jobs == 0:
            self.finished.emit({})
            return

        profile_enabled = _sim_profile_enabled()
        profilers: Dict[str, SimProfiler] = {}
        if profile_enabled:
            for key in self.work_order.keys():
                profilers[key] = SimProfiler()

        # NEW: Update roughly every 1% or every 1 game, whichever is larger
        update_interval = max(1, total_jobs // 100)
        checkpoint_interval = max(1, total_jobs // 20)

        done = 0
        pending: Dict[str, Dict[str, object]] = {}
        last_checkpoint_done = 0

        for key, num_to_run in self.work_order.items():
            if key not in pending:
                pending[key] = self._blank_stats()
            stats = pending[key]

            for _ in range(num_to_run):
                if self._cancelled:
                    break
                shots, phase_counts = simulate_model_game_with_phases(
                    key,
                    self.placements,
                    self.ship_ids,
                    self.board_size,
                    rng,
                    profiler=profilers.get(key) if profile_enabled else None,
                )

                stats["total_games"] = int(stats.get("total_games", 0)) + 1
                stats["total_shots"] = float(stats.get("total_shots", 0.0)) + shots
                stats["sum_sq_shots"] = float(stats.get("sum_sq_shots", 0.0)) + (shots * shots)

                if stats["total_games"] == 1:
                    stats["min_shots"] = shots
                    stats["max_shots"] = shots
                else:
                    stats["min_shots"] = min(int(stats.get("min_shots", shots)), shots)
                    stats["max_shots"] = max(int(stats.get("max_shots", shots)), shots)

                hist = stats.get("hist")
                if not isinstance(hist, list) or len(hist) < self.total_cells + 1:
                    hist = [0] * (self.total_cells + 1)
                if 0 <= shots < len(hist):
                    hist[shots] += 1
                else:
                    hist[-1] += 1
                stats["hist"] = hist

                if phase_counts:
                    phase_stats = stats.get("phase")
                    if not isinstance(phase_stats, dict):
                        phase_stats = self._blank_stats().get("phase", {})
                    for phase in PHASES:
                        phase_shots = int(phase_counts.get(phase, 0))
                        entry = phase_stats.get(phase, {})
                        entry["total_games"] = int(entry.get("total_games", 0)) + 1
                        entry["total_shots"] = float(entry.get("total_shots", 0.0)) + phase_shots
                        entry["sum_sq_shots"] = float(entry.get("sum_sq_shots", 0.0)) + (phase_shots * phase_shots)
                        phist = entry.get("hist")
                        if not isinstance(phist, list):
                            phist = [0] * (self.total_cells + 1)
                        idx = phase_shots if 0 <= phase_shots < len(phist) else len(phist) - 1
                        phist[idx] += 1
                        entry["hist"] = phist
                        phase_stats[phase] = entry
                    stats["phase"] = phase_stats

                done += 1
                # NEW: Smoother progress emission
                if done % update_interval == 0 or done == total_jobs:
                    self.progress.emit(done, total_jobs)
                if done - last_checkpoint_done >= checkpoint_interval:
                    payload = {k: v for k, v in pending.items() if int(v.get("total_games", 0)) > 0}
                    if payload:
                        self.checkpoint.emit(payload, done, total_jobs)
                        for k in payload.keys():
                            pending[k] = self._blank_stats()
                        last_checkpoint_done = done

            if self._cancelled:
                break
            if profile_enabled and key in profilers and profilers[key].games > 0:
                summary = profilers[key].format_summary(f"model={key}")
                print(summary)
                profilers[key].reset()

        final_payload = {k: v for k, v in pending.items() if int(v.get("total_games", 0)) > 0}
        self.finished.emit(final_payload)


class CustomSimWorker(QtCore.QThread):
    """Runs a custom simulation batch off the UI thread."""
    progress = QtCore.pyqtSignal(int)
    result = QtCore.pyqtSignal(float, float, int)  # avg, std, count
    error = QtCore.pyqtSignal(str)

    def __init__(self, strategy: str, placements, ship_ids: List[str], board_size: int, count: int, params: dict):
        super().__init__()
        self.strategy = strategy
        self.placements = placements
        self.ship_ids = list(ship_ids)
        self.board_size = int(board_size)
        self.count = int(count)
        self.params = dict(params or {})
        self.is_cancelled = False

    def cancel(self):
        self.is_cancelled = True

    def run(self):
        try:
            rng = random.Random()
            total_shots = 0
            sum_sq = 0
            ran = 0
            profile_enabled = _sim_profile_enabled()
            profiler = SimProfiler() if profile_enabled else None

            for _ in range(self.count):
                if self.is_cancelled:
                    break

                shots = simulate_model_game(
                    self.strategy,
                    self.placements,
                    self.ship_ids,
                    self.board_size,
                    rng=rng,
                    params=self.params,
                    profiler=profiler,
                )
                total_shots += shots
                sum_sq += shots * shots
                ran += 1

                # Emit progress occasionally to reduce signal overhead.
                if ran % 5 == 0 or ran == self.count:
                    self.progress.emit(ran)

            if ran > 0:
                avg = total_shots / ran
                var = max(0.0, sum_sq / ran - avg * avg)
                std = math.sqrt(var)
                self.result.emit(avg, std, ran)
            else:
                self.result.emit(0.0, 0.0, 0)
            if profile_enabled and profiler is not None and profiler.games > 0:
                print(profiler.format_summary(f"custom={self.strategy}"))
        except Exception:
            import traceback as _tb
            self.error.emit(_tb.format_exc())


def _sweep_values(min_v: float, max_v: float, step: float) -> List[float]:
    """Inclusive float range helper for parameter sweeps."""
    if step <= 0:
        return [min_v]
    if max_v < min_v:
        min_v, max_v = max_v, min_v
    out: List[float] = []
    # Guard against float drift
    v = float(min_v)
    # Include endpoint with a small epsilon
    eps = abs(step) * 1.0e-9 + 1.0e-12
    while v <= max_v + eps:
        out.append(float(v))
        v += step
    # Snap last value exactly to max if we're within epsilon
    if out and abs(out[-1] - max_v) <= eps:
        out[-1] = float(max_v)
    return out


def _count_values_for_range(min_v: float, max_v: float, step: float) -> int:
    """Count how many values will be generated for a sweep (inclusive endpoints)."""
    if step <= 0:
        return 0
    lo = min(min_v, max_v)
    hi = max(min_v, max_v)
    n = int(math.floor((hi - lo) / step + 1e-12)) + 1
    return max(0, n)


def _values_for_range(min_v: float, max_v: float, step: float, is_int: bool) -> List[float]:
    if step <= 0:
        return [min_v]
    lo = min(min_v, max_v)
    hi = max(min_v, max_v)

    if is_int:
        lo_i = int(round(lo))
        hi_i = int(round(hi))
        st_i = max(1, int(round(step)))
        return [float(v) for v in range(lo_i, hi_i + 1, st_i)]

    out: List[float] = []
    v = float(lo)
    eps = abs(step) * 1.0e-9 + 1.0e-12
    cap = 5000
    while v <= hi + eps and len(out) < cap:
        out.append(float(round(v, 10)))
        v += step
    if out and abs(out[-1] - hi) <= eps:
        out[-1] = float(hi)
    return out


def _grid_from_params(params: List[Dict[str, object]], step_scale: float = 1.0, centers: Optional[Dict[str, float]] = None,
                      spans: Optional[Dict[str, float]] = None) -> List[Dict[str, float]]:
    keys: List[str] = []
    value_lists: List[List[float]] = []
    for p in params:
        key = str(p.get("key"))
        min_v = float(p.get("min", 0.0))
        max_v = float(p.get("max", 0.0))
        step = float(p.get("step", 1.0)) * float(step_scale)
        is_int = bool(p.get("is_int", False))

        if spans and centers and key in centers and key in spans:
            center = float(centers[key])
            span = float(spans[key])
            min_v = max(min_v, center - span)
            max_v = min(max_v, center + span)

        values = _values_for_range(min_v, max_v, step, is_int)
        if not values:
            values = [min_v]
        keys.append(key)
        value_lists.append(values)

    grid: List[Dict[str, float]] = []
    for combo in itertools.product(*value_lists):
        grid.append({k: float(v) for k, v in zip(keys, combo)})
    return grid


class ParamSweepWorker(QtCore.QThread):
    """Runs a parameter sweep for a given strategy off the UI thread.

    Notes:
      - Do NOT override QThread.finished; use `result` for returning data.

    Emits:
      - progress(int): number of sweep points completed
      - result(object, int): (results_list, points_completed)
      - error(str): traceback text
    """
    progress = QtCore.pyqtSignal(int)
    result = QtCore.pyqtSignal(object, int)
    error = QtCore.pyqtSignal(str)

    def __init__(
        self,
        strategy: str,
        placements,
        ship_ids: List[str],
        board_size: int,
        games_per: int,
        sweep_config: Dict[str, object],
    ):
        super().__init__()
        self.strategy = strategy
        self.placements = placements
        self.ship_ids = list(ship_ids)
        self.board_size = int(board_size)
        self.games_per = int(games_per)
        self.sweep_config = dict(sweep_config)
        self._cancelled = False
        self.was_cancelled = False
        self.results = []
        self.points_completed = 0
        self.stop_reason = "completed"
        self.elapsed_seconds = 0.0

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            self._run_impl()
        except Exception:
            import traceback as _tb
            self.error.emit(_tb.format_exc())

    def _run_impl(self):
        rng = random.Random()
        results: List[Dict[str, object]] = []
        total_cells = self.board_size * self.board_size
        points_completed = 0

        config = self.sweep_config or {}
        params = list(config.get("params") or [])
        mode = str(config.get("mode") or "grid")
        coarse_factor = float(config.get("coarse_factor") or 2.0)
        refine_top_n = int(config.get("refine_top_n") or 1)
        max_points = int(config.get("max_points") or 0)
        time_budget_s = float(config.get("time_budget_s") or 0.0)
        early_stop = int(config.get("early_stop_no_improve") or 0)

        key_order = [str(p.get("key")) for p in params]
        seen = set()
        best_avg = None
        no_improve = 0
        start = time.monotonic()

        def _param_tuple(pdict: Dict[str, float]) -> tuple:
            return tuple(round(float(pdict.get(k, 0.0)), 10) for k in key_order)

        def _should_stop() -> bool:
            nonlocal points_completed
            if self._cancelled:
                self.was_cancelled = True
                self.stop_reason = "cancelled"
                return True
            if max_points > 0 and points_completed >= max_points:
                self.stop_reason = "point_budget"
                return True
            if time_budget_s > 0 and (time.monotonic() - start) >= time_budget_s:
                self.stop_reason = "time_budget"
                return True
            if early_stop > 0 and no_improve >= early_stop:
                self.stop_reason = "early_stop"
                return True
            return False

        def _span_for_param(p: Dict[str, object], factor: float) -> float:
            base_step = float(p.get("step", 1.0))
            if bool(p.get("is_int", False)):
                return float(max(1, int(round(base_step * factor))))
            return float(base_step * factor)

        def _run_grid(grid: List[Dict[str, float]]) -> None:
            nonlocal points_completed, best_avg, no_improve
            for params_dict in grid:
                if _should_stop():
                    break
                key = _param_tuple(params_dict)
                if key in seen:
                    continue
                seen.add(key)

                total_shots = 0
                sum_sq = 0
                ran = 0
                min_shots = None
                max_shots = None
                hist = [0] * (total_cells + 1)

                for _ in range(self.games_per):
                    if self._cancelled:
                        self.was_cancelled = True
                        self.stop_reason = "cancelled"
                        break

                    shots = simulate_model_game(
                        self.strategy,
                        self.placements,
                        self.ship_ids,
                        self.board_size,
                        rng=rng,
                        params=params_dict,
                    )
                    total_shots += shots
                    sum_sq += shots * shots
                    ran += 1

                    if min_shots is None or shots < min_shots:
                        min_shots = shots
                    if max_shots is None or shots > max_shots:
                        max_shots = shots

                    if 0 <= shots <= total_cells:
                        hist[shots] += 1

                if ran > 0:
                    avg = total_shots / ran
                    var = max(0.0, (sum_sq / ran) - (avg * avg))
                    std = math.sqrt(var)
                    results.append({
                        "params": dict(params_dict),
                        "games": ran,
                        "avg": avg,
                        "std": std,
                        "min": int(min_shots) if min_shots is not None else 0,
                        "max": int(max_shots) if max_shots is not None else 0,
                        "hist": hist,
                    })
                    points_completed += 1

                    if best_avg is None or avg < best_avg:
                        best_avg = avg
                        no_improve = 0
                    else:
                        no_improve += 1

                self.progress.emit(points_completed)

                if self.was_cancelled:
                    break

        # Stage 1: base or coarse grid
        if mode == "coarse_to_fine":
            coarse_grid = _grid_from_params(params, step_scale=max(1.0, coarse_factor))
            _run_grid(coarse_grid)

            if not _should_stop() and results and refine_top_n > 0:
                candidates = sorted(results, key=lambda r: float(r.get("avg", math.inf)))
                top = candidates[:max(1, refine_top_n)]
                spans = {str(p.get("key")): _span_for_param(p, max(1.0, coarse_factor)) for p in params}
                for best in top:
                    if _should_stop():
                        break
                    centers = best.get("params", {}) or {}
                    refine_grid = _grid_from_params(params, step_scale=1.0, centers=centers, spans=spans)
                    _run_grid(refine_grid)
        else:
            grid = _grid_from_params(params, step_scale=1.0)
            _run_grid(grid)

        self.elapsed_seconds = max(0.0, time.monotonic() - start)
        if self.was_cancelled:
            self.stop_reason = "cancelled"
        elif self.stop_reason == "completed" and (max_points or time_budget_s or early_stop):
            # If budgets were present but we didn't trigger, keep completed.
            self.stop_reason = "completed"

        self.results = results
        self.points_completed = points_completed
        self.result.emit(results, points_completed)


class ReportCardDialog(QtWidgets.QDialog):
    def __init__(self, model_defs: List[Dict[str, object]], model_stats: Dict[str, Dict[str, object]], parent=None):
        super().__init__(parent)
        self.model_defs = model_defs
        self.model_stats = model_stats
        self.setWindowTitle("Model Report Card")
        self.resize(980, 520)
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels(
            ["Model", "Games", "Avg", "Median", "P90", "P95", "HUNT Avg", "TARGET Avg", "ENDGAME Avg"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        self.table.setRowCount(len(self.model_defs))

        best_overall_key = None
        best_overall_mean = None
        best_by_phase: Dict[str, str] = {}

        for md in self.model_defs:
            key = md["key"]
            stats = self.model_stats.get(key, {})
            games = int(stats.get("total_games", 0))
            if games <= 0:
                continue
            mean = float(stats.get("total_shots", 0.0)) / games
            if best_overall_mean is None or mean < best_overall_mean:
                best_overall_mean = mean
                best_overall_key = key

        for phase in PHASES:
            best_key = None
            best_avg = None
            for md in self.model_defs:
                key = md["key"]
                avg = _compute_phase_avg(self.model_stats.get(key, {}), phase)
                if avg is None:
                    continue
                if best_avg is None or avg < best_avg:
                    best_avg = avg
                    best_key = key
            if best_key:
                best_by_phase[phase] = best_key

        for row, md in enumerate(self.model_defs):
            key = md["key"]
            stats = self.model_stats.get(key, {})
            metrics = _compute_basic_stats(stats)

            vals = [
                md["name"],
                str(int(metrics["games"])),
                f"{metrics['mean']:.2f}" if metrics["games"] > 0 else "-",
                f"{metrics['median']:.0f}" if metrics["games"] > 0 else "-",
                f"{metrics['p90']:.0f}" if metrics["games"] > 0 else "-",
                f"{metrics['p95']:.0f}" if metrics["games"] > 0 else "-",
            ]

            for col, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(v)
                if col > 0:
                    it.setTextAlignment(QtCore.Qt.AlignCenter)
                self.table.setItem(row, col, it)

            phase_avgs = []
            for phase in PHASES:
                avg = _compute_phase_avg(stats, phase)
                phase_avgs.append(f"{avg:.2f}" if avg is not None else "-")

            for i, v in enumerate(phase_avgs, start=6):
                it = QtWidgets.QTableWidgetItem(v)
                it.setTextAlignment(QtCore.Qt.AlignCenter)
                self.table.setItem(row, i, it)

            if best_overall_key == key:
                item = self.table.item(row, 0)
                if item is not None:
                    item.setFont(QtGui.QFont(item.font().family(), item.font().pointSize(), QtGui.QFont.Bold))

            for i, phase in enumerate(PHASES, start=6):
                if best_by_phase.get(phase) == key:
                    cell = self.table.item(row, i)
                    if cell is not None:
                        cell.setBackground(QtGui.QColor(Theme.HIGHLIGHT))
                        cell.setForeground(QtGui.QColor(Theme.TEXT_DARK))

        layout.addWidget(self.table)

        summary = QtWidgets.QLabel()
        if best_overall_key:
            best_name = next((md["name"] for md in self.model_defs if md["key"] == best_overall_key), best_overall_key)
            parts = [f"Best overall: {best_name}"]
            for phase in PHASES:
                key = best_by_phase.get(phase)
                if key:
                    name = next((md["name"] for md in self.model_defs if md["key"] == key), key)
                    parts.append(f"{phase}: {name}")
            summary.setText(" | ".join(parts))
        else:
            summary.setText("No simulation data yet.")
        layout.addWidget(summary)

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close, alignment=QtCore.Qt.AlignRight)

class SavedSweepsDialog(QtWidgets.QDialog):
    def __init__(self, model_def: Dict[str, object], sweeps: List[Dict[str, object]], parent=None):
        super().__init__(parent)
        self.model_def = model_def
        self.sweeps = sweeps[:] if isinstance(sweeps, list) else []
        self.setWindowTitle(f"Saved Sweeps: {model_def.get('name', model_def.get('key', 'Model'))}")
        self.resize(900, 520)
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        if not self.sweeps:
            layout.addWidget(QtWidgets.QLabel("No saved sweeps for this model yet."))
            btn_close = QtWidgets.QPushButton("Close")
            btn_close.clicked.connect(self.accept)
            layout.addWidget(btn_close, alignment=QtCore.Qt.AlignRight)
            return

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["Name", "Date", "Games/pt", "Grid", "Best Avg", "Ranges"])
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(5, QtWidgets.QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        self.table.setRowCount(len(self.sweeps))
        for r, sw in enumerate(self.sweeps):
            name = sw.get("name") or "(unnamed sweep)"
            created = sw.get("created_at") or ""
            gpp = sw.get("games_per_point", "")
            grid = sw.get("grid_size", "")
            best = sw.get("best", {}) or {}
            best_avg = best.get("avg", "")
            ranges = sw.get("ranges", {}) or {}

            it0 = QtWidgets.QTableWidgetItem(str(name))
            it0.setData(QtCore.Qt.UserRole, sw)
            self.table.setItem(r, 0, it0)

            self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(str(created)))
            self.table.setItem(r, 2, QtWidgets.QTableWidgetItem(str(gpp)))
            self.table.setItem(r, 3, QtWidgets.QTableWidgetItem(str(grid)))

            try:
                self.table.setItem(r, 4, QtWidgets.QTableWidgetItem("" if best_avg == "" else f"{float(best_avg):.2f}"))
            except Exception:
                self.table.setItem(r, 4, QtWidgets.QTableWidgetItem(str(best_avg)))

            self.table.setItem(r, 5, QtWidgets.QTableWidgetItem(json.dumps(ranges, sort_keys=True)))

        self.table.cellDoubleClicked.connect(self.accept)
        layout.addWidget(self.table)

        hint = QtWidgets.QLabel("Select a sweep and press OK (or double-click) to open it.")
        hint.setStyleSheet(f"color: {Theme.TEXT_MUTED};")
        layout.addWidget(hint)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def selected_sweep(self) -> Optional[Dict[str, object]]:
        if not hasattr(self, "table"):
            return None
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        if not item:
            return None
        sw = item.data(QtCore.Qt.UserRole)
        return sw if isinstance(sw, dict) else None


class SweepResultsDialog(QtWidgets.QDialog):
    def __init__(
        self,
        results: List[Dict[str, object]],
        best: Dict[str, object],
        model_def: Optional[Dict[str, object]] = None,
        sweep_meta: Optional[Dict[str, object]] = None,
        stats_tab: Optional["ModelStatsTab"] = None,
        parent=None,
        view_only: bool = False,
    ):
        super().__init__(parent)
        self.results = results or []
        self.best = best or {}
        self.model_def = model_def or {}
        self.sweep_meta = sweep_meta or {}
        self.stats_tab = stats_tab
        self.view_only = view_only

        title_model = self.model_def.get("name") or self.model_def.get("key") or "Model"
        self.setWindowTitle(f"Parameter Sweep Results — {title_model}")
        self.resize(1100, 650)
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        header = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel("Sweep Summary")
        f = lbl.font()
        f.setPointSize(14)
        f.setBold(True)
        lbl.setFont(f)
        header.addWidget(lbl)
        header.addStretch()

        self.btn_save = QtWidgets.QPushButton("Save Sweep")
        can_save = (not self.view_only) and (self.stats_tab is not None) and bool(self.sweep_meta)
        self.btn_save.setEnabled(bool(can_save))
        self.btn_save.clicked.connect(self.save_sweep)
        header.addWidget(self.btn_save)

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        header.addWidget(btn_close)
        layout.addLayout(header)

        best_avg = self.best.get("avg", None)
        best_params = self.best.get("params", {})

        if isinstance(best_avg, (int, float)):
            summary_txt = (
                f"Best Average Shots: <b>{float(best_avg):.2f}</b>"
                f"   |   Best Params: <b>{json.dumps(best_params)}</b>"
            )
        else:
            summary_txt = f"Best Params: <b>{json.dumps(best_params)}</b>"

        summary = QtWidgets.QLabel(summary_txt)
        summary.setTextFormat(QtCore.Qt.RichText)
        layout.addWidget(summary)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(split, stretch=1)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Params", "Avg", "Std", "Best", "Worst"])
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        self.table.setRowCount(len(self.results))
        for i, row in enumerate(self.results):
            params = row.get("params", {})
            avg = row.get("avg", 0.0)
            std = row.get("std", 0.0)
            min_s = row.get("min", 0)
            max_s = row.get("max", 0)

            it0 = QtWidgets.QTableWidgetItem(json.dumps(params, sort_keys=True))
            it0.setData(QtCore.Qt.UserRole, row.get("hist", []))
            self.table.setItem(i, 0, it0)

            self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{float(avg):.2f}"))
            self.table.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{float(std):.2f}"))
            self.table.setItem(i, 3, QtWidgets.QTableWidgetItem(str(min_s)))
            self.table.setItem(i, 4, QtWidgets.QTableWidgetItem(str(max_s)))

        self.table.cellClicked.connect(self._on_row_selected)
        left_layout.addWidget(self.table)
        split.addWidget(left)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.addWidget(QtWidgets.QLabel("Histogram (hover for exact counts):"))
        self.graph = StatsGraphWidget([])
        right_layout.addWidget(self.graph, stretch=1)
        split.addWidget(right)

        if self.results:
            self.table.selectRow(0)
            self._on_row_selected(0, 0)

    def _on_row_selected(self, row, col):
        item = self.table.item(row, 0)
        if not item:
            return
        hist = item.data(QtCore.Qt.UserRole)
        if not isinstance(hist, list):
            hist = []
        self.graph.set_counts(hist)

    def save_sweep(self):
        if self.view_only or self.stats_tab is None or not self.sweep_meta:
            return

        default_name = self.sweep_meta.get("name") or f"{self.model_def.get('name', self.model_def.get('key', 'model'))} sweep"
        name, ok = QtWidgets.QInputDialog.getText(self, "Save Sweep", "Name:", text=str(default_name))
        if not ok:
            return
        name = str(name).strip() or default_name

        record = dict(self.sweep_meta)
        record["name"] = name
        record["results"] = self.results
        record["best"] = self.best
        record.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
        record.setdefault("id", datetime.now().strftime("%Y%m%d_%H%M%S"))

        model_key = str(record.get("model_key") or self.model_def.get("key") or "unknown")
        self.stats_tab.add_param_sweep(model_key, record)

        self.btn_save.setEnabled(False)
        self.btn_save.setText("Saved")


class ParamSweepDialog(QtWidgets.QDialog):
    """Configure a parameter sweep for a tunable model.

    Returns a sweep config dict with ranges, mode, budgets, and games per setting.
    """

    MAX_TOTAL_CONFIGS = 20000  # safety cap to avoid accidental massive sweeps

    def __init__(self, model_key: str, model_name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Parameter Sweep: {model_name}")
        self.model_key = model_key
        self.inputs = {}  # key -> (min_sb, max_sb, step_sb)
        self.param_widgets = {}  # key -> (min_sb, max_sb, step_sb, is_int)
        self.sb_games = None
        self.lbl_total = None
        self.mode_combo = None
        self.sb_coarse_factor = None
        self.sb_refine_top = None
        self.sb_max_points = None
        self.sb_time_budget = None
        self.sb_early_stop = None
        self._total_configs = 0
        self._estimated_points = 0
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        specs = PARAM_SPECS.get(self.model_key, [])
        if not specs:
            layout.addWidget(QtWidgets.QLabel("This model has no tunable parameters to sweep."))
            btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
            btns.rejected.connect(self.reject)
            btns.accepted.connect(self.accept)
            layout.addWidget(btns)
            return

        # --- Grid for consistent label/input alignment ---
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        grid.setColumnMinimumWidth(0, 190)
        grid.setColumnMinimumWidth(1, 110)
        grid.setColumnMinimumWidth(2, 110)
        grid.setColumnMinimumWidth(3, 110)
        grid.setColumnMinimumWidth(4, 120)

        r = 0
        lbl_games = QtWidgets.QLabel("Games per setting:")
        self.sb_games = QtWidgets.QSpinBox()
        self.sb_games.setRange(1, 10000)
        self.sb_games.setValue(200)

        grid.addWidget(lbl_games, r, 0)
        grid.addWidget(self.sb_games, r, 1)
        self.lbl_total = QtWidgets.QLabel("")
        self.lbl_total.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        grid.addWidget(self.lbl_total, r, 2, 1, 3)

        r += 1
        lbl_mode = QtWidgets.QLabel("Sweep mode:")
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItem("Standard grid", "grid")
        self.mode_combo.addItem("Coarse → fine", "coarse_to_fine")
        self.mode_combo.currentIndexChanged.connect(self._update_counts)
        grid.addWidget(lbl_mode, r, 0)
        grid.addWidget(self.mode_combo, r, 1)
        mode_hint = QtWidgets.QLabel("Coarse → fine runs a wider first pass, then refines around the best settings.")
        mode_hint.setWordWrap(True)
        mode_hint.setStyleSheet(f"color: {Theme.TEXT_MUTED};")
        grid.addWidget(mode_hint, r, 2, 1, 3)

        r += 1
        lbl_coarse = QtWidgets.QLabel("Coarse factor:")
        lbl_coarse.setToolTip(
            "Run a low-resolution pass first by widening the step size. Higher values sample fewer points."
        )
        self.sb_coarse_factor = QtWidgets.QDoubleSpinBox()
        self.sb_coarse_factor.setRange(1.0, 10.0)
        self.sb_coarse_factor.setSingleStep(0.5)
        self.sb_coarse_factor.setValue(2.0)
        self.sb_coarse_factor.setToolTip(
            "Run a low-resolution pass first by widening the step size. Higher values sample fewer points."
        )
        self.sb_coarse_factor.valueChanged.connect(self._update_counts)
        lbl_refine = QtWidgets.QLabel("Refine top N:")
        lbl_refine.setToolTip(
            "After the coarse pass, refine around the best N settings."
        )
        self.sb_refine_top = QtWidgets.QSpinBox()
        self.sb_refine_top.setRange(1, 10)
        self.sb_refine_top.setValue(1)
        self.sb_refine_top.setToolTip(
            "After the coarse pass, refine around the best N settings."
        )
        self.sb_refine_top.valueChanged.connect(self._update_counts)
        grid.addWidget(lbl_coarse, r, 0)
        grid.addWidget(self.sb_coarse_factor, r, 1)
        grid.addWidget(lbl_refine, r, 2)
        grid.addWidget(self.sb_refine_top, r, 3)

        r += 1
        lbl_max_points = QtWidgets.QLabel("Max points:")
        lbl_max_points.setToolTip(
            "Hard cap on how many parameter combinations to evaluate. 0 means no limit."
        )
        self.sb_max_points = QtWidgets.QSpinBox()
        self.sb_max_points.setRange(0, self.MAX_TOTAL_CONFIGS)
        self.sb_max_points.setValue(0)
        self.sb_max_points.setToolTip(
            "Hard cap on how many parameter combinations to evaluate. 0 means no limit."
        )
        self.sb_max_points.valueChanged.connect(self._update_counts)
        lbl_time = QtWidgets.QLabel("Time budget (sec):")
        lbl_time.setToolTip(
            "Stop the sweep once this many seconds have elapsed. 0 means no limit."
        )
        self.sb_time_budget = QtWidgets.QSpinBox()
        self.sb_time_budget.setRange(0, 36000)
        self.sb_time_budget.setValue(0)
        self.sb_time_budget.setToolTip(
            "Stop the sweep once this many seconds have elapsed. 0 means no limit."
        )
        self.sb_time_budget.valueChanged.connect(self._update_counts)
        grid.addWidget(lbl_max_points, r, 0)
        grid.addWidget(self.sb_max_points, r, 1)
        grid.addWidget(lbl_time, r, 2)
        grid.addWidget(self.sb_time_budget, r, 3)

        r += 1
        lbl_early = QtWidgets.QLabel("Early stop (no-improve points):")
        lbl_early.setToolTip(
            "Stop after this many consecutive points without improvement. 0 disables."
        )
        self.sb_early_stop = QtWidgets.QSpinBox()
        self.sb_early_stop.setRange(0, self.MAX_TOTAL_CONFIGS)
        self.sb_early_stop.setValue(0)
        self.sb_early_stop.setToolTip(
            "Stop after this many consecutive points without improvement. 0 disables."
        )
        self.sb_early_stop.valueChanged.connect(self._update_counts)
        grid.addWidget(lbl_early, r, 0)
        grid.addWidget(self.sb_early_stop, r, 1)

        r += 1
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setStyleSheet(f"color: {Theme.TEXT_LABEL};")
        grid.addWidget(line, r, 0, 1, 5)

        r += 1
        # Header row
        grid.addWidget(QtWidgets.QLabel(""), r, 0)
        hdr_min = QtWidgets.QLabel("Min")
        hdr_max = QtWidgets.QLabel("Max")
        hdr_step = QtWidgets.QLabel("Step")
        hdr_cnt = QtWidgets.QLabel("Configs")
        for w in (hdr_min, hdr_max, hdr_step, hdr_cnt):
            w.setStyleSheet(f"color: {Theme.TEXT_LABEL}; font-weight: bold;")
        grid.addWidget(hdr_min, r, 1)
        grid.addWidget(hdr_max, r, 2)
        grid.addWidget(hdr_step, r, 3)
        grid.addWidget(hdr_cnt, r, 4)

        # Param rows
        for spec in specs:
            r += 1
            key = spec["key"]
            label = spec.get("label", key)

            lbl = QtWidgets.QLabel(label)
            lbl.setStyleSheet(f"color: {Theme.TEXT_LABEL};")
            grid.addWidget(lbl, r, 0)

            # Use int spinboxes when values look integral and the step is integral.
            # Otherwise use double spinboxes.
            is_int = (
                float(spec.get("min", 0)).is_integer()
                and float(spec.get("max", 0)).is_integer()
                and float(spec.get("step", 1)).is_integer()
            )

            if is_int:
                sb_min = QtWidgets.QSpinBox()
                sb_max = QtWidgets.QSpinBox()
                sb_step = QtWidgets.QSpinBox()
                sb_min.setRange(int(spec["min"]), int(spec["max"]))
                sb_max.setRange(int(spec["min"]), int(spec["max"]))
                sb_step.setRange(1, max(1, int(spec["max"]) - int(spec["min"])))
                sb_min.setValue(int(spec["min"]))
                sb_max.setValue(int(spec["max"]))
                sb_step.setValue(max(1, int(spec.get("step", 1))))
            else:
                sb_min = QtWidgets.QDoubleSpinBox()
                sb_max = QtWidgets.QDoubleSpinBox()
                sb_step = QtWidgets.QDoubleSpinBox()
                for sb in (sb_min, sb_max, sb_step):
                    sb.setDecimals(6)
                sb_min.setRange(float(spec["min"]), float(spec["max"]))
                sb_max.setRange(float(spec["min"]), float(spec["max"]))
                sb_step.setRange(1e-9, max(1e-9, float(spec["max"]) - float(spec["min"])))
                sb_min.setValue(float(spec["min"]))
                sb_max.setValue(float(spec["max"]))
                sb_step.setValue(float(spec.get("step", 0.1)))

            sb_min.valueChanged.connect(self._update_counts)
            sb_max.valueChanged.connect(self._update_counts)
            sb_step.valueChanged.connect(self._update_counts)

            grid.addWidget(sb_min, r, 1)
            grid.addWidget(sb_max, r, 2)
            grid.addWidget(sb_step, r, 3)

            lbl_count = QtWidgets.QLabel("0")
            lbl_count.setAlignment(QtCore.Qt.AlignCenter)
            grid.addWidget(lbl_count, r, 4)

            self.inputs[key] = (sb_min, sb_max, sb_step, lbl_count, is_int)
            self.param_widgets[key] = (sb_min, sb_max, sb_step, is_int)

        layout.addLayout(grid)

        hint = QtWidgets.QLabel(
            "Tip: keep total configs under the cap to avoid long runs. "
            "You can increase Games per setting once you narrow in."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {Theme.TEXT_LABEL}; margin-top: 8px;")
        layout.addWidget(hint)

        layout.addStretch()

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        self._update_counts()

    @staticmethod
    def _count_for_range(min_v: float, max_v: float, step: float) -> int:
        """Count how many values will be generated for the sweep (inclusive endpoints)."""
        if step <= 0:
            return 0
        lo = min(min_v, max_v)
        hi = max(min_v, max_v)
        # Inclusive count with some float tolerance
        n = int(math.floor((hi - lo) / step + 1e-12)) + 1
        return max(0, n)

    @staticmethod
    def _sync_limits(self, key: str) -> None:
        widgets = self.param_widgets.get(key)
        if not widgets:
            return
        sb_min, sb_max, sb_step = widgets

        if sb_min.value() > sb_max.value():
            sb_max.setValue(sb_min.value())

        sb_max.setMinimum(sb_min.value())
        sb_min.setMaximum(sb_max.value())

        span = max(0.0, sb_max.value() - sb_min.value())
        if span <= 0:
            return
        sb_step.setMaximum(max(sb_step.minimum(), span))
        if sb_step.value() > span:
            sb_step.setValue(span)

    def _values_for_range(self, min_v: float, max_v: float, step: float, is_int: bool) -> list:
        if step <= 0:
            return []
        lo = min(min_v, max_v)
        hi = max(min_v, max_v)

        if is_int:
            lo_i = int(round(lo))
            hi_i = int(round(hi))
            st_i = max(1, int(round(step)))
            return list(range(lo_i, hi_i + 1, st_i))

        # Float stepping: cap length for safety
        out = []
        v = lo
        cap = 5000
        # Add a tiny epsilon to include hi when within tolerance
        while v <= hi + 1e-12 and len(out) < cap:
            out.append(float(round(v, 10)))
            v += step
        return out

    def _update_counts(self):
        total = 1
        coarse_total = 1
        refine_total = 1
        coarse_factor = float(self.sb_coarse_factor.value()) if self.sb_coarse_factor is not None else 2.0
        refine_top = int(self.sb_refine_top.value()) if self.sb_refine_top is not None else 1

        mode = "grid"
        if self.mode_combo is not None:
            mode = str(self.mode_combo.currentData() or "grid")

        # Update per-param counts
        for key, (sb_min, sb_max, sb_step, lbl_count, is_int) in self.inputs.items():
            min_v = float(sb_min.value())
            max_v = float(sb_max.value())
            step = float(sb_step.value())
            n = _count_values_for_range(min_v, max_v, step)
            lbl_count.setText(str(n))
            total *= max(1, n)

            if mode == "coarse_to_fine":
                if is_int:
                    coarse_step = max(1.0, float(int(round(step * coarse_factor))))
                else:
                    coarse_step = max(step, step * coarse_factor)
                coarse_n = _count_values_for_range(min_v, max_v, coarse_step)
                coarse_total *= max(1, coarse_n)

                span = min(max_v - min_v, 2.0 * coarse_step)
                refine_n = _count_values_for_range(0.0, max(0.0, span), step)
                refine_total *= max(1, refine_n)

        if mode == "coarse_to_fine":
            est_total = coarse_total + (max(1, refine_top) * refine_total)
        else:
            est_total = total

        self._total_configs = total
        self._estimated_points = est_total

        if self.sb_coarse_factor is not None:
            self.sb_coarse_factor.setEnabled(mode == "coarse_to_fine")
        if self.sb_refine_top is not None:
            self.sb_refine_top.setEnabled(mode == "coarse_to_fine")

        if self.lbl_total is not None:
            if est_total > self.MAX_TOTAL_CONFIGS:
                self.lbl_total.setText(
                    f"Estimated configs: {est_total:,} (over cap {self.MAX_TOTAL_CONFIGS:,})"
                )
                self.lbl_total.setStyleSheet("color: #ff6666; font-weight: bold;")
            else:
                if mode == "coarse_to_fine":
                    self.lbl_total.setText(f"Estimated configs: {est_total:,} (coarse {coarse_total:,})")
                else:
                    self.lbl_total.setText(f"Total configs: {est_total:,}")
                self.lbl_total.setStyleSheet(f"color: {Theme.HIGHLIGHT}; font-weight: bold;")

    def accept(self):
        try:
            self._accept_impl()
        except Exception:
            import traceback as _tb
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("Sweep Config Error")
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("An error occurred while validating the sweep settings.")
            msg.setDetailedText(_tb.format_exc())
            msg.exec_()

    def _accept_impl(self):
        # Recompute (in case user never changed focus)
        self._update_counts()

        if self._estimated_points > self.MAX_TOTAL_CONFIGS:
            QtWidgets.QMessageBox.warning(
                self,
                "Sweep Too Large",
                f"This sweep would run about {self._estimated_points:,} configurations.\n\nPlease reduce the ranges or increase the step size so the total is <= {self.MAX_TOTAL_CONFIGS:,}.",
            )
            return

        super().accept()

    def get_config(self):
        games = int(self.sb_games.value()) if self.sb_games is not None else 200
        mode = "grid"
        if self.mode_combo is not None:
            mode = str(self.mode_combo.currentData() or "grid")
        coarse_factor = float(self.sb_coarse_factor.value()) if self.sb_coarse_factor is not None else 2.0
        refine_top = int(self.sb_refine_top.value()) if self.sb_refine_top is not None else 1
        max_points = int(self.sb_max_points.value()) if self.sb_max_points is not None else 0
        time_budget_s = float(self.sb_time_budget.value()) if self.sb_time_budget is not None else 0.0
        early_stop = int(self.sb_early_stop.value()) if self.sb_early_stop is not None else 0

        params = []
        for key, (sb_min, sb_max, sb_step, _lbl_count, is_int) in self.inputs.items():
            params.append(
                {
                    "key": str(key),
                    "min": float(sb_min.value()),
                    "max": float(sb_max.value()),
                    "step": float(sb_step.value()),
                    "is_int": bool(is_int),
                }
            )

        return {
            "games_per": games,
            "params": params,
            "mode": mode,
            "coarse_factor": coarse_factor,
            "refine_top_n": refine_top,
            "max_points": max_points,
            "time_budget_s": time_budget_s,
            "early_stop_no_improve": early_stop,
            "estimated_points": int(self._estimated_points),
        }


class ParamTestDialog(QtWidgets.QDialog):
    def __init__(self, model_key: str, model_name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Tune: {model_name}")
        self.model_key = model_key
        self.inputs: Dict[str, QtWidgets.QDoubleSpinBox] = {}
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Games input
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Games to Run:"))
        self.sb_games = QtWidgets.QSpinBox()
        self.sb_games.setRange(1, 10000)
        self.sb_games.setValue(200)
        row.addWidget(self.sb_games)
        layout.addLayout(row)

        layout.addSpacing(10)
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setStyleSheet(f"color: {Theme.TEXT_MUTED};")
        layout.addWidget(line)
        layout.addSpacing(10)

        specs = PARAM_SPECS.get(self.model_key, [])

        if specs:
            for spec in specs:
                row = QtWidgets.QHBoxLayout()
                row.addWidget(QtWidgets.QLabel(spec["label"]))
                sb = QtWidgets.QDoubleSpinBox()
                sb.setRange(float(spec["min"]), float(spec["max"]))
                sb.setSingleStep(float(spec["step"]))
                sb.setValue(float(spec["default"]))
                sb.setDecimals(3)
                row.addWidget(sb)
                layout.addLayout(row)
                self.inputs[spec["key"]] = sb
        else:
            layout.addWidget(QtWidgets.QLabel("No tunable parameters for this model."))
            layout.addWidget(QtWidgets.QLabel("(Running a custom batch with default behavior.)"))

        layout.addStretch()

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_config(self) -> Tuple[int, Dict[str, float]]:
        p: Dict[str, float] = {k: sb.value() for k, sb in self.inputs.items()}
        return int(self.sb_games.value()), p


class ModelDetailDialog(QtWidgets.QDialog):
    def __init__(self, model_def: Dict, stats: Dict[str, object], placements,
                 stats_tab: Optional['ModelStatsTab'] = None, parent=None):
        super().__init__(parent)
        self.model_def = model_def
        self.stats = stats
        self.placements = placements
        self.stats_tab = stats_tab
        self.worker: Optional[CustomSimWorker] = None
        self.progress: Optional[QtWidgets.QProgressDialog] = None
        self._sim_thread = None
        self._sim_worker = None
        self._closing_after_workers = False
        self._model_key = str(model_def.get("key") or "")

        self.setWindowTitle(f"Analysis: {model_def.get('name', model_def.get('key', 'Model'))}")
        self.resize(900, 650)
        self._build_ui()

    def _build_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)

        override_name = ""
        override_notes = ""
        try:
            if self.stats_tab is not None and self._model_key:
                override = self.stats_tab.model_overrides.get(self._model_key, {})
                if isinstance(override, dict):
                    override_name = str(override.get("name") or "")
                    override_notes = str(override.get("notes") or "")
        except Exception:
            pass

        # Header (editable)
        title_text = override_name.strip() or self.model_def.get("name", self.model_def.get("key", "Model"))
        self.title_input = QtWidgets.QLineEdit(title_text)
        f = self.title_input.font()
        f.setPointSize(16)
        f.setBold(True)
        self.title_input.setFont(f)
        self.title_input.setStyleSheet(
            f"background-color: {Theme.BG_PANEL}; border: 1px solid {Theme.BG_BUTTON}; color: {Theme.TEXT_MAIN};"
        )
        main_layout.addWidget(self.title_input)

        # Notes (editable)
        notes_label = QtWidgets.QLabel("Model Notes:")
        notes_label.setStyleSheet(f"color: {Theme.TEXT_MUTED}; font-weight: bold; margin-top: 10px;")
        main_layout.addWidget(notes_label)

        default_notes = self.model_def.get("notes") or self.model_def.get("description") or ""
        notes_text = override_notes.strip() or str(default_notes)

        self.notes_edit = QtWidgets.QTextEdit()
        self.notes_edit.setPlainText(notes_text)
        self.notes_edit.setMaximumHeight(140)
        self.notes_edit.setStyleSheet(
            f"background-color: {Theme.BG_PANEL}; border: 1px solid {Theme.BG_BUTTON}; color: {Theme.TEXT_MAIN};"
        )
        main_layout.addWidget(self.notes_edit)

        # Stats & Graph
        mid_layout = QtWidgets.QHBoxLayout()

        # Left: metrics
        stats_panel = QtWidgets.QWidget()
        stats_panel.setFixedWidth(260)
        sp_layout = QtWidgets.QVBoxLayout(stats_panel)

        metrics = _compute_basic_stats(self.stats)
        games = int(metrics.get("games", 0))
        mean = float(metrics.get("mean", 0.0))
        std = float(metrics.get("std", 0.0))
        median = float(metrics.get("median", 0.0))
        p90 = float(metrics.get("p90", 0.0))
        p95 = float(metrics.get("p95", 0.0))

        metrics_rows = [
            ("Total Games", f"{games}"),
            ("Avg Shots", f"{mean:.2f}" if games > 0 else "-"),
            ("Median", f"{median:.0f}" if games > 0 else "-"),
            ("Std Dev", f"{std:.2f}" if games > 0 else "-"),
            ("P90", f"{p90:.0f}" if games > 0 else "-"),
            ("P95", f"{p95:.0f}" if games > 0 else "-"),
            ("Best Game", f"{self.stats.get('min_shots', '-')}"),
            ("Worst Game", f"{self.stats.get('max_shots', '-')}"),
        ]

        for phase in PHASES:
            avg_phase = _compute_phase_avg(self.stats, phase)
            metrics_rows.append((f"{phase} Avg", f"{avg_phase:.2f}" if avg_phase is not None else "-"))

        for label, val in metrics_rows:
            row = QtWidgets.QHBoxLayout()
            lbl = QtWidgets.QLabel(label)
            lbl.setStyleSheet(f"color: {Theme.TEXT_MUTED}; font-weight: bold;")
            val_lbl = QtWidgets.QLabel(val)
            val_lbl.setAlignment(QtCore.Qt.AlignRight)
            val_lbl.setStyleSheet(f"color: {Theme.HIGHLIGHT}; font-weight: bold;")
            row.addWidget(lbl)
            row.addWidget(val_lbl)
            sp_layout.addLayout(row)

        sp_layout.addStretch()

        self.btn_custom = QtWidgets.QPushButton("Run Custom Sim")
        self.btn_custom.setStyleSheet(
            f"background-color: {Theme.HIGHLIGHT}; color: #000; font-weight: bold; padding: 8px;"
        )
        self.btn_custom.clicked.connect(self.run_custom_sim)
        sp_layout.addWidget(self.btn_custom)

        sweep_specs = PARAM_SPECS.get(self.model_def.get("key", ""), [])
        gb_sweep = QtWidgets.QGroupBox("Parameter Sweep")
        gb_sweep.setEnabled(bool(sweep_specs))
        gb_sweep.setStyleSheet(
            f"""
QGroupBox {{ color: {Theme.TEXT_LABEL}; }}
QGroupBox:disabled {{ color: {Theme.TEXT_MUTED}; }}
QLabel {{ color: {Theme.TEXT_LABEL}; }}
QLabel:disabled {{ color: {Theme.TEXT_MUTED}; }}
QPushButton {{ background-color: {Theme.BG_BUTTON}; color: {Theme.TEXT_MAIN}; padding: 8px; font-weight: bold; }}
QPushButton:disabled {{ background-color: {Theme.BG_PANEL}; color: {Theme.TEXT_MUTED}; font-weight: normal; }}
"""
        )
        gb_layout = QtWidgets.QVBoxLayout(gb_sweep)

        hint = QtWidgets.QLabel("Run a grid sweep over tunable parameters and compare average shots.")
        hint.setWordWrap(True)
        gb_layout.addWidget(hint)

        btn_sweep = QtWidgets.QPushButton("Run Parameter Sweep")
        btn_sweep.setEnabled(bool(sweep_specs))
        btn_sweep.clicked.connect(self.run_param_sweep)
        gb_layout.addWidget(btn_sweep)

        btn_saved = QtWidgets.QPushButton("View Saved Sweeps")
        has_saved = False
        try:
            if self.stats_tab is not None:
                has_saved = bool(self.stats_tab.get_param_sweeps(self.model_def.get("key", "")))
        except Exception:
            has_saved = False
        btn_saved.setEnabled(has_saved)
        btn_saved.clicked.connect(self.open_saved_sweeps)
        gb_layout.addWidget(btn_saved)
        self.btn_saved_sweeps = btn_saved

        sp_layout.addWidget(gb_sweep)
        self.btn_sweep = btn_sweep

        mid_layout.addWidget(stats_panel)

        # Right: histogram graph
        self.graph = StatsGraphWidget(self.stats.get("hist", []))
        mid_layout.addWidget(self.graph, stretch=1)

        main_layout.addLayout(mid_layout)

        footer = QtWidgets.QHBoxLayout()
        self.save_overrides_btn = QtWidgets.QPushButton("Save")
        self.save_overrides_btn.clicked.connect(self._save_overrides)
        footer.addWidget(self.save_overrides_btn)
        footer.addStretch(1)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        footer.addWidget(close_btn)
        main_layout.addLayout(footer)

    def _save_overrides(self) -> None:
        if self.stats_tab is None or not self._model_key:
            return
        name = self.title_input.text() if hasattr(self, "title_input") else ""
        notes = self.notes_edit.toPlainText() if hasattr(self, "notes_edit") else ""
        self.stats_tab.update_model_override(self._model_key, name, notes)
        title_name = name.strip() or self.model_def.get("name") or self._model_key
        self.setWindowTitle(f"Analysis: {title_name}")

    def accept(self):
        try:
            self._save_overrides()
        except Exception:
            pass
        super().accept()

    def _refresh_saved_sweeps_button(self):
        if not hasattr(self, "btn_saved_sweeps"):
            return
        has_saved = False
        try:
            if self.stats_tab is not None:
                has_saved = bool(self.stats_tab.get_param_sweeps(self.model_def.get("key", "")))
        except Exception:
            has_saved = False
        self.btn_saved_sweeps.setEnabled(has_saved)

    def open_saved_sweeps(self):
        if self.stats_tab is None:
            return
        sweeps = self.stats_tab.get_param_sweeps(self.model_def.get("key", ""))
        dlg = SavedSweepsDialog(self.model_def, sweeps, self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        selected = dlg.selected_sweep()
        if not selected:
            return
        results = selected.get("results", []) or []
        best = selected.get("best", {}) or {}
        sr = SweepResultsDialog(
            results=results,
            best=best,
            model_def=self.model_def,
            sweep_meta=selected,
            stats_tab=self.stats_tab,
            parent=self,
            view_only=True,
        )
        sr.exec_()

    def _safe_progress_set(self, dlg, value: int) -> None:
        """Safely set QProgressDialog value (guards against deleted Qt objects)."""
        try:
            if dlg is None:
                return
            dlg.setValue(int(value))
        except Exception:
            return

    def _safe_progress_close(self, dlg) -> None:
        """Safely close a QProgressDialog/QProgressBar wrapper.

        Note: Closing a QProgressDialog can sometimes emit 'canceled'. We block signals while
        closing so a normal completion isn't mistaken for a user cancel.
        """
        try:
            if dlg is None:
                return
            try:
                prev = dlg.blockSignals(True)
            except Exception:
                prev = None
            try:
                try:
                    dlg.hide()
                except Exception:
                    pass
                try:
                    dlg.close()
                except Exception:
                    pass
            finally:
                try:
                    if prev is not None:
                        dlg.blockSignals(prev)
                except Exception:
                    pass
        except Exception:
            return

    def _on_custom_progress(self, value: int) -> None:
        self._safe_progress_set(getattr(self, "progress", None), value)

    def _on_sweep_progress(self, value: int) -> None:
        self._safe_progress_set(getattr(self, "sweep_progress", None), value)

    def run_custom_sim(self):
        dlg = ParamTestDialog(self.model_def["key"], self.model_def.get("name", self.model_def["key"]), self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        count, params = dlg.get_config()

        # Prevent multiple concurrent runs
        if getattr(self, "worker", None) is not None and self.worker.isRunning():
            return
        if hasattr(self, "btn_custom"):
            self.btn_custom.setEnabled(False)

        self.progress = QtWidgets.QProgressDialog("Running Experiment...", "Cancel", 0, count, self)
        self.progress.setWindowModality(QtCore.Qt.WindowModal)
        self.progress.setMinimumDuration(0)
        self.progress.setValue(0)
        self.progress.setAutoClose(False)
        self.progress.setAutoReset(False)

        self.worker = CustomSimWorker(
            self.model_def["key"],
            self.placements,
            self.stats_tab.ship_ids,
            self.stats_tab.board_size,
            count,
            params,
        )
        self.worker.progress.connect(self._on_custom_progress)
        self.worker.result.connect(self.on_sim_finished)
        self.worker.error.connect(self.on_sim_error)
        self.worker.finished.connect(self._on_custom_thread_finished)
        self.progress.canceled.connect(self.worker.cancel)

        self.worker.start()

    def on_sim_finished(self, avg: float, std: float, count: int):
        """Handle completion of a custom experiment run."""
        try:
            # Close progress and re-enable UI safely
            try:
                self._safe_progress_close(getattr(self, "progress", None))
            except Exception:
                pass
            self.progress = None
            try:
                if hasattr(self, "btn_custom"):
                    self.btn_custom.setEnabled(True)
            except Exception:
                pass

            if count <= 0:
                return

            params = {}
            try:
                w = getattr(self, "worker", None) or getattr(self, "custom_worker", None)
                if w is not None:
                    params = dict(getattr(w, "params", {}) or {})
            except Exception:
                params = {}

            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("Experiment Results")
            msg.setText(
                f"<b>Model:</b> {self.model_def.get('name', self.model_def['key'])}<br>"
                f"<b>Params:</b> {json.dumps(params)}<br><br>"
                f"<b>Games Ran:</b> {count}<br>"
                f"<b>Average:</b> {avg:.2f}<br>"
                f"<b>Std Dev:</b> {std:.2f}"
            )
            msg.exec_()

            # If the user tried to close while workers were running, finish closing once they're done
            if getattr(self, "_closing_after_workers", False):
                try:
                    sweep_worker = getattr(self, "sweep_worker", None)
                    custom_worker = getattr(self, "worker", None) or getattr(self, "custom_worker", None)
                    sweep_running = bool(sweep_worker) and hasattr(sweep_worker, "isRunning") and sweep_worker.isRunning()
                    custom_running = bool(custom_worker) and hasattr(custom_worker, "isRunning") and custom_worker.isRunning()
                    if not (sweep_running or custom_running):
                        self._closing_after_workers = False
                        self.accept()
                except Exception:
                    pass
        except Exception as e:
            import traceback
            traceback.print_exc()
            try:
                QtWidgets.QMessageBox.critical(self, "Error", "Sim finished handler crashed: " + str(e))
            except Exception:
                pass

    def on_sim_error(self, msg: str) -> None:
        """Handle errors raised by the custom experiment worker."""
        try:
            self._safe_progress_close(getattr(self, "progress", None))
        except Exception:
            pass
        self.progress = None
        try:
            if hasattr(self, "btn_custom"):
                self.btn_custom.setEnabled(True)
        except Exception:
            pass
        QtWidgets.QMessageBox.critical(self, "Experiment error", msg)

        # If the user tried to close while workers were running, finish closing once they're done
        if getattr(self, "_closing_after_workers", False):
            try:
                sweep_worker = getattr(self, "sweep_worker", None)
                custom_worker = getattr(self, "worker", None) or getattr(self, "custom_worker", None)
                sweep_running = bool(sweep_worker) and hasattr(sweep_worker, "isRunning") and sweep_worker.isRunning()
                custom_running = bool(custom_worker) and hasattr(custom_worker, "isRunning") and custom_worker.isRunning()
                if not (sweep_running or custom_running):
                    self._closing_after_workers = False
                    self.accept()
            except Exception:
                pass

    def _on_custom_thread_finished(self):
        """Cleanup after the CustomSimWorker thread has fully stopped."""
        try:
            # Only clear references after the thread is actually finished
            if getattr(self, "worker", None) is not None and hasattr(self.worker, "isRunning") and not self.worker.isRunning():
                self.worker = None
        except Exception:
            pass
        try:
            if hasattr(self, "btn_custom"):
                self.btn_custom.setEnabled(True)
        except Exception:
            pass
        try:
            # If progress dialog somehow lingered, close it.
            self._safe_progress_close(getattr(self, "progress", None))
        except Exception:
            pass
        self.progress = None

        # If the user tried to close while workers were running, finish closing once they're done
        if getattr(self, "_closing_after_workers", False):
            try:
                sweep_worker = getattr(self, "sweep_worker", None)
                custom_worker = getattr(self, "worker", None) or getattr(self, "custom_worker", None)
                sweep_running = bool(sweep_worker) and hasattr(sweep_worker, "isRunning") and sweep_worker.isRunning()
                custom_running = bool(custom_worker) and hasattr(custom_worker, "isRunning") and custom_worker.isRunning()
                if not (sweep_running or custom_running):
                    self._closing_after_workers = False
                    self.accept()
            except Exception:
                pass

    def run_param_sweep(self):
        try:
            self._run_param_sweep_impl()
        except Exception:
            import traceback as _tb
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("Sweep Error")
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Failed to start the parameter sweep.")
            msg.setDetailedText(_tb.format_exc())
            msg.exec_()

    def _run_param_sweep_impl(self):
        # Prevent multiple concurrent sweeps
        if getattr(self, "sweep_worker", None) is not None and self.sweep_worker.isRunning():
            return
        specs = PARAM_SPECS.get(self.model_def.get("key", ""), [])
        if not specs:
            QtWidgets.QMessageBox.information(self, "No Parameters", "This model has no tunable parameters to sweep.")
            return

        dlg = ParamSweepDialog(self.model_def["key"], self.model_def["name"], self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        sweep_config = dlg.get_config()
        games_per = int(sweep_config.get("games_per", 200))
        self._last_sweep_games_per = int(games_per)
        if hasattr(self, "btn_sweep"):
            self.btn_sweep.setEnabled(False)
        # Track whether the user requested cancellation for this sweep
        self._sweep_cancel_requested = False
        ranges = {}
        try:
            for p in sweep_config.get("params", []):
                if not isinstance(p, dict):
                    continue
                key = str(p.get("key"))
                ranges[key] = {
                    "min": float(p.get("min", 0.0)),
                    "max": float(p.get("max", 0.0)),
                    "step": float(p.get("step", 0.0)),
                }
        except Exception:
            ranges = {}

        estimated_points = int(sweep_config.get("estimated_points") or 0)
        self._last_sweep_meta = {
            "id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "model_key": self.model_def.get("key"),
            "model_name": self.model_def.get("name"),
            "games_per_point": int(games_per),
            "grid_size": int(estimated_points),
            "ranges": ranges,
            "mode": sweep_config.get("mode", "grid"),
            "coarse_factor": sweep_config.get("coarse_factor"),
            "refine_top_n": sweep_config.get("refine_top_n"),
            "max_points": sweep_config.get("max_points"),
            "time_budget_s": sweep_config.get("time_budget_s"),
            "early_stop_no_improve": sweep_config.get("early_stop_no_improve"),
        }
        if estimated_points <= 0:
            return

        max_points = int(sweep_config.get("max_points") or 0)
        total_cfg = estimated_points
        if max_points > 0:
            total_cfg = min(total_cfg, max_points)
        self.sweep_progress = QtWidgets.QProgressDialog("Sweeping parameters...", "Cancel", 0, total_cfg, self)
        self.sweep_progress.setWindowModality(QtCore.Qt.WindowModal)
        self.sweep_progress.setMinimumDuration(0)
        self.sweep_progress.setValue(0)
        self.sweep_progress.setAutoClose(False)
        self.sweep_progress.setAutoReset(False)

        self.sweep_worker = ParamSweepWorker(
            self.model_def["key"],
            self.placements,
            self.stats_tab.ship_ids,
            self.stats_tab.board_size,
            games_per,
            sweep_config,
        )
        self.sweep_worker.progress.connect(self._on_sweep_progress)
        self.sweep_worker.result.connect(self.on_sweep_finished)
        self.sweep_worker.error.connect(self.on_sweep_error)
        self.sweep_worker.finished.connect(self._on_sweep_thread_finished)
        self.sweep_progress.canceled.connect(self._request_cancel_sweep)

        self.sweep_worker.start()

    def on_sweep_finished(self, results, ran: int):
        """Handle completion of a parameter sweep."""
        debug_event(self, "Sweep finished",
                    f"ran={ran} results={len(results) if results else 0} cancel={getattr(self, '_sweep_cancel_requested', False)}")

        try:
            worker = getattr(self, "sweep_worker", None)
            meta = getattr(self, "_last_sweep_meta", None)
            if worker is not None and isinstance(meta, dict):
                meta["points_completed"] = int(getattr(worker, "points_completed", ran))
                meta["stop_reason"] = getattr(worker, "stop_reason", None)
                meta["elapsed_seconds"] = float(getattr(worker, "elapsed_seconds", 0.0))
        except Exception:
            pass

        try:
            # Close progress and re-enable UI safely
            try:
                self._safe_progress_close(getattr(self, "sweep_progress", None))
            except Exception:
                pass
            self.sweep_progress = None
            try:
                if hasattr(self, "btn_sweep"):
                    self.btn_sweep.setEnabled(True)
            except Exception:
                pass

            # If the user cancelled, stop here (do not pop more dialogs).
            if getattr(self, "_sweep_cancel_requested", False):
                debug_event(self, "Sweep finished", "Skipping results dialog because cancel flag is set.",
                            force_popup=True, level="warning")
                return

            # User may have cancelled immediately or produced no valid points
            if ran <= 0 or not results:
                debug_event(self, "Sweep finished", "No results to show (ran<=0 or empty results).", force_popup=True,
                            level="warning")
                return

            # Choose the best setting by lowest average shots (tie-breaker: std dev)
            best = {}
            try:
                candidates = [r for r in results if isinstance(r, dict)]

                def _key(r):
                    try:
                        avg = float(r.get("avg", math.inf))
                        std = float(r.get("std", math.inf))
                        if math.isnan(avg):
                            avg = math.inf
                        if math.isnan(std):
                            std = math.inf
                        return (avg, std)
                    except Exception:
                        return (math.inf, math.inf)

                if candidates:
                    best = min(candidates, key=_key)
            except Exception:
                best = {}

            debug_event(self, "Sweep results", "Opening SweepResultsDialog.")
            dlg = SweepResultsDialog(
                results=results,
                best=best,
                model_def=getattr(self, "model_def", None),
                sweep_meta=getattr(self, "_last_sweep_meta", None),
                stats_tab=getattr(self, "stats_tab", None),
                parent=self,
                view_only=False,
            )
            dlg.exec_()
            try:
                self._refresh_saved_sweeps_button()
            except Exception:
                pass

            # If the user tried to close while workers were running, finish closing once they're done
            if getattr(self, "_closing_after_workers", False):
                try:
                    sweep_worker = getattr(self, "sweep_worker", None)
                    custom_worker = getattr(self, "worker", None) or getattr(self, "custom_worker", None)
                    sweep_running = bool(sweep_worker) and hasattr(sweep_worker, "isRunning") and sweep_worker.isRunning()
                    custom_running = bool(custom_worker) and hasattr(custom_worker, "isRunning") and custom_worker.isRunning()
                    if not (sweep_running or custom_running):
                        self._closing_after_workers = False
                        self.accept()
                except Exception:
                    pass
        except Exception as e:
            import traceback
            traceback.print_exc()
            try:
                QtWidgets.QMessageBox.critical(self, "Error", "Sweep finished handler crashed: " + str(e))
            except Exception:
                pass
        finally:
            # reset cancel flag for next run
            try:
                self._sweep_cancel_requested = False
            except Exception:
                pass

    def on_sweep_error(self, msg: str):
        """Handle errors raised by the sweep worker."""
        debug_event(self, "Sweep error", msg, level="error")

        try:
            self._safe_progress_close(getattr(self, "sweep_progress", None))
        except Exception:
            pass
        self.sweep_progress = None
        try:
            if hasattr(self, "btn_sweep"):
                self.btn_sweep.setEnabled(True)
        except Exception:
            pass
        try:
            self._sweep_cancel_requested = False
        except Exception:
            pass
        QtWidgets.QMessageBox.critical(self, "Sweep error", msg)

        # If the user tried to close while workers were running, finish closing once they're done
        if getattr(self, "_closing_after_workers", False):
            try:
                sweep_worker = getattr(self, "sweep_worker", None)
                custom_worker = getattr(self, "worker", None) or getattr(self, "custom_worker", None)
                sweep_running = bool(sweep_worker) and hasattr(sweep_worker, "isRunning") and sweep_worker.isRunning()
                custom_running = bool(custom_worker) and hasattr(custom_worker, "isRunning") and custom_worker.isRunning()
                if not (sweep_running or custom_running):
                    self._closing_after_workers = False
                    self.accept()
            except Exception:
                pass

    def _on_sweep_thread_finished(self):
        """Cleanup after the ParamSweepWorker thread has fully stopped."""
        try:
            w = getattr(self, "sweep_worker", None)
            if w is not None and hasattr(w, "isRunning") and not w.isRunning():
                self.sweep_worker = None
        except Exception:
            pass
        try:
            if hasattr(self, "btn_sweep"):
                self.btn_sweep.setEnabled(True)
        except Exception:
            pass
        try:
            self._safe_progress_close(getattr(self, "sweep_progress", None))
        except Exception:
            pass
        self.sweep_progress = None

        # If the user tried to close while workers were running, finish closing once they're done
        if getattr(self, "_closing_after_workers", False):
            try:
                sweep_worker = getattr(self, "sweep_worker", None)
                custom_worker = getattr(self, "worker", None) or getattr(self, "custom_worker", None)
                sweep_running = bool(sweep_worker) and hasattr(sweep_worker, "isRunning") and sweep_worker.isRunning()
                custom_running = bool(custom_worker) and hasattr(custom_worker, "isRunning") and custom_worker.isRunning()
                if not (sweep_running or custom_running):
                    self._closing_after_workers = False
                    self.accept()
            except Exception:
                pass

    def _cancel_running_workers(self):
        """Request cancellation for running worker threads without dropping references prematurely."""
        for w_attr, p_attr in (
                ("sweep_worker", "sweep_progress"),
                ("worker", "progress"),
                ("custom_worker", "progress"),
        ):
            worker = getattr(self, w_attr, None)
            prog = getattr(self, p_attr, None)
            if worker is None:
                continue
            try:
                if hasattr(worker, "isRunning") and worker.isRunning() and hasattr(worker, "cancel"):
                    worker.cancel()
                    worker.wait(1500)
            except Exception:
                pass

            try:
                still_running = hasattr(worker, "isRunning") and worker.isRunning()
            except Exception:
                still_running = True
            if still_running:
                continue

            try:
                self._safe_progress_close(prog)
            except Exception:
                pass
            try:
                setattr(self, w_attr, None)
            except Exception:
                pass
            try:
                setattr(self, p_attr, None)
            except Exception:
                pass

    def _request_cancel_sweep(self):
        """Handle sweep cancellation from the progress dialog safely."""
        self._sweep_cancel_requested = True
        try:
            w = getattr(self, "sweep_worker", None)
            if w is not None and hasattr(w, "cancel"):
                w.cancel()
        except Exception:
            import traceback
            traceback.print_exc()
        try:
            p = getattr(self, "sweep_progress", None)
            if p is not None:
                btn = p.cancelButton()
                if btn is not None:
                    btn.setEnabled(False)
                p.setLabelText("Canceling… finishing current game")
        except Exception:
            pass

    def closeEvent(self, event):
        """Ensure background workers are cancelled safely before closing."""
        try:
            sweep_worker = getattr(self, "sweep_worker", None)
            custom_worker = getattr(self, "worker", None) or getattr(self, "custom_worker", None)
            sweep_running = bool(sweep_worker) and hasattr(sweep_worker, "isRunning") and sweep_worker.isRunning()
            custom_running = bool(custom_worker) and hasattr(custom_worker, "isRunning") and custom_worker.isRunning()
            sim_running = self._sim_thread is not None

            if sweep_running or custom_running or sim_running:
                setattr(self, "_closing_after_workers", True)
                try:
                    if sweep_running and hasattr(sweep_worker, "cancel"):
                        sweep_worker.cancel()
                except Exception:
                    pass
                try:
                    if custom_running and hasattr(custom_worker, "cancel"):
                        custom_worker.cancel()
                except Exception:
                    pass
                try:
                    if sim_running and self._sim_worker is not None:
                        self._sim_worker.cancel()
                except Exception:
                    pass

                try:
                    sp = getattr(self, "sweep_progress", None)
                    if sp is not None:
                        btn = sp.cancelButton()
                        if btn is not None:
                            btn.setEnabled(False)
                        sp.setLabelText("Canceling… finishing current game")
                except Exception:
                    pass
                try:
                    p = getattr(self, "progress", None)
                    if p is not None:
                        btn = p.cancelButton()
                        if btn is not None:
                            btn.setEnabled(False)
                        p.setLabelText("Canceling… finishing current game")
                except Exception:
                    pass
                try:
                    if sim_running:
                        self.progress_bar.setFormat("Canceling…")
                except Exception:
                    pass

                event.ignore()
                return
        except Exception:
            import traceback
            traceback.print_exc()

        super().closeEvent(event)


class StatsGraphWidget(QtWidgets.QWidget):
    def __init__(self, hist, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._hover_idx = None
        self._bar_hit_rects = []  # list[QtCore.QRectF]
        self.hist = []
        self.set_hist(hist)

    def set_hist(self, hist):
        self.hist = list(hist) if hist else []
        self._hover_idx = None
        self._bar_hit_rects = []
        self.update()

    # Backward compatibility: older parts of the UI call this method name.
    # The data are histogram counts indexed by "shots".
    def set_counts(self, counts):
        self.set_hist(counts)

    def leaveEvent(self, event):
        self._hover_idx = None
        QtWidgets.QToolTip.hideText()
        self.update()
        return super().leaveEvent(event)

    def mouseMoveEvent(self, event):
        if not self._bar_hit_rects:
            return super().mouseMoveEvent(event)

        pos = event.pos()
        hovered = None
        in_draw_rect = False
        for i, rect in enumerate(self._bar_hit_rects):
            if rect.contains(pos):
                if 0 <= i < len(self.hist) and self.hist[i] > 0:
                    hovered = i
                    in_draw_rect = True
                break

        if hovered != self._hover_idx:
            self._hover_idx = hovered
            if hovered is not None and in_draw_rect and 0 <= hovered < len(self.hist):
                count = int(self.hist[hovered])
                total = max(1, sum(self.hist))
                pct = 100.0 * (count / total)
                shots = hovered
                QtWidgets.QToolTip.showText(
                    event.globalPos(),
                    f"Shots: {shots}\nCount: {count}\nShare: {pct:.1f}%"
                )
            else:
                QtWidgets.QToolTip.hideText()
            self.update()

        return super().mouseMoveEvent(event)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, False)

        rect = self.rect()
        painter.fillRect(rect, QtGui.QColor(Theme.BG_PANEL))

        if not self.hist:
            painter.setPen(QtGui.QColor(Theme.TEXT_LABEL))
            painter.drawText(rect, QtCore.Qt.AlignCenter, "No histogram data.")
            return

        total = sum(self.hist)
        max_count = max(self.hist) if self.hist else 0
        if max_count <= 0:
            painter.setPen(QtGui.QColor(Theme.TEXT_LABEL))
            painter.drawText(rect, QtCore.Qt.AlignCenter, "No samples yet.")
            return

        pad = 10
        w = max(1, rect.width() - 2 * pad)
        h = max(1, rect.height() - 2 * pad)
        n = len(self.hist)

        bar_w = w / max(1, n)

        base = QtGui.QColor(Theme.HIGHLIGHT)
        hover = base.lighter(135)
        outline = QtGui.QColor(Theme.BG_BUTTON)

        self._bar_hit_rects = []

        for i, count in enumerate(self.hist):
            x0 = pad + i * bar_w
            hit_rect = QtCore.QRectF(x0, pad, bar_w, h)
            self._bar_hit_rects.append(hit_rect)

            if count <= 0:
                continue

            frac = float(count) / float(max_count)
            bar_h = max(1.0, frac * h)
            y0 = pad + (h - bar_h)
            draw_rect = QtCore.QRectF(x0, y0, max(1.0, bar_w - 1.0), bar_h)

            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(hover if i == self._hover_idx else base)
            painter.drawRect(draw_rect)

            if i == self._hover_idx:
                painter.setPen(QtGui.QPen(outline, 1))
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.drawRect(draw_rect)

        # Border
        painter.setPen(QtGui.QPen(outline, 1))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawRect(QtCore.QRect(pad, pad, w, h))
