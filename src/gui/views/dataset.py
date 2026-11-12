"""Training payload manager: browse, edit and save the corpus the trainer consumes.

The view is a paginated QTableView over a Qt-free document model, so a corpus with
tens of thousands of records never lands in the widget at once: only the current page
is rendered. The three editable columns are the payload fields the trainer reads -
the natural prompt, the target grammar and the SMT-LIB constraint - and saving always
writes the file shape back out, so a JSONL corpus stays JSONL and a wrapped document
keeps its wrapper.

Nothing here imports the trainer or the parser: the panel only reads and writes files,
which keeps the GUI on one side of the dependency arrow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QSize,
    Qt,
    pyqtSignal,
)
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QStyledItemDelegate,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from gui.model.dataset import (
    FIELD_AST,
    FIELD_LABELS,
    FIELD_NATURAL,
    FIELD_ORDER,
    FIELD_SMT,
    DatasetDocument,
    DatasetRecord,
    filter_records,
    load_dataset,
    save_dataset,
)

from .. import theme, widgets
from ..paths import Workspace, open_workspace

PAGE_SIZES = (10, 25, 50, 100, 250, 1000)
INDEX_COLUMN = 0
STATUS_COLUMN = 4
KEYS_COLUMN = 5
COLUMNS = (
    "#",
    FIELD_LABELS[FIELD_NATURAL],
    FIELD_LABELS[FIELD_AST],
    FIELD_LABELS[FIELD_SMT],
    "Shape",
    "Payload keys",
)
EDITABLE_COLUMNS = (1, 2, 3)
FIELD_OF_COLUMN = {1: FIELD_NATURAL, 2: FIELD_AST, 3: FIELD_SMT}
SAVE_FORMATS = ("json", "jsonl", "txt", "payload")


# ---------------------------------------------------------------------------
# Table model
# ---------------------------------------------------------------------------
class RecordTableModel(QAbstractTableModel):
    """Rows are positions in the loaded document; only the page is ever asked for."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.document: Optional[DatasetDocument] = None
        self._rows: List[int] = []
        self.dirty: set = set()

    # --- plumbing ------------------------------------------------------------
    def set_document(self, document: Optional[DatasetDocument]) -> None:
        self.beginResetModel()
        self.document = document
        self._rows = list(range(len(document.records))) if document else []
        self.dirty.clear()
        self.endResetModel()

    def set_rows(self, rows: List[int]) -> None:
        self.beginResetModel()
        self._rows = list(rows)
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(COLUMNS)

    def record_at(self, row: int) -> Optional[DatasetRecord]:
        if self.document is None or row < 0 or row >= len(self._rows):
            return None
        return self.document.records[self._rows[row]]

    def document_row(self, row: int) -> int:
        return self._rows[row] if 0 <= row < len(self._rows) else -1

    # --- data ----------------------------------------------------------------
    def _value_at(self, row: int, column: int) -> str:
        record = self.record_at(row)
        if record is None:
            return ""
        if column == INDEX_COLUMN:
            return str(self.document_row(row))
        if column in FIELD_OF_COLUMN:
            return record.get(FIELD_OF_COLUMN[column])
        if column == STATUS_COLUMN:
            return shape_of(record)
        return ", ".join(record.keys.get(field, "") for field in FIELD_ORDER if record.keys.get(field, ""))

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or self.document is None:
            return None
        column = index.column()
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole, Qt.ItemDataRole.ToolTipRole):
            value = self._value_at(index.row(), column)
            if role is Qt.ItemDataRole.ToolTipRole and len(value) > 400:
                return None
            return value
        if role is Qt.ItemDataRole.TextAlignmentRole and column == INDEX_COLUMN:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if role is Qt.ItemDataRole.ForegroundRole:
            record = self.record_at(index.row())
            if record is not None and self.document_row(index.row()) in self.dirty:
                return theme.color("data")
            if record is not None and column == STATUS_COLUMN and shape_of(record) != "3 of 3 bound":
                return theme.color("warn")
        return None

    def flags(self, index: QModelIndex) -> Any:
        base = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if index.isValid() and index.column() in EDITABLE_COLUMNS:
            return base | Qt.ItemFlag.ItemIsEditable
        return base

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:  # noqa: N802
        if role != Qt.ItemDataRole.EditRole or self.document is None or not index.isValid():
            return False
        field = FIELD_OF_COLUMN.get(index.column())
        record = self.record_at(index.row())
        if field is None or record is None:
            return False
        text = "" if value is None else str(value)
        if record.get(field) == text:
            return False
        record.set(field, text)
        self.dirty.add(self.document_row(index.row()))
        self.dataChanged.emit(index, index, [Qt.ItemDataRole.DisplayRole])
        return True

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> Any:  # noqa: N802
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return COLUMNS[section] if section < len(COLUMNS) else ""
        return str(self.document_row(section)) if section < len(self._rows) else ""


class MultilineDelegate(QStyledItemDelegate):
    """Editing a prompt or a constraint uses a multi-line box sized to the payload."""

    def createEditor(self, parent, option, index):  # noqa: N802 - Qt naming
        editor = QPlainTextEdit(parent)
        editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        editor.setMinimumHeight(140)
        editor.setFont(theme.mono_font(9.5))
        return editor

    def setEditorData(self, editor, index):  # noqa: N802 - Qt naming
        value = index.data(Qt.ItemDataRole.EditRole) or index.data(Qt.ItemDataRole.DisplayRole) or ""
        editor.setPlainText(str(value))

    def setModelData(self, editor, model, index):  # noqa: N802 - Qt naming
        model.setData(index, editor.toPlainText(), Qt.ItemDataRole.EditRole)

    def sizeHint(self, option, index):  # noqa: N802 - Qt naming
        value = index.data(Qt.ItemDataRole.DisplayRole) or ""
        lines = max(1, min(4, str(value).count("\n") + 1))
        return QSize(320, 18 * lines + 8)


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------
class DatasetPanel(QWidget):
    """Dataset module: file list, paginated table, record editor, save round trip."""

    record_requested = pyqtSignal(object)

    def __init__(self, workspace: Optional[Workspace] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.workspace = workspace or open_workspace(Path.cwd())
        self.document: Optional[DatasetDocument] = None
        self.console: Optional[widgets.LogConsole] = None
        self._loading_detail = False
        self._page = 0
        self._matches: List[int] = []
        self._build()

    # --- layout --------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.addLayout(self._file_row())

        self.model = RecordTableModel(self)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        self.table.setItemDelegate(MultilineDelegate(self.table))
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(INDEX_COLUMN, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(STATUS_COLUMN, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.selectionModel().currentChanged.connect(self._row_changed)
        self.model.dataChanged.connect(lambda *_: self._update_chips())
        root.addWidget(self.table, 3)

        root.addLayout(self._page_row())
        widgets.share_column(root, self._editor_card(), 2)

        self.chip_records = widgets.Chip("records -")
        self.chip_page = widgets.Chip("page -")
        self.chip_format = widgets.Chip("format -")
        self.chip_edit = widgets.Chip("no edits")
        bar = widgets.ToolRow()
        for chip in (self.chip_records, self.chip_page, self.chip_format, self.chip_edit):
            bar.add(chip)
        bar.add_spacer()
        self.status = widgets.Chip("idle")
        bar.add(self.status)
        root.addWidget(bar)

        shortcut = QShortcut(QKeySequence.StandardKey.Save, self)
        shortcut.activated.connect(self._save)

    def _file_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)
        self.picker = widgets.PathPicker(self.workspace.datasets_dir, widgets.DATASET_SUFFIXES, "training payload file, json jsonl or txt")
        self.picker.chosen.connect(self.load_path)
        row.addWidget(self.picker, 2)
        self.recent = QComboBox()
        self.recent.setMinimumWidth(240)
        self.recent.currentIndexChanged.connect(self._recent_picked)
        row.addWidget(self.recent, 2)
        for text, slot in (("Reload", self._reload), ("Add record", self._add_record), ("Delete row", self._delete_record)):
            button = QPushButton(text)
            button.clicked.connect(slot)
            row.addWidget(button)
        self.btn_save = QPushButton("Save")
        self.btn_save.setObjectName("Primary")
        self.btn_save.clicked.connect(self._save)
        row.addWidget(self.btn_save)
        self.btn_save_as = QPushButton("Save as")
        self.btn_save_as.clicked.connect(self._save_as)
        row.addWidget(self.btn_save_as)
        self.format_combo = QComboBox()
        self.format_combo.addItems(SAVE_FORMATS)
        row.addWidget(QLabel("write as"))
        row.addWidget(self.format_combo)
        return row

    def _page_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(QLabel("filter"))
        self.search = QLineEdit()
        self.search.setPlaceholderText("substring across the three payload fields")
        self.search.textChanged.connect(self._refilter)
        row.addWidget(self.search, 2)
        self.scope = QComboBox()
        self.scope.addItem("any field", None)
        for field in FIELD_ORDER:
            self.scope.addItem(FIELD_LABELS[field], field)
        self.scope.currentIndexChanged.connect(self._refilter)
        row.addWidget(self.scope)
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self._clear_filter)
        row.addWidget(self.btn_clear)
        row.addSpacing(12)
        for text, slot in (("First", lambda: self._goto_page(0)), ("Prev", lambda: self._goto_page(self._page - 1)), ("Next", lambda: self._goto_page(self._page + 1)), ("Last", lambda: self._goto_page(self._page_count() - 1))):
            button = QPushButton(text)
            button.clicked.connect(slot)
            row.addWidget(button)
        row.addWidget(QLabel("rows"))
        self.page_size = QComboBox()
        for size in PAGE_SIZES:
            self.page_size.addItem(str(size), size)
        self.page_size.setCurrentIndex(1)
        self.page_size.currentIndexChanged.connect(lambda: self._goto_page(0))
        row.addWidget(self.page_size)
        row.addStretch(1)
        return row

    def _editor_card(self) -> widgets.Card:
        card = widgets.Card("Selected record: the three fields the trainer reads")
        self.detail_index = widgets.Chip("no row selected")
        head = widgets.ToolRow()
        head.add(self.detail_index)
        head.add_spacer()
        self.btn_push = QPushButton("Send this record to the canvas")
        self.btn_push.clicked.connect(self._send_record)
        head.add(self.btn_push)
        card.add_span(head, 0, 0, 2)

        self.editors: Dict[str, QPlainTextEdit] = {}
        for position, field in enumerate(FIELD_ORDER):
            card.add_row(QLabel(FIELD_LABELS[field]), self._make_editor(field), row=position + 1)
            # the editors share whatever height this card is given, instead of a fixed box
            card.set_body_stretch(position + 1, 1)
        self.extra = QLabel("")
        self.extra.setObjectName("MutedLabel")
        self.extra.setWordWrap(True)
        card.add_span(self.extra, 4, 0, 2)
        return card

    def _make_editor(self, field: str) -> QPlainTextEdit:
        editor = QPlainTextEdit()
        editor.setObjectName("SourceView")
        editor.setMinimumHeight(70)
        editor.setMaximumHeight(260)
        editor.setFont(theme.mono_font(9.5))
        editor.setPlaceholderText(FIELD_LABELS[field])
        editor.textChanged.connect(lambda: self._editor_typed(field))
        self.editors[field] = editor
        return editor

    # --- console and refresh -------------------------------------------------
    def attach_console(self, console: widgets.LogConsole) -> None:
        self.console = console

    def _say(self, message: str, level: str = "info") -> None:
        if self.console is not None:
            self.console.log(message, level)

    def refresh(self, scan: Any = None) -> None:
        """Rebuild the recent-files list from a workspace scan."""
        source = scan if isinstance(scan, dict) else self.workspace.scan
        previous = self.recent.currentData()
        self.recent.blockSignals(True)
        self.recent.clear()
        for role in ("dataset", "corpus"):
            for artifact in source.get(role, []):
                self.recent.addItem(f"{role}: {self.workspace.relative(artifact.path)}", str(artifact.path))
        self.recent.blockSignals(False)
        if self.recent.count() == 0:
            self.recent.addItem("no payload file in the scan", "")
            return
        if previous:
            index = self.recent.findData(previous)
            if index >= 0:
                self.recent.setCurrentIndex(index)
        if self.document is None:
            candidate = self.workspace.newest("dataset")
            if candidate is not None:
                self.load_path(candidate.path)

    def _recent_picked(self, _index: int) -> None:
        raw = self.recent.currentData()
        if raw:
            self.load_path(raw)

    # --- loading -------------------------------------------------------------
    def load_path(self, raw: Any) -> bool:
        path = Path(str(raw))
        if not path.is_file():
            self._say(f"no payload file at {path}", "error")
            return False
        try:
            document = load_dataset(path)
        except Exception as exc:  # a broken corpus file must stay readable as data
            self._say(f"payload load failed for {path.name}: {exc}", "error")
            return False
        self.document = document
        self.model.set_document(document)
        self.format_combo.setCurrentText(self._format_label(document.format))
        self._matches = list(range(len(document.records)))
        self._page = 0
        self._apply_page()
        self._update_chips()
        stats = document.stats()
        self._say(f"{self.workspace.relative(path)}: {stats['records']} records, format {document.format}, {len(document.notes)} notes")
        for note in document.notes[:6]:
            self._say(f"payload note: {note}", "warn")
        if not any(document.stats()["fields"].values()):
            keys = ", ".join(sorted(document.schema())) or "none"
            self._say(f"no record carries the three trainer fields: the payload keys in this file are {keys}", "warn")
        if document.truncated_at is not None:
            self._say(f"the loader stopped at record {document.truncated_at}: the rest stays on disk until you raise the limit", "warn")
        self.status.setText("loaded")
        self.status.set_tone("pass")
        return True

    def _format_label(self, format_name: str) -> str:
        """Map the format the loader detected onto a choice the panel offers."""
        mapped = {"text": "txt"}.get(format_name, format_name)
        return mapped if mapped in SAVE_FORMATS else "json"

    def _reload(self) -> None:
        if self.document is not None and self.document.path is not None:
            self.load_path(self.document.path)

    # --- paging and filtering ------------------------------------------------
    def _page_count(self) -> int:
        size = int(self.page_size.currentData() or 25)
        total = len(self._matches)
        return max(1, (total + size - 1) // size)

    def _goto_page(self, page: int) -> None:
        self._page = max(0, min(page, self._page_count() - 1))
        self._apply_page()

    def _apply_page(self) -> None:
        size = int(self.page_size.currentData() or 25)
        start = self._page * size
        self.model.set_rows(self._matches[start : start + size])
        self._update_chips()

    def _refilter(self, _text: str = "") -> None:
        if self.document is None:
            return
        needle = self.search.text().strip()
        if not needle:
            self._matches = list(range(len(self.document.records)))
        else:
            self._matches = filter_records(self.document, needle, self.scope.currentData())
        self._page = 0
        self._apply_page()
        self._say(f"filter {needle or 'none'}: {len(self._matches)} of {len(self.document.records)} records")

    def _clear_filter(self) -> None:
        self.search.blockSignals(True)
        self.search.setText("")
        self.search.blockSignals(False)
        self._refilter()

    # --- record editing ------------------------------------------------------
    def _row_changed(self, index: QModelIndex, _previous: QModelIndex) -> None:
        record = self.model.record_at(index.row()) if index.isValid() else None
        self._loading_detail = True
        if record is None:
            self.detail_index.setText("no row selected")
            for editor in self.editors.values():
                editor.setPlainText("")
            self.extra.setText("")
        else:
            self.detail_index.setText(f"record {self.model.document_row(index.row())}")
            for field, editor in self.editors.items():
                editor.setPlainText(record.get(field))
            extras = record.extra_fields()
            bound = "; ".join(f"{FIELD_LABELS[field]} = {record.keys.get(field) or 'unbound'}" for field in FIELD_ORDER)
            other = ", ".join(extras) if extras else "none"
            self.extra.setText(f"bound keys: {bound}   |   other payload keys: {other}")
        self._loading_detail = False

    def _editor_typed(self, field: str) -> None:
        if self._loading_detail:
            return
        index = self.table.currentIndex()
        if not index.isValid():
            return
        record = self.model.record_at(index.row())
        if record is None:
            return
        value = self.editors[field].toPlainText()
        if record.get(field) == value:
            return
        record.set(field, value)
        self.model.dirty.add(self.model.document_row(index.row()))
        column = FIELD_ORDER.index(field) + 1
        edited = self.model.index(index.row(), column)
        shape = self.model.index(index.row(), STATUS_COLUMN)
        self.model.dataChanged.emit(edited, edited, [Qt.ItemDataRole.DisplayRole])
        self.model.dataChanged.emit(shape, shape, [Qt.ItemDataRole.DisplayRole])
        self._update_chips()

    def _add_record(self) -> None:
        if self.document is None:
            self._say("open a payload file before adding a record", "warn")
            return
        record = DatasetRecord(index=len(self.document.records), format=self.document.format)
        self.document.records.append(record)
        self._matches.append(len(self.document.records) - 1)
        self._apply_page()
        self.model.dirty.add(record.index)
        self._say(f"appended record {record.index}; fill the three fields and save", "warn")
        self._update_chips()

    def _delete_record(self) -> None:
        index = self.table.currentIndex()
        if not index.isValid() or self.document is None:
            self._say("select the row to remove first", "warn")
            return
        position = self.model.document_row(index.row())
        if position < 0:
            return
        removed = self.document.records.pop(position)
        for row, record in enumerate(self.document.records):
            record.index = row
        self._refilter()
        self._say(f"removed record {removed.index} from the working copy: save to write it out", "warn")
        self._update_chips()

    def _send_record(self) -> None:
        index = self.table.currentIndex()
        record = self.model.record_at(index.row()) if index.isValid() else None
        if record is None:
            self._say("select a row to send to the canvas", "warn")
            return
        self.record_requested.emit(record)

    # --- saving --------------------------------------------------------------
    def _save(self) -> None:
        if self.document is None:
            self._say("nothing to save: no payload file is loaded", "warn")
            return
        if self.document.path is None:
            self._save_as()
            return
        self._write(self.document.path, self.format_combo.currentText())

    def _save_as(self) -> None:
        if self.document is None:
            self._say("nothing to save: no payload file is loaded", "warn")
            return
        start = str(self.document.path or (self.workspace.datasets_dir / "payload.json"))
        chosen, _filter = QFileDialog.getSaveFileName(self, "Save payload", start, widgets.filter_for(widgets.DATASET_SUFFIXES))
        if not chosen:
            return
        path = Path(chosen)
        format_hint = path.suffix.lstrip(".").lower() or self.format_combo.currentText()
        self._write(path, format_hint if format_hint in SAVE_FORMATS else self.format_combo.currentText())

    def _write(self, path: Path, format_hint: str) -> None:
        if self.document is None:
            return
        try:
            written = save_dataset(self.document, path, format_hint)
        except Exception as exc:
            self._say(f"payload save failed for {path.name}: {exc}", "error")
            self.status.setText("save failed")
            self.status.set_tone("fail")
            return
        self.document.path = written
        self.model.dirty.clear()
        self.picker.set_path(str(written))
        self.format_combo.setCurrentText(self._format_label(self.document.format))
        self._update_chips()
        self._say(f"wrote {self.workspace.relative(written)} in {format_hint} shape: {len(self.document.records)} records")
        self.status.setText("saved")
        self.status.set_tone("pass")

    # --- status --------------------------------------------------------------
    def _update_chips(self) -> None:
        if self.document is None:
            self.chip_records.setText("records -")
            self.chip_page.setText("page -")
            self.chip_format.setText("format -")
            return
        stats = self.document.stats()
        self.chip_records.setText(f"records {stats['records']}")
        self.chip_page.setText(f"page {self._page + 1} of {self._page_count()}  shown {len(self._matches)}")
        self.chip_format.setText(f"format {self.document.format}")
        count = len(self.model.dirty)
        self.chip_edit.setText(f"{count} edits pending" if count else "no edits pending")
        self.chip_edit.set_tone("warn" if count else "pass")

    # --- stream hook ---------------------------------------------------------
    def handle_stream_event(self, event: Any) -> None:
        """Highlight the record whose prompt the orchestrator just processed."""
        if self.document is None:
            return
        preview_of_event = getattr(event, "preview", None)
        if callable(preview_of_event):
            needle = str(preview_of_event(200) or "")
        else:
            needle = str(getattr(event, "envelope", "") or "")
        needle = needle.strip()
        if not needle:
            return
        for position, record in enumerate(self.document.records):
            prompt = record.get(FIELD_NATURAL).strip()
            if prompt and (prompt in needle or needle in prompt):
                self._say(f"stream event touches record {record.index}: {prompt[:60]}")
                return

    def stop_worker(self) -> None:
        """Nothing runs in the background here: reading a payload file is synchronous."""
        return None


def shape_of(record: DatasetRecord) -> str:
    """How many of the three trainer fields carry a non-empty value."""
    bound = sum(1 for field in FIELD_ORDER if record.get(field).strip())
    return f"{bound} of {len(FIELD_ORDER)} bound"


__all__ = ["DatasetPanel", "MultilineDelegate", "RecordTableModel", "shape_of"]
