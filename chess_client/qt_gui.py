from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import chess
from PySide6 import QtCore, QtGui, QtWidgets


Color = Tuple[int, int, int]


@dataclass
class Theme:
    light: Color = (240, 217, 181)
    dark: Color = (181, 136, 99)
    select: Color = (246, 246, 105)
    highlight: Color = (186, 202, 68)
    border: Color = (40, 40, 40)
    text: Color = (20, 20, 20)


PIECE_UNICODE = {
    chess.PAWN:   {True: "♙", False: "♟"},
    chess.KNIGHT: {True: "♘", False: "♞"},
    chess.BISHOP: {True: "♗", False: "♝"},
    chess.ROOK:   {True: "♖", False: "♜"},
    chess.QUEEN:  {True: "♕", False: "♛"},
    chess.KING:   {True: "♔", False: "♚"},
}


class ChessWidget(QtWidgets.QWidget):
    move_made = QtCore.Signal(chess.Move)

    def __init__(self, size: int = 720, theme: Optional[Theme] = None, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.size_px = size
        self.square_px = size // 8
        self.theme = theme or Theme()
        self.board = chess.Board()
        self.selected_square: Optional[int] = None
        self.legal_targets_for_selected: List[int] = []
        self.last_move: Optional[chess.Move] = None

        # Choose a reasonable font with chess glyphs
        font = QtGui.QFont("DejaVu Sans", max(12, self.square_px - 18))
        self.piece_font = font
        self.info_font = QtGui.QFont("DejaVu Sans", 12)

    def reset(self) -> None:
        self.board.reset()
        self.selected_square = None
        self.legal_targets_for_selected = []
        self.last_move = None
        self.update()

    def set_board(self, board: chess.Board) -> None:
        self.board = board
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
        painter = QtGui.QPainter(self)
        self._draw_board(painter)
        self._draw_pieces(painter)
        self._draw_status(painter)
        painter.end()

    def _draw_board(self, painter: QtGui.QPainter) -> None:
        for rank in range(8):
            for file in range(8):
                square = chess.square(file, 7 - rank)
                is_light = (file + rank) % 2 == 0
                color = self.theme.light if is_light else self.theme.dark
                if self.last_move and (square == self.last_move.from_square or square == self.last_move.to_square):
                    color = self.theme.highlight
                if self.selected_square is not None and square == self.selected_square:
                    color = self.theme.select

                painter.fillRect(
                    file * self.square_px,
                    rank * self.square_px,
                    self.square_px,
                    self.square_px,
                    QtGui.QColor(*color),
                )

        # Target dots
        pen = QtGui.QPen(QtGui.QColor(*self.theme.border))
        brush = QtGui.QBrush(QtGui.QColor(*self.theme.border))
        painter.setPen(pen)
        painter.setBrush(brush)
        for target in self.legal_targets_for_selected:
            file = chess.square_file(target)
            rank = 7 - chess.square_rank(target)
            cx = file * self.square_px + self.square_px // 2
            cy = rank * self.square_px + self.square_px // 2
            radius = max(4, self.square_px // 12)
            painter.drawEllipse(QtCore.QPoint(cx, cy), radius, radius)

    def _draw_pieces(self, painter: QtGui.QPainter) -> None:
        painter.setFont(self.piece_font)
        painter.setPen(QtGui.QColor(*self.theme.text))
        for square, piece in self.board.piece_map().items():
            file = chess.square_file(square)
            rank = 7 - chess.square_rank(square)
            symbol = PIECE_UNICODE[piece.piece_type][piece.color]
            rect = QtCore.QRect(
                file * self.square_px,
                rank * self.square_px,
                self.square_px,
                self.square_px,
            )
            painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, symbol)

    def _draw_status(self, painter: QtGui.QPainter) -> None:
        painter.setFont(self.info_font)
        painter.setPen(QtGui.QColor(*self.theme.border))
        status = "White to move" if self.board.turn == chess.WHITE else "Black to move"
        if self.board.is_game_over():
            if self.board.is_checkmate():
                status = "Checkmate"
            elif self.board.is_stalemate():
                status = "Stalemate"
            else:
                status = "Game over"
        painter.drawText(8, 18, status)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        pos = event.position().toPoint()
        file = pos.x() // self.square_px
        rank = pos.y() // self.square_px
        if not (0 <= file < 8 and 0 <= rank < 8):
            return
        square = chess.square(file, 7 - rank)

        if self.selected_square is None:
            targets = [m.to_square for m in self.board.legal_moves if m.from_square == square]
            if targets:
                self.selected_square = square
                self.legal_targets_for_selected = targets
                self.update()
            return

        # Destination
        if square in self.legal_targets_for_selected:
            move = self._resolve_move_with_promotion(self.selected_square, square)
            self._clear_selection()
            if move in self.board.legal_moves:
                self.board.push(move)
                self.last_move = move
                self.update()
                self.move_made.emit(move)
            return

        # Reselect if clicking another legal-from square
        targets = [m.to_square for m in self.board.legal_moves if m.from_square == square]
        if targets:
            self.selected_square = square
            self.legal_targets_for_selected = targets
            self.update()
            return

        self._clear_selection()
        self.update()

    def _resolve_move_with_promotion(self, from_sq: int, to_sq: int) -> chess.Move:
        move = chess.Move(from_sq, to_sq)
        if move in self.board.legal_moves:
            return move
        for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
            move = chess.Move(from_sq, to_sq, promotion=promo)
            if move in self.board.legal_moves:
                return move
        return chess.Move(from_sq, to_sq)

    def _clear_selection(self) -> None:
        self.selected_square = None
        self.legal_targets_for_selected = []


class ChessWindow(QtWidgets.QMainWindow):
    def __init__(self, size: int = 720, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Chess vs O3")

        # Main splitter: left board, right reasoning
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.widget = ChessWidget(size=size)
        self.splitter.addWidget(self.widget)

        self.reasoning_text = QtWidgets.QPlainTextEdit()
        self.reasoning_text.setReadOnly(True)
        self.reasoning_text.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.reasoning_text.setFont(mono)
        self.splitter.addWidget(self.reasoning_text)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        self.setCentralWidget(self.splitter)

        self.status = self.statusBar()

        # Shortcuts
        QtGui.QShortcut(QtGui.QKeySequence("Q"), self, activated=self.close)
        QtGui.QShortcut(QtGui.QKeySequence("R"), self, activated=self.reset)
        QtGui.QShortcut(QtGui.QKeySequence("T"), self, activated=self.toggle_reasoning_panel)

    def reset(self) -> None:
        self.widget.reset()
        self.reasoning_text.clear()
        self.status.showMessage("New game", 1500)

    def set_reasoning_text(self, text: str) -> None:
        self.reasoning_text.setPlainText(text)
        # Scroll to top for new reasoning
        self.reasoning_text.moveCursor(QtGui.QTextCursor.MoveOperation.Start)

    def show_thinking(self) -> None:
        self.set_reasoning_text("Thinking...")

    def toggle_reasoning_panel(self) -> None:
        # Show/hide right pane
        if self.reasoning_text.isVisible():
            self.reasoning_text.hide()
        else:
            self.reasoning_text.show()

