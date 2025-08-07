from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

import chess
from PySide6 import QtWidgets, QtCore

from .qt_gui import ChessWindow
from .o3_client import O3ChessClient, O3Config


def play_game(ai_color: str, model: str, effort: str, size: int) -> None:
    if ai_color not in ("white", "black"):
        raise SystemExit("--ai-color must be 'white' or 'black'")

    app = QtWidgets.QApplication([])
    window = ChessWindow(size=size)
    window.show()

    o3 = O3ChessClient(O3Config(model=model, effort=effort))
    human_is_white = ai_color == "black"

    def maybe_ai_move() -> None:
        board = window.widget.board
        if board.is_game_over():
            return
        if (board.turn == chess.WHITE and not human_is_white) or (
            board.turn == chess.BLACK and human_is_white
        ):
            move = o3.choose_move(board)
            if move is not None and move in board.legal_moves:
                board.push(move)
                window.widget.last_move = move
                window.widget.update()

    # When human moves, let AI respond
    window.widget.move_made.connect(lambda _m: maybe_ai_move())

    # If AI starts
    QtCore.QTimer.singleShot(50, maybe_ai_move)

    app.exec()


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Play local chess vs O3")
    parser.add_argument("--ai-color", default="black", choices=["white", "black"], help="which side the AI plays")
    parser.add_argument("--model", default="o3-mini", help="OpenAI model name")
    parser.add_argument("--effort", default="medium", choices=["low", "medium", "high"], help="reasoning effort")
    parser.add_argument("--size", type=int, default=720, help="window size in pixels")
    args = parser.parse_args(argv)

    play_game(ai_color=args.ai_color, model=args.model, effort=args.effort, size=args.size)


if __name__ == "__main__":
    main()

