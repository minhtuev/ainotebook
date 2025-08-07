from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

import chess
from PySide6 import QtWidgets, QtCore
import threading

from .qt_gui import ChessWindow
from .o3_client import O3ChessClient, O3Config


def play_game(
    ai_color: str,
    model: str,
    effort: str,
    size: int,
    temperature: float,
    top_p: float | None,
    shuffle_legal_moves: bool,
    samples: int,
    heuristic: bool,
) -> None:
    if ai_color not in ("white", "black"):
        raise SystemExit("--ai-color must be 'white' or 'black'")

    app = QtWidgets.QApplication([])
    window = ChessWindow(size=size)
    window.show()

    o3 = O3ChessClient(
        O3Config(
            model=model,
            effort=effort,
            temperature=temperature,
            top_p=top_p,
            shuffle_legal_moves=shuffle_legal_moves,
            num_samples=samples,
            heuristic_selection=heuristic,
        )
    )
    human_is_white = ai_color == "black"

    class Notifier(QtCore.QObject):
        done = QtCore.Signal(str, str)  # (uci, reason)

    ai_job_running = {"flag": False}
    notifier = Notifier()

    def maybe_ai_move() -> None:
        board = window.widget.board
        if board.is_game_over() or ai_job_running["flag"]:
            return
        if (board.turn == chess.WHITE and not human_is_white) or (
            board.turn == chess.BLACK and human_is_white
        ):
            window.show_thinking()
            ai_job_running["flag"] = True

            def worker_fn(fen: str) -> None:
                try:
                    local_board = chess.Board(fen)
                    move, reason = o3.choose_move_with_reason(local_board)
                    uci = move.uci() if move else ""
                except Exception:
                    uci, reason = "", ""
                notifier.done.emit(uci, reason)

            def on_done(uci: str, reason: str) -> None:
                ai_job_running["flag"] = False
                # Apply to current board if still legal
                if uci:
                    try:
                        mv = chess.Move.from_uci(uci)
                    except Exception:
                        mv = None
                    if mv is not None and mv in board.legal_moves:
                        board.push(mv)
                        window.widget.last_move = mv
                        window.widget.update()
                window.set_reasoning_text(reason)

            notifier.done.connect(on_done)
            t = threading.Thread(target=worker_fn, args=(board.fen(),), daemon=True)
            t.start()

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
    parser.add_argument("--temperature", type=float, default=0.7, help="sampling temperature (higher = more random)")
    parser.add_argument("--top-p", dest="top_p", type=float, default=None, help="nucleus sampling probability mass cap")
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false", help="disable random shuffling of legal moves in prompt")
    parser.add_argument("--samples", type=int, default=1, help="number of samples to request from the model (aggregated)")
    parser.add_argument("--heuristic", action="store_true", help="use a simple material heuristic to pick among sampled moves")
    args = parser.parse_args(argv)

    play_game(
        ai_color=args.ai_color,
        model=args.model,
        effort=args.effort,
        size=args.size,
        temperature=args.temperature,
        top_p=args.top_p,
        shuffle_legal_moves=args.shuffle,
        samples=args.samples,
        heuristic=args.heuristic,
    )


if __name__ == "__main__":
    main()

