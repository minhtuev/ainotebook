from __future__ import annotations

import os
from dataclasses import dataclass
import random
import chess.pgn
from typing import List, Optional

import chess
from openai import OpenAI


@dataclass
class O3Config:
    model: str = "o3-mini"
    effort: str = "high"  # low | medium | high
    temperature: float = 0.2
    top_p: float | None = None
    shuffle_legal_moves: bool = False
    num_samples: int = 1
    heuristic_selection: bool = False


class O3ChessClient:
    def __init__(self, config: Optional[O3Config] = None) -> None:
        self.config = config or O3Config()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=api_key)

    def _build_prompt(self, board: chess.Board, legal_moves: List[str]) -> str:
        color_to_move = "White" if board.turn == chess.WHITE else "Black"
        history_san = self._history_san(board, max_moves=40)
        legal_block = "\n".join(legal_moves)
        return (
            "You are a strong chess player. Choose the BEST move from the legal move list.\n"
            "Guidelines (for internal reasoning only): prioritize checkmates and tactics, avoid blunders/hanging pieces, consider opponent threats, develop pieces, control the center, improve king safety.\n"
            "Return format: output ONLY one UCI move from the LEGAL MOVES list, no commentary.\n\n"
            f"Side to move: {color_to_move}\n"
            f"FEN: {board.fen()}\n"
            f"Moves so far (SAN): {history_san}\n"
            "LEGAL MOVES (UCI, one per line):\n"
            f"{legal_block}\n"
            "Output: <one UCI move from the list>\n"
        )

    @staticmethod
    def _history_san(board: chess.Board, max_moves: int = 40) -> str:
        # Build SAN list from move stack without mutating the original board
        temp = chess.Board()
        sans: List[str] = []
        for move in board.move_stack:
            sans.append(temp.san(move))
            temp.push(move)
        if not sans:
            return "(start)"
        return " ".join(sans[-max_moves:])

    def choose_move(self, board: chess.Board) -> Optional[chess.Move]:
        legal_moves = [m.uci() for m in board.legal_moves]
        if not legal_moves:
            return None

        ordered_moves = list(legal_moves)
        if self.config.shuffle_legal_moves:
            random.shuffle(ordered_moves)

        prompt = self._build_prompt(board, ordered_moves)

        # Single-sample fast path without heuristic
        if self.config.num_samples <= 1 and not self.config.heuristic_selection:
            move = self._one_shot_move(board, prompt, legal_moves)
            if move is not None:
                return move
            return chess.Move.from_uci(legal_moves[0])

        # Multi-sample + heuristic selection
        candidate_uci: list[str] = []
        attempts = max(self.config.num_samples, 1)
        for _ in range(attempts):
            move_str = self._ask_o3_for_move(prompt)
            if move_str in legal_moves and move_str not in candidate_uci:
                candidate_uci.append(move_str)
        # Ensure at least one
        if not candidate_uci:
            move = self._one_shot_move(board, prompt, legal_moves)
            if move is not None:
                return move
            return chess.Move.from_uci(legal_moves[0])

        # Immediate tactical: prefer mate in 1 if found
        for uci in candidate_uci:
            mv = chess.Move.from_uci(uci)
            temp = board.copy(stack=False)
            temp.push(mv)
            if temp.is_checkmate():
                return mv

        if not self.config.heuristic_selection:
            return chess.Move.from_uci(candidate_uci[0])

        # Heuristic: simple material evaluation after our move
        acting_color = board.turn
        best_score = -10_000.0
        best_move: Optional[chess.Move] = None
        for uci in candidate_uci:
            mv = chess.Move.from_uci(uci)
            temp = board.copy(stack=False)
            temp.push(mv)
            score = self._material_score(temp, perspective=acting_color)
            if score > best_score or best_move is None:
                best_score = score
                best_move = mv

        return best_move or chess.Move.from_uci(candidate_uci[0])

    def choose_move_with_reason(self, board: chess.Board) -> tuple[Optional[chess.Move], str]:
        """Return (move, short_reason). The reason is a brief justification suitable for UI.
        This deliberately avoids chain-of-thought and requests only a short summary."""
        move = self.choose_move(board)
        if move is None:
            return None, "No legal move available."
        try:
            reason = self._ask_o3_for_reason(board, move)
        except Exception:
            reason = ""
        return move, (reason or "")

    def _one_shot_move(self, board: chess.Board, prompt: str, legal_moves: list[str]) -> Optional[chess.Move]:
        move_str = self._ask_o3_for_move(prompt)
        if move_str in legal_moves:
            return chess.Move.from_uci(move_str)
        strict_prompt = prompt + "\nIMPORTANT: Output ONLY the UCI string, no commentary."
        move_str = self._ask_o3_for_move(strict_prompt)
        if move_str in legal_moves:
            return chess.Move.from_uci(move_str)
        return None

    def _ask_o3_for_move(self, prompt: str) -> str:
        try:
            print("Making request to model", self.config.model)
            print("Prompt:", prompt)
            kwargs = {
                "model": self.config.model,
                "reasoning": {"effort": self.config.effort},
                "input": prompt,
                "max_output_tokens": 8,
                "temperature": self.config.temperature,
            }
            if self.config.top_p is not None:
                kwargs["top_p"] = self.config.top_p

            completion = self.client.responses.create(
                **kwargs,
            )
            text = self._extract_text(completion)
            return self._only_uci(text)
        except Exception:
            return ""

    @staticmethod
    def _extract_text(resp) -> str:
        # Prefer the convenience field if available
        try:
            text = getattr(resp, "output_text", None)
            if isinstance(text, str) and text.strip():
                return text.strip()
        except Exception:
            pass

        # Fallback to manual extraction if SDK structure changes
        try:
            output = getattr(resp, "output", None)
            if not output:
                return ""
            parts = []
            for item in output:
                if isinstance(item, dict) and item.get("type") == "message":
                    for content in item.get("content", []):
                        if not isinstance(content, dict):
                            continue
                        ctype = content.get("type")
                        if ctype == "output_text" or ctype == "text":
                            text_val = content.get("text", "")
                            if isinstance(text_val, str):
                                parts.append(text_val)
            return "\n".join(p.strip() for p in parts if p and p.strip())
        except Exception:
            return ""

    @staticmethod
    def _only_uci(text: str) -> str:
        # Extract first 4-5 char token that looks like UCI (with optional promotion)
        if not text:
            return ""
        cand = text.strip().split()[0]
        cand = cand.strip().lower()
        # Accept patterns like e2e4, g7g8q
        if len(cand) in (4, 5):
            return cand
        # Remove punctuation
        cand = "".join(ch for ch in cand if ch.isalnum())
        if len(cand) in (4, 5):
            return cand
        return ""

    def _ask_o3_for_reason(self, board: chess.Board, move: chess.Move) -> str:
        # Request a concise one-sentence justification only.
        color_to_move = "White" if board.turn == chess.WHITE else "Black"
        history_san = self._history_san(board, max_moves=40)
        uci = move.uci()
        prompt = (
            "Provide a concise one-sentence justification for the selected chess move.\n"
            "Do not reveal step-by-step chain-of-thought.\n"
            f"Side to move: {color_to_move}\n"
            f"FEN: {board.fen()}\n"
            f"Moves so far (SAN): {history_san}\n"
            f"Selected move (UCI): {uci}\n"
            "Output: one short sentence."
        )
        completion = self.client.responses.create(
            model=self.config.model,
            input=prompt,
            max_output_tokens=80,
            temperature=max(0.1, min(self.config.temperature, 0.7)),
        )
        text = self._extract_text(completion)
        return (text or "").strip()

    @staticmethod
    def _material_score(board: chess.Board, perspective: bool) -> float:
        piece_values = {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.0,
            chess.BISHOP: 3.25,
            chess.ROOK: 5.0,
            chess.QUEEN: 9.0,
            chess.KING: 0.0,
        }
        white_score = 0.0
        black_score = 0.0
        for piece_type in piece_values:
            white_score += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
            black_score += len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
        diff = white_score - black_score
        return diff if perspective == chess.WHITE else -diff

