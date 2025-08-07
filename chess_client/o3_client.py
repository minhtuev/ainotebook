from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import chess
from openai import OpenAI


@dataclass
class O3Config:
    model: str = "o3-mini"
    effort: str = "medium"  # low | medium | high


class O3ChessClient:
    def __init__(self, config: Optional[O3Config] = None) -> None:
        self.config = config or O3Config()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=api_key)

    def _build_prompt(self, board: chess.Board, legal_moves: List[str]) -> str:
        color_to_move = "White" if board.turn == chess.WHITE else "Black"
        return (
            "You are a super, super strong chess engine. You must choose exactly one legal move. Try to use the best move possible with strong openings and tactics.\n"
            "Rules:\n"
            "- Only output a single UCI move string (e.g., e2e4, g1f3, e7e8q).\n"
            "- The move MUST be chosen from the provided LEGAL MOVES list.\n"
            "- Do not add any other text.\n\n"
            f"Side to move: {color_to_move}\n"
            f"FEN: {board.fen()}\n"
            f"LEGAL MOVES (UCI): {', '.join(legal_moves)}\n"
            "Output: <one UCI move from the list>\n"
        )

    def choose_move(self, board: chess.Board) -> Optional[chess.Move]:
        legal_moves = [m.uci() for m in board.legal_moves]
        if not legal_moves:
            return None

        prompt = self._build_prompt(board, legal_moves)

        # First attempt
        move_str = self._ask_o3_for_move(prompt)
        if move_str in legal_moves:
            return chess.Move.from_uci(move_str)

        # Retry with stricter formatting
        strict_prompt = prompt + "\nIMPORTANT: Output ONLY the UCI string, no commentary."
        move_str = self._ask_o3_for_move(strict_prompt)
        if move_str in legal_moves:
            return chess.Move.from_uci(move_str)

        # Fallback to a random legal move to keep the game flowing
        return chess.Move.from_uci(legal_moves[0])

    def _ask_o3_for_move(self, prompt: str) -> str:
        try:
            print("Making request to model", self.config.model)
            print("Prompt:", prompt)
            completion = self.client.responses.create(
                model=self.config.model,
                reasoning={"effort": self.config.effort},
                input=prompt,
                max_output_tokens=8,
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
                        if isinstance(content, dict) and content.get("type") == "output_text":
                            parts.append(content.get("text", ""))
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

