"""
workers/pruner_worker.py
------------------------
Micro-Worker #9 — Context Window Pruner

Responsibility: Given a multi-turn conversation history and a token budget,
prune oldest dialogue turns while always preserving the system prompt
and the latest user message. Pure-Python, no model weights.

RAM budget: ~10 MB
"""

from __future__ import annotations

import traceback
from typing import Dict, List
from workers.base_worker import BaseWorker, WorkerResult


class PrunerWorker(BaseWorker):
    """
    Prunes a chat message list to fit within a token budget.

    run() kwargs
    ------------
    messages        : List[Dict[str,str]]  — [{"role": "...", "content": "..."}, ...]
    max_budget_tokens : int                — max tokens allowed for all messages

    WorkerResult.data
    -----------------
    dict with keys:
      pruned_messages     : List[Dict[str,str]]  — messages that fit the budget
      original_count      : int
      pruned_count        : int
      turns_dropped       : int
      estimated_tokens    : int  — estimated tokens of result
    """

    worker_id = "pruner"
    ram_budget_mb = 10.0
    requires_model = False

    def run(
        self,
        messages: List[Dict[str, str]] = None,
        max_budget_tokens: int = 2048,
        **kwargs,
    ) -> WorkerResult:
        if messages is None:
            return self._err("PrunerWorker.run() requires 'messages' kwarg.")

        try:
            original_count = len(messages)
            pruned = self._prune(messages, max_budget_tokens)
            pruned_count = len(pruned)
            turns_dropped = original_count - pruned_count
            est_tokens = sum(
                self._estimate_tokens(m.get("content", "")) for m in pruned
            )

            data = {
                "pruned_messages":   pruned,
                "original_count":    original_count,
                "pruned_count":      pruned_count,
                "turns_dropped":     turns_dropped,
                "estimated_tokens":  est_tokens,
            }
            return self._ok(data, ram_mb=2.0, turns_dropped=turns_dropped)

        except Exception as exc:
            return self._err(f"PrunerWorker error: {exc}\n{traceback.format_exc()}")

    # ── Pruning logic ─────────────────────────────────────────────────────

    def _prune(
        self,
        messages: List[Dict[str, str]],
        max_budget: int,
    ) -> List[Dict[str, str]]:
        """
        FIFO pruning: always keep system prompt (if any) + latest user message.
        Drop oldest intermediate turns first.
        """
        if not messages:
            return []

        # Identify system prompt
        has_system = messages[0].get("role") == "system"
        system_msg = messages[0] if has_system else None
        dialogue = list(messages[1:]) if has_system else list(messages)

        if not dialogue:
            return [system_msg] if system_msg else []

        latest = dialogue[-1]
        intermediate = dialogue[:-1]

        # Always-keep tokens
        base_tok = self._estimate_tokens(latest.get("content", ""))
        if system_msg:
            base_tok += self._estimate_tokens(system_msg.get("content", ""))

        remaining = max_budget - base_tok
        if remaining <= 0:
            return ([system_msg] if system_msg else []) + [latest]

        # Pack from newest intermediate → oldest
        packed: List[Dict[str, str]] = []
        for msg in reversed(intermediate):
            t = self._estimate_tokens(msg.get("content", ""))
            if t <= remaining:
                packed.insert(0, msg)
                remaining -= t
            else:
                break

        result = []
        if system_msg:
            result.append(system_msg)
        result.extend(packed)
        result.append(latest)
        return result

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        if not text:
            return 0
        words = text.split()
        word_est = len(words) * 1.3
        char_est = len(text) / 3.8
        return max(1, int((word_est + char_est) / 2))
