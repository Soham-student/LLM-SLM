"""
workers/token_worker.py
-----------------------
Micro-Worker #6 — Token Optimizer / Compressor

Responsibility: Compress and clean prompt text to reduce token count
without changing semantic meaning. Pure-Python regex, no model weights.

RAM budget: ~10 MB
"""

from __future__ import annotations

import re
import traceback
from workers.base_worker import BaseWorker, WorkerResult


class TokenWorker(BaseWorker):
    """
    Compresses a prompt to use fewer tokens.

    run() kwargs
    ------------
    prompt          : str  — raw user prompt
    system_prompt   : str  — optional system prompt (for context-aware budgeting)
    max_tokens      : int  — target max new tokens hint

    WorkerResult.data
    -----------------
    dict with keys:
      original_text          : str
      optimized_text         : str
      original_char_count    : int
      optimized_char_count   : int
      estimated_original_tokens : int
      estimated_optimized_tokens: int
      tokens_saved           : int
      compression_ratio      : float  (0.0 = no saving, 0.3 = 30% fewer tokens)
    """

    worker_id = "token"
    ram_budget_mb = 10.0
    requires_model = False

    def run(
        self,
        prompt: str = "",
        system_prompt: str = "",
        max_tokens: int = 512,
        **kwargs,
    ) -> WorkerResult:
        if not isinstance(prompt, str):
            return self._err("TokenWorker.run() requires 'prompt' as str.")

        try:
            orig_text = prompt
            orig_chars = len(orig_text)
            orig_tok = self._estimate_tokens(orig_text)

            cleaned = orig_text

            # 1. Normalize line endings
            cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

            # 2. Collapse 3+ blank lines → 2
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

            # 3. Collapse 3+ spaces/tabs → 2 spaces
            cleaned = re.sub(r"[ \t]{3,}", "  ", cleaned)

            # 4. Remove trailing whitespace per line
            cleaned = "\n".join(line.rstrip() for line in cleaned.split("\n"))

            # 5. Trim overall
            cleaned = cleaned.strip()

            # 6. Repetitive punctuation collapse
            cleaned = re.sub(r"\?{2,}", "?", cleaned)
            cleaned = re.sub(r"!{2,}", "!", cleaned)
            cleaned = re.sub(r"\.{4,}", "...", cleaned)

            # 7. Remove zero-width / invisible Unicode chars
            cleaned = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", cleaned)

            opt_chars = len(cleaned)
            opt_tok = self._estimate_tokens(cleaned)
            saved = max(0, orig_tok - opt_tok)
            ratio = (saved / orig_tok) if orig_tok > 0 else 0.0

            data = {
                "original_text":               orig_text,
                "optimized_text":              cleaned,
                "original_char_count":         orig_chars,
                "optimized_char_count":        opt_chars,
                "estimated_original_tokens":   orig_tok,
                "estimated_optimized_tokens":  opt_tok,
                "tokens_saved":                saved,
                "compression_ratio":           round(ratio, 4),
            }
            return self._ok(data, ram_mb=2.0, tokens_saved=saved)

        except Exception as exc:
            return self._err(f"TokenWorker error: {exc}\n{traceback.format_exc()}")

    # ── Internal token estimator ──────────────────────────────────────────

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """
        Fast heuristic: ~4 chars per English token.
        Averages word-count and char-count estimates for ~10% accuracy.
        """
        if not text:
            return 0
        words = text.split()
        word_estimate = len(words) * 1.3
        char_estimate = len(text) / 3.8
        return max(1, int((word_estimate + char_estimate) / 2))
