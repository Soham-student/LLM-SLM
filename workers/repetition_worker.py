"""
workers/repetition_worker.py
----------------------------
Micro-Worker #7 — Repetition / Loop Detector

Responsibility: Detect if text generation is stuck in an n-gram repetition
loop so the orchestrator can stop generation early and free compute.
Pure-Python, no model weights, minimal RAM.

RAM budget: ~5 MB
"""

from __future__ import annotations

import traceback
from typing import List
from workers.base_worker import BaseWorker, WorkerResult


class RepetitionWorker(BaseWorker):
    """
    Detects n-gram repetition loops in streaming generation output.

    run() kwargs
    ------------
    words          : List[str] — recent generated words (sliding window)
    ngram_size     : int       — pattern size to test (default 4)
    threshold      : int       — how many consecutive repeats = loop (default 3)

    WorkerResult.data
    -----------------
    dict with keys:
      is_loop       : bool   — True if a repetition loop was detected
      pattern       : list   — the repeating n-gram (if detected)
      ngram_size    : int
      threshold     : int
    """

    worker_id = "repetition"
    ram_budget_mb = 5.0
    requires_model = False

    def run(
        self,
        words: List[str] = None,
        ngram_size: int = 4,
        threshold: int = 3,
        **kwargs,
    ) -> WorkerResult:
        if words is None:
            return self._err("RepetitionWorker.run() requires 'words' (List[str]) kwarg.")

        try:
            is_loop, pattern = self._detect(words, ngram_size, threshold)
            data = {
                "is_loop":    is_loop,
                "pattern":    pattern,
                "ngram_size": ngram_size,
                "threshold":  threshold,
            }
            return self._ok(data, ram_mb=1.0, is_loop=is_loop)
        except Exception as exc:
            return self._err(f"RepetitionWorker error: {exc}\n{traceback.format_exc()}")

    # ── Core algorithm ────────────────────────────────────────────────────

    @staticmethod
    def _detect(
        words: List[str],
        max_ngram_size: int = 8,
        threshold: int = 3,
    ):
        """
        Slide over candidate block lengths k from 1 to max_ngram_size.
        If the last k words repeat `threshold` times consecutively → loop.

        Returns (is_loop: bool, repeating_pattern: List[str])
        """
        n = len(words)
        if n < threshold:
            return False, []

        max_k = min(max_ngram_size, n // threshold)
        for k in range(1, max_k + 1):
            pattern = words[-k:]
            repeats = 1
            for step in range(2, threshold + 1):
                start = -k * step
                end   = -k * (step - 1)
                if words[start:end] == pattern:
                    repeats += 1
                else:
                    break
            if repeats >= threshold:
                return True, pattern

        return False, []
