"""
workers/quant_worker.py
-----------------------
Micro-Worker #3 — Quantization Advisor

Responsibility: Given a ModelSpec and available memory, select the best
quantization level (4-bit / 8-bit / fp16) that fits. Pure-Python math.

RAM budget: ~10 MB
"""

from __future__ import annotations

import traceback
from workers.base_worker import BaseWorker, WorkerResult


class QuantWorker(BaseWorker):
    """
    Selects the highest quality quantization that fits in available memory.

    run() kwargs
    ------------
    model          : ModelSpec  — model specification
    usable_mem_mb  : float      — total usable memory (VRAM + RAM) in MB
    prefer_quality : bool       — prefer higher bit width (default False)

    WorkerResult.data
    -----------------
    dict with keys:
      quant_bits      : int     — chosen bits (4, 8, or 16)
      load_in_4bit    : bool
      load_in_8bit    : bool
      vram_needed_mb  : float   — estimated VRAM at chosen quant
      ram_needed_mb   : float   — estimated RAM at chosen quant
      label           : str     — human-readable label, e.g. "4-bit (Q4_K_M)"
    """

    worker_id = "quant"
    ram_budget_mb = 10.0
    requires_model = False

    # ── bit-width options from highest to lowest quality ──────────────────
    _HIGH_QUALITY_ORDER = [16, 8, 4]
    _LOW_MEM_ORDER      = [4, 8, 16]

    _LABELS = {
        4:  "4-bit (int4 / Q4_K_M)  — minimal quality loss, best compression",
        8:  "8-bit (int8 / Q8_0)    — near-lossless, 2× size vs 4-bit",
        16: "fp16                   — full half-precision, largest footprint",
    }

    def run(
        self,
        model=None,
        usable_mem_mb: float = 8192.0,
        prefer_quality: bool = False,
        **kwargs,
    ) -> WorkerResult:
        if model is None:
            return self._err("QuantWorker.run() requires 'model' (ModelSpec) kwarg.")

        try:
            order = self._HIGH_QUALITY_ORDER if prefer_quality else self._LOW_MEM_ORDER

            chosen_bits = 4  # hard fallback
            for bits in order:
                needed = model.vram_estimate_mb(bits)
                if needed <= usable_mem_mb:
                    chosen_bits = bits
                    break

            result = {
                "quant_bits":     chosen_bits,
                "load_in_4bit":   chosen_bits == 4,
                "load_in_8bit":   chosen_bits == 8,
                "vram_needed_mb": model.vram_estimate_mb(chosen_bits),
                "ram_needed_mb":  model.ram_estimate_mb(chosen_bits),
                "label":          self._LABELS.get(chosen_bits, f"{chosen_bits}-bit"),
            }
            return self._ok(result, ram_mb=2.0, chosen_bits=chosen_bits)

        except Exception as exc:
            return self._err(f"QuantWorker error: {exc}\n{traceback.format_exc()}")
