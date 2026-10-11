"""
workers/kv_budget_worker.py
---------------------------
Micro-Worker #5 — KV-Cache Budget Calculator

Responsibility: Calculate the safe KV-cache token budget given the
hardware profile and model spec. Pure-Python math.

RAM budget: ~10 MB
"""

from __future__ import annotations

import traceback
from typing import Optional
from workers.base_worker import BaseWorker, WorkerResult


class KVBudgetWorker(BaseWorker):
    """
    Computes how many tokens fit safely in the KV cache given current hardware.

    run() kwargs
    ------------
    model            : ModelSpec
    profile          : SystemProfile
    prompt_tokens    : int   — estimated tokens already in prompt
    requested_max    : int   — user-requested max new tokens (default 512)

    WorkerResult.data
    -----------------
    dict with keys:
      safe_context_tokens   : int  — total context window safely usable
      recommended_max_new   : int  — recommended max_new_tokens
      kv_mb_per_1k_tokens   : float — MB cost per 1K tokens of KV cache
      kv_headroom_mb        : float — MB allocated for KV cache
    """

    worker_id = "kv_budget"
    ram_budget_mb = 10.0
    requires_model = False

    def run(
        self,
        model=None,
        profile=None,
        prompt_tokens: int = 0,
        requested_max: int = 512,
        **kwargs,
    ) -> WorkerResult:
        if model is None:
            return self._err("KVBudgetWorker.run() requires 'model' kwarg.")

        try:
            model_ctx = model.context_length

            # Per-token KV cache cost (keys + values in fp16)
            # Formula: 2 (K+V) × num_layers × hidden_size × 2 (bytes for fp16)
            bytes_per_token = 2 * model.num_layers * model.hidden_size * 2
            mb_per_1k = (bytes_per_token * 1000) / (1024 ** 2)

            if profile is not None:
                usable_mb = profile.usable_vram_mb() + profile.usable_ram_mb()
            else:
                usable_mb = 4096.0  # fallback: assume 4 GB

            # Keep 40% of usable memory for KV cache, minimum 256 MB
            kv_headroom_mb = max(256.0, usable_mb * 0.4)
            max_possible = int((kv_headroom_mb / max(1.0, mb_per_1k)) * 1000)

            safe_ctx = min(model_ctx, max(512, max_possible))
            safe_max_new = min(requested_max, max(64, safe_ctx - prompt_tokens))

            data = {
                "safe_context_tokens":  safe_ctx,
                "recommended_max_new":  safe_max_new,
                "kv_mb_per_1k_tokens":  round(mb_per_1k, 3),
                "kv_headroom_mb":       round(kv_headroom_mb, 1),
            }
            return self._ok(data, ram_mb=2.0)

        except Exception as exc:
            return self._err(f"KVBudgetWorker error: {exc}\n{traceback.format_exc()}")
