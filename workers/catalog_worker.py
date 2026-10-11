"""
workers/catalog_worker.py
-------------------------
Micro-Worker #2 — Model Catalog

Responsibility: Resolve model keys, list available models, and return
ModelSpec objects. Pure-Python, zero RAM overhead.

RAM budget: ~10 MB
"""

from __future__ import annotations

import traceback
from typing import Optional
from workers.base_worker import BaseWorker, WorkerResult


class CatalogWorker(BaseWorker):
    """
    Provides model metadata from the built-in catalog.

    run() kwargs
    ------------
    action : str   — "get" | "list" | "recommend" | "tags"
    key    : str   — model key (for action="get")
    tier   : str   — system tier (for action="recommend")
    tag    : str   — tag to filter by (for action="tags")

    WorkerResult.data
    -----------------
    "get"       → ModelSpec | None
    "list"      → List[ModelSpec]
    "recommend" → List[ModelSpec]
    "tags"      → List[ModelSpec]
    """

    worker_id = "catalog"
    ram_budget_mb = 10.0
    requires_model = False

    def run(self, action: str = "list", key: str = "", tier: str = "mid", tag: str = "", **kwargs) -> WorkerResult:
        try:
            from model_catalog import CATALOG, get_model, list_models, recommend_models, models_by_tag

            if action == "get":
                if not key:
                    return self._err("CatalogWorker.run(action='get') requires 'key' kwarg.")
                model = get_model(key)
                if model is None:
                    return self._err(
                        f"Model key '{key}' not found. "
                        f"Valid keys: {', '.join(CATALOG.keys())}"
                    )
                return self._ok(model, ram_mb=2.0, key=key)

            elif action == "list":
                return self._ok(list_models(), ram_mb=2.0)

            elif action == "recommend":
                return self._ok(recommend_models(tier), ram_mb=2.0, tier=tier)

            elif action == "tags":
                if not tag:
                    return self._err("CatalogWorker.run(action='tags') requires 'tag' kwarg.")
                return self._ok(models_by_tag(tag), ram_mb=2.0, tag=tag)

            else:
                return self._err(f"Unknown CatalogWorker action: '{action}'")

        except Exception as exc:
            return self._err(f"CatalogWorker error: {exc}\n{traceback.format_exc()}")
