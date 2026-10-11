"""
workers/base_worker.py
----------------------
Abstract base class shared by ALL micro-SLM workers.

Design principles:
  - Each worker is stateless after run() unless it holds a loaded model.
  - Workers expose a WorkerResult so the orchestrator can inspect
    success/failure without catching exceptions.
  - Every worker reports its own RAM budget so the orchestrator can
    decide execution order on very low-RAM systems.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class WorkerResult:
    """
    Universal result envelope returned by every worker.

    Attributes
    ----------
    success   : True if the worker completed without fatal error.
    data      : The primary output of the worker (type depends on worker).
    error     : Error message if success is False.
    worker_id : Name of the worker that produced this result.
    ram_used_mb : Estimated RAM consumed by this operation (for logging).
    """
    success: bool
    data: Any = None
    error: Optional[str] = None
    worker_id: str = ""
    ram_used_mb: float = 0.0
    metadata: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.success


class BaseWorker(ABC):
    """
    Abstract base class for all micro-SLM workers.

    Subclasses must implement:
      - worker_id  (class-level string)
      - ram_budget_mb  (estimated peak RAM for this worker in MB)
      - run(**kwargs) -> WorkerResult
    """

    # ── Subclass must set these ────────────────────────────────────────────
    worker_id: str = "base"
    ram_budget_mb: float = 50.0    # peak MB this worker will use
    requires_model: bool = False   # True only for model-bearing workers

    def __repr__(self) -> str:
        return f"<Worker:{self.worker_id} ram={self.ram_budget_mb}MB model={self.requires_model}>"

    @abstractmethod
    def run(self, **kwargs) -> WorkerResult:
        """
        Execute the worker's task.

        Parameters are worker-specific and passed by the orchestrator
        as keyword arguments.

        Returns a WorkerResult. Workers must NOT raise exceptions —
        they should catch all errors internally and return a failed result.
        """
        ...

    # ── Helpers available to all workers ──────────────────────────────────

    def _ok(self, data: Any, ram_mb: float = 0.0, **meta) -> WorkerResult:
        return WorkerResult(
            success=True,
            data=data,
            worker_id=self.worker_id,
            ram_used_mb=ram_mb,
            metadata=meta,
        )

    def _err(self, message: str) -> WorkerResult:
        return WorkerResult(
            success=False,
            error=message,
            worker_id=self.worker_id,
        )
