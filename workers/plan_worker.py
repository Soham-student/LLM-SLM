"""
workers/plan_worker.py
----------------------
Micro-Worker #4 — Partition Planner

Responsibility: Given a ModelSpec and SystemProfile, compute how to split
the model into layer-shards across GPU / CPU / Disk. Pure-Python math.

RAM budget: ~30 MB
"""

from __future__ import annotations

import traceback
from workers.base_worker import BaseWorker, WorkerResult


class PlanWorker(BaseWorker):
    """
    Computes the optimal PartitionPlan for a given model + hardware profile.

    run() kwargs
    ------------
    model          : ModelSpec
    profile        : SystemProfile
    prefer_quality : bool  (default False)
    force_cpu      : bool  (default False)

    WorkerResult.data
    -----------------
    model_partitioner.PartitionPlan
    """

    worker_id = "plan"
    ram_budget_mb = 30.0
    requires_model = False

    def run(
        self,
        model=None,
        profile=None,
        prefer_quality: bool = False,
        force_cpu: bool = False,
        **kwargs,
    ) -> WorkerResult:
        if model is None:
            return self._err("PlanWorker.run() requires 'model' kwarg.")
        if profile is None:
            return self._err("PlanWorker.run() requires 'profile' kwarg.")

        try:
            from model_partitioner import partition_model
            plan = partition_model(
                model,
                profile,
                prefer_quality=prefer_quality,
                force_cpu=force_cpu,
            )
            return self._ok(
                plan,
                ram_mb=10.0,
                strategy=plan.strategy,
                quant_bits=plan.quant_bits,
                is_feasible=plan.is_feasible,
                num_shards=len(plan.shards),
            )
        except Exception as exc:
            return self._err(f"PlanWorker error: {exc}\n{traceback.format_exc()}")
