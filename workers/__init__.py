"""
workers/__init__.py
-------------------
Micro-SLM worker package.

Each worker is a self-contained unit that handles ONE specific task.
Workers are cheap to instantiate and garbage-collectable independently,
keeping memory usage minimal on low-end hardware.
"""

from workers.base_worker import BaseWorker, WorkerResult
from workers.hardware_worker import HardwareWorker
from workers.catalog_worker import CatalogWorker
from workers.quant_worker import QuantWorker
from workers.plan_worker import PlanWorker
from workers.kv_budget_worker import KVBudgetWorker
from workers.token_worker import TokenWorker
from workers.repetition_worker import RepetitionWorker
from workers.template_worker import TemplateWorker
from workers.pruner_worker import PrunerWorker
from workers.download_worker import DownloadWorker
from workers.shard_loader_worker import ShardLoaderWorker
from workers.chat_worker import ChatWorker

__all__ = [
    "BaseWorker",
    "WorkerResult",
    "HardwareWorker",
    "CatalogWorker",
    "QuantWorker",
    "PlanWorker",
    "KVBudgetWorker",
    "TokenWorker",
    "RepetitionWorker",
    "TemplateWorker",
    "PrunerWorker",
    "DownloadWorker",
    "ShardLoaderWorker",
    "ChatWorker",
]
