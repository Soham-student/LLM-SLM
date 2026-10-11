"""
workers/hardware_worker.py
--------------------------
Micro-Worker #1 — Hardware Detection

Responsibility: Detect CPU, RAM, GPU/VRAM, Disk and classify the system
into a performance tier. Pure-Python, no model weights.

RAM budget: ~30 MB (psutil + torch import only)
"""

from __future__ import annotations

import traceback
from workers.base_worker import BaseWorker, WorkerResult


class HardwareWorker(BaseWorker):
    """
    Scans local hardware and returns a SystemProfile.

    run() kwargs
    ------------
    (none required)

    WorkerResult.data
    -----------------
    system_detector.SystemProfile
    """

    worker_id = "hardware"
    ram_budget_mb = 30.0
    requires_model = False

    def run(self, **kwargs) -> WorkerResult:
        try:
            from system_detector import detect_system
            profile = detect_system()
            return self._ok(profile, ram_mb=30.0, tier=profile.tier)
        except Exception as exc:
            return self._err(f"Hardware detection failed: {exc}\n{traceback.format_exc()}")
