"""
workers/shard_loader_worker.py
------------------------------
Micro-Worker #11 — Shard Loader

Responsibility: Load model weights from HuggingFace Hub into memory
according to a PartitionPlan (GPU/CPU/disk shard layout).

This is the ONLY worker that actually loads neural network weights.
It holds the tokenizer and model objects until explicitly unloaded.

RAM budget: varies by model (set dynamically from PartitionPlan)
"""

from __future__ import annotations

import gc
import traceback
from typing import Any, Callable, Optional
from workers.base_worker import BaseWorker, WorkerResult


class ShardLoaderWorker(BaseWorker):
    """
    Loads a model into memory according to a PartitionPlan.

    run() kwargs
    ------------
    action          : str          — "load" | "unload" | "status"
    model_spec      : ModelSpec    — model specification
    plan            : PartitionPlan — partition plan from PlanWorker
    progress_callback: Callable[[str], None] — optional progress logger

    WorkerResult.data  (action="load")
    -----------------
    dict:
      tokenizer     : AutoTokenizer  — ready tokenizer
      model         : AutoModel      — loaded model
      load_time_sec : float

    WorkerResult.data  (action="unload")
    -----------------
    dict:
      freed : True

    WorkerResult.data  (action="status")
    -----------------
    dict:
      is_loaded     : bool
      ram_mb        : float
      gpu_mb        : dict  — {device: mb}
    """

    worker_id = "shard_loader"
    ram_budget_mb = 0.0   # set dynamically from plan.total_memory_mb
    requires_model = True

    def __init__(self):
        self._tokenizer = None
        self._model = None
        self._loaded = False
        self._loaded_repo: Optional[str] = None

    # ── Public run() ──────────────────────────────────────────────────────

    def run(
        self,
        action: str = "load",
        model_spec=None,
        plan=None,
        progress_callback: Optional[Callable[[str], None]] = None,
        **kwargs,
    ) -> WorkerResult:
        if action == "load":
            return self._load(model_spec, plan, progress_callback)
        elif action == "unload":
            return self._unload()
        elif action == "status":
            return self._status()
        else:
            return self._err(f"Unknown ShardLoaderWorker action: '{action}'")

    # ── Load ─────────────────────────────────────────────────────────────

    def _load(self, model_spec, plan, progress_callback) -> WorkerResult:
        if model_spec is None:
            return self._err("ShardLoaderWorker load() requires 'model_spec' kwarg.")
        if plan is None:
            return self._err("ShardLoaderWorker load() requires 'plan' kwarg.")

        if self._loaded and self._loaded_repo == model_spec.hf_repo:
            return self._ok(
                {"tokenizer": self._tokenizer, "model": self._model, "load_time_sec": 0.0},
                ram_mb=plan.total_memory_mb,
            )

        try:
            _cb = progress_callback or (lambda m: None)
            import time

            # ── Check library availability ─────────────────────────────
            try:
                import torch
                TORCH_OK = True
            except ImportError:
                TORCH_OK = False

            try:
                from transformers import (
                    AutoTokenizer, AutoModelForCausalLM,
                    BitsAndBytesConfig,
                )
            except ImportError:
                return self._err(
                    "transformers not installed. Run: pip install transformers accelerate"
                )

            try:
                import accelerate
                ACCEL_OK = True
            except ImportError:
                ACCEL_OK = False

            hf_repo = model_spec.hf_repo
            t0 = time.time()

            # ── 1. Tokenizer ───────────────────────────────────────────
            _cb("Loading tokenizer…")
            self._tokenizer = AutoTokenizer.from_pretrained(
                hf_repo,
                trust_remote_code=True,
                use_fast=True,
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # ── 2. Quantization config ─────────────────────────────────
            bnb_config = None
            if TORCH_OK:
                try:
                    import bitsandbytes  # noqa: F401
                    if plan.load_in_4bit:
                        bnb_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True,
                        )
                    elif plan.load_in_8bit:
                        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                except ImportError:
                    pass

            # ── 3. dtype ───────────────────────────────────────────────
            if TORCH_OK and torch.cuda.is_available():
                dtype = torch.float16
            elif TORCH_OK:
                dtype = torch.float32
            else:
                dtype = None

            # ── 4. device_map ──────────────────────────────────────────
            device_map = plan.device_map if ACCEL_OK else None
            if not device_map:
                if plan.strategy == "gpu_only" and TORCH_OK and torch.cuda.is_available():
                    device_map = "cuda"
                elif plan.strategy == "hybrid" and ACCEL_OK:
                    device_map = "auto"
                else:
                    device_map = "cpu"

            # ── 5. Load model ──────────────────────────────────────────
            _cb(f"Loading '{model_spec.name}' ({plan.quant_bits}-bit, strategy={plan.strategy})…")

            load_kwargs: dict = dict(
                pretrained_model_name_or_path=hf_repo,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            if dtype is not None:
                load_kwargs["torch_dtype"] = dtype
            if bnb_config is not None:
                load_kwargs["quantization_config"] = bnb_config
            if device_map is not None:
                load_kwargs["device_map"] = device_map

            self._model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

            if device_map in (None, "cpu"):
                self._model = self._model.to("cpu")

            self._model.eval()
            self._loaded = True
            self._loaded_repo = hf_repo
            elapsed = time.time() - t0

            self.ram_budget_mb = plan.total_memory_mb
            _cb(f"Model loaded in {elapsed:.1f}s.")

            return self._ok(
                {"tokenizer": self._tokenizer, "model": self._model, "load_time_sec": elapsed},
                ram_mb=plan.total_memory_mb,
            )

        except Exception as exc:
            self._loaded = False
            return self._err(f"ShardLoaderWorker load failed: {exc}\n{traceback.format_exc()}")

    # ── Unload ────────────────────────────────────────────────────────────

    def _unload(self) -> WorkerResult:
        try:
            if self._model is not None:
                del self._model
                self._model = None
            if self._tokenizer is not None:
                del self._tokenizer
                self._tokenizer = None
            self._loaded = False
            self._loaded_repo = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            return self._ok({"freed": True}, ram_mb=0.0)
        except Exception as exc:
            return self._err(f"Unload failed: {exc}")

    # ── Status ────────────────────────────────────────────────────────────

    def _status(self) -> WorkerResult:
        gpu_mb: dict = {}
        ram_mb = 0.0
        try:
            import torch, psutil, os
            for i in range(torch.cuda.device_count()):
                gpu_mb[f"cuda:{i}"] = torch.cuda.memory_allocated(i) / (1024 ** 2)
            proc = psutil.Process(os.getpid())
            ram_mb = proc.memory_info().rss / (1024 ** 2)
        except Exception:
            pass

        return self._ok(
            {"is_loaded": self._loaded, "ram_mb": ram_mb, "gpu_mb": gpu_mb},
            ram_mb=ram_mb,
        )

    # ── Convenience accessors ─────────────────────────────────────────────

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    @property
    def is_loaded(self) -> bool:
        return self._loaded
