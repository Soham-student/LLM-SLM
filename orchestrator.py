"""
orchestrator.py
---------------
Central coordinator for the SLM Partitioner.

The Orchestrator owns all micro-worker instances, routes tasks between them,
and manages lifecycle (load → chat loop → unload) in a hardware-aware way.

Design goals:
  1. Zero hardware restrictions: always finds SOME working path regardless of tier.
  2. Memory-safe ordering: lower-RAM workers always run before model-bearing ones.
  3. Single import point for UI code — orchestrator communicates with ui.py,
     workers never import ui directly.
  4. Graceful degradation: if any worker fails, the orchestrator logs the error
     and applies a fallback instead of crashing.
"""

from __future__ import annotations

import time
from typing import Callable, List, Optional

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
from workers.base_worker import WorkerResult


class Orchestrator:
    """
    Manages all micro-SLM workers and routes tasks between them.

    Usage
    -----
    orch = Orchestrator()
    orch.scan_hardware()
    orch.select_model("qwen2.5-3b")
    orch.plan(force_cpu=False, prefer_quality=False)
    orch.load_model(progress_callback=print)
    orch.chat("Hello, who are you?", on_token=print)
    orch.unload()
    """

    def __init__(self):
        # ── Instantiate all workers ──────────────────────────────────────
        self._hw       = HardwareWorker()
        self._catalog  = CatalogWorker()
        self._quant    = QuantWorker()
        self._plan     = PlanWorker()
        self._kv       = KVBudgetWorker()
        self._token    = TokenWorker()
        self._rep      = RepetitionWorker()
        self._template = TemplateWorker()
        self._pruner   = PrunerWorker()
        self._dl       = DownloadWorker()
        self._loader   = ShardLoaderWorker()
        self._chat_w   = ChatWorker()

        # ── State shared between methods ─────────────────────────────────
        self.profile   = None   # SystemProfile
        self.model     = None   # ModelSpec
        self.plan      = None   # PartitionPlan
        self._messages: List[dict] = []   # multi-turn history

    # ─────────────────────────────────────────────
    # Step 1: Hardware scan
    # ─────────────────────────────────────────────

    def scan_hardware(self, log: Optional[Callable[[str], None]] = None) -> WorkerResult:
        """Detect hardware and store the SystemProfile."""
        _log(log, "HardwareWorker: scanning system…")
        res = self._hw.run()
        if res:
            self.profile = res.data
            _log(log, f"HardwareWorker: tier={self.profile.tier}  "
                      f"RAM={self.profile.ram_total_mb/1024:.1f}GB  "
                      f"VRAM={self.profile.total_vram_mb/1024:.1f}GB")
        else:
            _log(log, f"HardwareWorker error: {res.error}")
        return res

    # ─────────────────────────────────────────────
    # Step 2: Model selection
    # ─────────────────────────────────────────────

    def select_model(self, key: str, log: Optional[Callable[[str], None]] = None) -> WorkerResult:
        """Resolve a model key from the catalog."""
        _log(log, f"CatalogWorker: resolving model key '{key}'…")
        res = self._catalog.run(action="get", key=key)
        if res:
            self.model = res.data
            _log(log, f"CatalogWorker: {self.model.name}  ({self.model.params_billions}B params)")
        else:
            _log(log, f"CatalogWorker error: {res.error}")
        return res

    def list_models(self, log: Optional[Callable[[str], None]] = None) -> WorkerResult:
        """Return all catalog models."""
        return self._catalog.run(action="list")

    def recommend_models(self, log: Optional[Callable[[str], None]] = None) -> WorkerResult:
        """Return models recommended for detected hardware tier."""
        tier = self.profile.tier if self.profile else "mid"
        return self._catalog.run(action="recommend", tier=tier)

    # ─────────────────────────────────────────────
    # Step 3: Partition planning
    # ─────────────────────────────────────────────

    def compute_plan(
        self,
        force_cpu: bool = False,
        prefer_quality: bool = False,
        log: Optional[Callable[[str], None]] = None,
    ) -> WorkerResult:
        """Compute the optimal partition plan for the selected model."""
        if self.model is None:
            return _fail("No model selected. Call select_model() first.")
        if self.profile is None:
            return _fail("No hardware profile. Call scan_hardware() first.")

        _log(log, f"PlanWorker: planning shards for {self.model.name}…")
        res = self._plan.run(
            model=self.model,
            profile=self.profile,
            force_cpu=force_cpu,
            prefer_quality=prefer_quality,
        )
        if res:
            self.plan = res.data
            _log(log, f"PlanWorker: strategy={self.plan.strategy}  "
                      f"quant={self.plan.quant_bits}bit  "
                      f"shards={len(self.plan.shards)}")
        else:
            _log(log, f"PlanWorker error: {res.error}")
        return res

    # ─────────────────────────────────────────────
    # Step 4: KV budget (optional advisory)
    # ─────────────────────────────────────────────

    def get_kv_budget(
        self,
        prompt_tokens: int = 0,
        requested_max: int = 512,
    ) -> WorkerResult:
        """Get safe KV-cache token budget for the current model + hardware."""
        return self._kv.run(
            model=self.model,
            profile=self.profile,
            prompt_tokens=prompt_tokens,
            requested_max=requested_max,
        )

    # ─────────────────────────────────────────────
    # Step 5: Download check (optional)
    # ─────────────────────────────────────────────

    def check_cache(self, log: Optional[Callable[[str], None]] = None) -> WorkerResult:
        """Check if the selected model is already cached locally."""
        if self.model is None:
            return _fail("No model selected.")
        _log(log, f"DownloadWorker: checking cache for {self.model.hf_repo}…")
        return self._dl.run(action="check", hf_repo=self.model.hf_repo)

    def download_model(
        self,
        token: Optional[str] = None,
        log: Optional[Callable[[str], None]] = None,
    ) -> WorkerResult:
        """Pre-download the model snapshot to local cache."""
        if self.model is None:
            return _fail("No model selected.")
        _log(log, f"DownloadWorker: downloading {self.model.hf_repo}…")
        return self._dl.run(action="download", hf_repo=self.model.hf_repo, token=token)

    # ─────────────────────────────────────────────
    # Step 6: Load model weights
    # ─────────────────────────────────────────────

    def load_model(
        self,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> WorkerResult:
        """Load the model into memory according to the partition plan."""
        if self.model is None:
            return _fail("No model selected.")
        if self.plan is None:
            return _fail("No partition plan. Call compute_plan() first.")

        res = self._loader.run(
            action="load",
            model_spec=self.model,
            plan=self.plan,
            progress_callback=progress_callback,
        )
        return res

    # ─────────────────────────────────────────────
    # Step 7: Chat turn
    # ─────────────────────────────────────────────

    def chat(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = True,
        optimize_tokens: bool = True,
        on_token: Optional[Callable[[str], None]] = None,
        log: Optional[Callable[[str], None]] = None,
    ) -> WorkerResult:
        """
        Run a single chat turn through the full worker pipeline.

        Pipeline:
          TokenWorker → KVBudgetWorker → PrunerWorker →
          TemplateWorker → ChatWorker  →  WorkerResult
        """
        if not self._loader.is_loaded:
            return _fail("Model not loaded. Call load_model() first.")

        # ── 1. Token optimization ──────────────────────────────────────
        effective_prompt = user_input
        effective_max    = max_new_tokens

        if optimize_tokens:
            tok_res = self._token.run(
                prompt=user_input,
                system_prompt=system_prompt or "",
                max_tokens=max_new_tokens,
            )
            if tok_res:
                effective_prompt = tok_res.data["optimized_text"]
                _log(log, f"TokenWorker: saved {tok_res.data['tokens_saved']} tokens "
                          f"({tok_res.data['compression_ratio']*100:.1f}% compression)")

        # ── 2. KV-cache budget ─────────────────────────────────────────
        prompt_tok_est = len(effective_prompt.split()) * 1 + len(effective_prompt) // 4
        kv_res = self.get_kv_budget(
            prompt_tokens=prompt_tok_est,
            requested_max=max_new_tokens,
        )
        if kv_res:
            effective_max = kv_res.data["recommended_max_new"]
            _log(log, f"KVBudgetWorker: budget={kv_res.data['safe_context_tokens']}  "
                      f"max_new={effective_max}")

        # ── 3. Context pruning (multi-turn) ────────────────────────────
        self._messages.append({"role": "user", "content": effective_prompt})
        if len(self._messages) > 2:
            prune_res = self._pruner.run(
                messages=self._messages,
                max_budget_tokens=kv_res.data["safe_context_tokens"] if kv_res else 2048,
            )
            if prune_res and prune_res.data["turns_dropped"] > 0:
                self._messages = prune_res.data["pruned_messages"]
                _log(log, f"PrunerWorker: dropped {prune_res.data['turns_dropped']} old turns")

        # ── 4. Chat template formatting ────────────────────────────────
        tmpl_res = self._template.run(
            family=self.model.family if self.model else "llama",
            user_prompt=effective_prompt,
            system_prompt=system_prompt or "",
        )
        formatted = effective_prompt
        if tmpl_res:
            formatted = tmpl_res.data["formatted_prompt"]
            _log(log, f"TemplateWorker: source={tmpl_res.data['template_source']}")

        # ── 5. Generate ────────────────────────────────────────────────
        gen_res = self._chat_w.run(
            shard_loader=self._loader,
            model_spec=self.model,
            plan=self.plan,
            prompt=formatted,
            max_new_tokens=effective_max,
            temperature=temperature,
            stream=stream,
            on_token=on_token,
        )

        if gen_res:
            # Store assistant reply in history
            self._messages.append({
                "role": "assistant",
                "content": gen_res.data["generated_text"],
            })
            _log(log, f"ChatWorker: {gen_res.data['token_count']} tokens @ "
                      f"{gen_res.data['tokens_per_sec']} tok/s  "
                      f"stopped_early={gen_res.data['stopped_early']}")

        return gen_res

    # ─────────────────────────────────────────────
    # Lifecycle helpers
    # ─────────────────────────────────────────────

    def unload(self) -> WorkerResult:
        """Unload model weights and free all memory."""
        self._messages.clear()
        return self._loader.run(action="unload")

    def memory_status(self) -> WorkerResult:
        """Return current memory usage from the shard loader."""
        return self._loader.run(action="status")

    def clear_history(self):
        """Clear multi-turn conversation history."""
        self._messages.clear()

    def reset_plan(self):
        """Clear partition plan (forces recompute on next compute_plan call)."""
        self.plan = None

    @property
    def is_loaded(self) -> bool:
        return self._loader.is_loaded

    @property
    def worker_summary(self) -> list:
        """Return a list of all workers with their RAM budgets."""
        return [
            {"id": w.worker_id, "ram_mb": w.ram_budget_mb, "model": w.requires_model}
            for w in [
                self._hw, self._catalog, self._quant, self._plan,
                self._kv, self._token, self._rep, self._template,
                self._pruner, self._dl, self._loader, self._chat_w,
            ]
        ]


# ─────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────

def _log(cb: Optional[Callable], msg: str):
    if cb:
        cb(msg)


def _fail(msg: str) -> WorkerResult:
    from workers.base_worker import WorkerResult
    return WorkerResult(success=False, error=msg, worker_id="orchestrator")
