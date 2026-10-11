"""
workers/chat_worker.py
----------------------
Micro-Worker #12 — Chat / Text Generation

Responsibility: Given a loaded tokenizer + model (from ShardLoaderWorker),
format a prompt and generate a streamed or batched response. Also calls
RepetitionWorker inline to detect and stop generation loops.

This worker does NOT hold model weights itself — it borrows them from
a ShardLoaderWorker instance. This keeps the memory boundary clear.

RAM budget: ~10 MB overhead (uses shard_loader's already-allocated memory)
"""

from __future__ import annotations

import traceback
from typing import Callable, Generator, List, Optional
from workers.base_worker import BaseWorker, WorkerResult
from workers.repetition_worker import RepetitionWorker


class ChatWorker(BaseWorker):
    """
    Runs inference using a pre-loaded model/tokenizer pair.

    run() kwargs
    ------------
    shard_loader      : ShardLoaderWorker — must have is_loaded=True
    model_spec        : ModelSpec
    plan              : PartitionPlan
    prompt            : str           — formatted prompt (from TemplateWorker)
    max_new_tokens    : int           (default 512)
    temperature       : float         (default 0.7)
    top_p             : float         (default 0.9)
    top_k             : int           (default 50)
    repetition_penalty: float         (default 1.1)
    stream            : bool          (default True)
    on_token          : Callable[[str], None]  — optional streaming callback

    WorkerResult.data
    -----------------
    dict:
      generated_text   : str
      token_count      : int
      stopped_early    : bool   — True if repetition loop detected
      elapsed_sec      : float
      tokens_per_sec   : float
    """

    worker_id = "chat"
    ram_budget_mb = 10.0   # overhead only; model memory is in shard_loader
    requires_model = True

    def __init__(self):
        self._rep_worker = RepetitionWorker()

    def run(
        self,
        shard_loader=None,
        model_spec=None,
        plan=None,
        prompt: str = "",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        stream: bool = True,
        on_token: Optional[Callable[[str], None]] = None,
        **kwargs,
    ) -> WorkerResult:
        # ── Guard: need a loaded shard_loader ────────────────────────────
        if shard_loader is None or not shard_loader.is_loaded:
            return self._err(
                "ChatWorker.run() requires a loaded ShardLoaderWorker "
                "passed as 'shard_loader'."
            )
        if not prompt:
            return self._err("ChatWorker.run() requires 'prompt' kwarg.")

        try:
            import time
            TORCH_OK = self._torch_ok()

            tokenizer = shard_loader.tokenizer
            model     = shard_loader.model

            # ── Tokenize ───────────────────────────────────────────────
            ctx_len = model_spec.context_length if model_spec else 4096
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max(1, ctx_len - max_new_tokens),
            )

            primary_dev = self._primary_device(plan)
            if TORCH_OK:
                import torch
                input_ids     = inputs["input_ids"].to(primary_dev)
                attn_mask     = inputs.get("attention_mask")
                if attn_mask is not None:
                    attn_mask = attn_mask.to(primary_dev)
            else:
                input_ids = inputs["input_ids"]
                attn_mask = inputs.get("attention_mask")

            gen_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
            )

            # ── Generate ───────────────────────────────────────────────
            t0 = time.time()
            generated_text = ""
            token_count = 0
            stopped_early = False
            recent_words: List[str] = []

            if stream and TORCH_OK:
                for chunk in self._stream_gen(model, tokenizer, input_ids, attn_mask, gen_kwargs):
                    generated_text += chunk
                    words = chunk.split()
                    if words:
                        recent_words.extend(words)
                        if len(recent_words) > 30:
                            recent_words = recent_words[-30:]
                        # Inline repetition check
                        rep_res = self._rep_worker.run(words=recent_words, ngram_size=4, threshold=3)
                        if rep_res.success and rep_res.data["is_loop"]:
                            stopped_early = True
                            break
                    token_count += max(1, len(words))
                    if on_token:
                        on_token(chunk)
            else:
                full_text = self._batch_gen(model, tokenizer, input_ids, attn_mask, gen_kwargs, TORCH_OK)
                generated_text = full_text
                token_count = len(full_text.split())
                if on_token:
                    on_token(full_text)

            elapsed = time.time() - t0
            tps = round(token_count / max(0.01, elapsed), 2)

            return self._ok(
                {
                    "generated_text": generated_text,
                    "token_count":    token_count,
                    "stopped_early":  stopped_early,
                    "elapsed_sec":    round(elapsed, 2),
                    "tokens_per_sec": tps,
                },
                ram_mb=5.0,
                tokens_per_sec=tps,
            )

        except Exception as exc:
            return self._err(f"ChatWorker error: {exc}\n{traceback.format_exc()}")

    # ── Streaming generation ──────────────────────────────────────────────

    @staticmethod
    def _stream_gen(model, tokenizer, input_ids, attn_mask, gen_kwargs) -> Generator[str, None, None]:
        from threading import Thread
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        thread_kwargs = {
            "input_ids":      input_ids,
            "attention_mask": attn_mask,
            "streamer":       streamer,
            **{k: v for k, v in gen_kwargs.items()
               if k not in ("input_ids", "attention_mask")},
        }
        t = Thread(target=model.generate, kwargs=thread_kwargs)
        t.start()
        for token_text in streamer:
            yield token_text
        t.join()

    # ── Batch generation ──────────────────────────────────────────────────

    @staticmethod
    def _batch_gen(model, tokenizer, input_ids, attn_mask, gen_kwargs, torch_ok: bool) -> str:
        from contextlib import contextmanager

        @contextmanager
        def maybe_no_grad():
            if torch_ok:
                import torch
                with torch.no_grad():
                    yield
            else:
                yield

        with maybe_no_grad():
            output_ids = model.generate(**gen_kwargs)
        new_ids = output_ids[0, input_ids.shape[1]:]
        return tokenizer.decode(new_ids, skip_special_tokens=True)

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _torch_ok() -> bool:
        try:
            import torch
            return True
        except ImportError:
            return False

    @staticmethod
    def _primary_device(plan) -> str:
        """Return the first GPU device in the plan, or 'cpu'."""
        if plan is None:
            return "cpu"
        for shard in plan.shards:
            if shard.device.startswith("cuda"):
                return shard.device
        return "cpu"
