"""
model_runner.py
---------------
Loads and runs a model according to a PartitionPlan using
HuggingFace Transformers + Accelerate.

Supports:
  - 4-bit / 8-bit quantization via bitsandbytes
  - device_map for GPU/CPU/disk layer assignment
  - Streaming text generation
  - Chat template formatting
"""

from __future__ import annotations

import gc
import sys
import time
from typing import Generator, List, Optional

from model_partitioner import PartitionPlan
from model_catalog import ModelSpec


# ─────────────────────────────────────────────
# Optional imports
# ─────────────────────────────────────────────

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        TextIteratorStreamer,
        GenerationConfig,
        StoppingCriteriaList,
        StoppingCriteria,
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    import accelerate
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False


# ─────────────────────────────────────────────
# Custom stopping criteria
# ─────────────────────────────────────────────

class StopOnTokens(StoppingCriteria if HF_AVAILABLE else object):
    """Stop generation when one of the stop_ids is produced."""
    def __init__(self, stop_ids: List[int]):
        self.stop_ids = stop_ids

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return int(input_ids[0, -1]) in self.stop_ids


# ─────────────────────────────────────────────
# ModelRunner
# ─────────────────────────────────────────────

class ModelRunner:
    """
    Loads a model from HuggingFace Hub and runs it according
    to the PartitionPlan computed by the partitioner.
    """

    def __init__(self, model: ModelSpec, plan: PartitionPlan):
        if not HF_AVAILABLE:
            raise RuntimeError(
                "transformers is not installed. Run: pip install transformers accelerate"
            )
        self.model_spec = model
        self.plan = plan
        self._tokenizer = None
        self._model = None
        self._loaded = False

    # ── Public API ─────────────────────────────────────────────────────────

    def load(self, progress_callback=None) -> None:
        """Download (if needed) and load the model into memory."""
        if self._loaded:
            return

        hf_repo = self.model_spec.hf_repo

        # 1. Tokenizer
        if progress_callback:
            progress_callback("Downloading tokenizer…")
        self._tokenizer = AutoTokenizer.from_pretrained(
            hf_repo,
            trust_remote_code=True,
            use_fast=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # 2. BitsAndBytes quantization config
        bnb_config = None
        if TORCH_AVAILABLE:
            try:
                import bitsandbytes  # noqa: F401
                if self.plan.load_in_4bit:
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                elif self.plan.load_in_8bit:
                    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            except ImportError:
                pass  # bitsandbytes not available — fall back to fp16

        # 3. Choose dtype
        if TORCH_AVAILABLE and torch.cuda.is_available():
            dtype = torch.float16
        else:
            dtype = torch.float32

        # 4. Build device_map arg
        device_map = self.plan.device_map if ACCELERATE_AVAILABLE else None
        if not device_map:
            # Simple fallback
            if self.plan.strategy == "gpu_only" and TORCH_AVAILABLE and torch.cuda.is_available():
                device_map = "cuda"
            elif self.plan.strategy == "hybrid" and ACCELERATE_AVAILABLE:
                device_map = "auto"
            else:
                device_map = "cpu"

        # 5. Load model
        if progress_callback:
            progress_callback(f"Loading model '{self.model_spec.name}'…")

        load_kwargs = dict(
            pretrained_model_name_or_path=hf_repo,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

        if bnb_config is not None:
            load_kwargs["quantization_config"] = bnb_config

        if device_map is not None:
            load_kwargs["device_map"] = device_map

        self._model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

        if device_map is None or device_map == "cpu":
            self._model = self._model.to("cpu")

        self._model.eval()
        self._loaded = True

        if progress_callback:
            progress_callback("Model loaded and ready.")

    def unload(self) -> None:
        """Release GPU/CPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._loaded = False
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        system_prompt: Optional[str] = None,
        stream: bool = True,
    ) -> Generator[str, None, None]:
        """
        Generate text for a prompt. Yields tokens one-by-one when stream=True.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        # Format with chat template if available
        full_prompt = self._format_prompt(prompt, system_prompt)

        inputs = self._tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.model_spec.context_length - max_new_tokens,
        )

        # Move inputs to primary device
        primary_device = self._primary_device()
        input_ids = inputs["input_ids"].to(primary_device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(primary_device)

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        if stream and TORCH_AVAILABLE:
            yield from self._stream_generate(input_ids, attention_mask, gen_kwargs)
        else:
            yield from self._batch_generate(input_ids, attention_mask, gen_kwargs)

    # ── Private helpers ────────────────────────────────────────────────────

    def _format_prompt(self, user_prompt: str, system_prompt: Optional[str]) -> str:
        """Apply chat template if the tokenizer supports it."""
        if self._tokenizer is None:
            return user_prompt

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass

        # Fallback plain format
        if system_prompt:
            return f"System: {system_prompt}\n\nUser: {user_prompt}\nAssistant:"
        return f"User: {user_prompt}\nAssistant:"

    def _primary_device(self) -> str:
        """Return the device where input tensors should live."""
        for shard in self.plan.shards:
            if shard.device.startswith("cuda"):
                return shard.device
        return "cpu"

    def _stream_generate(self, input_ids, attention_mask, gen_kwargs: dict):
        """Streaming generation using TextIteratorStreamer."""
        from threading import Thread

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs_copy = {**gen_kwargs, "streamer": streamer}
        gen_kwargs_copy.pop("input_ids", None)
        gen_kwargs_copy.pop("attention_mask", None)

        thread = Thread(
            target=self._model.generate,
            kwargs={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                **{k: v for k, v in gen_kwargs_copy.items() if k != "streamer"},
                "streamer": streamer,
            },
        )
        thread.start()

        for token_text in streamer:
            yield token_text

        thread.join()

    def _batch_generate(self, input_ids, attention_mask, gen_kwargs: dict):
        """Non-streaming batch generation."""
        with (torch.no_grad() if TORCH_AVAILABLE else _null_ctx()):
            output_ids = self._model.generate(**gen_kwargs)
        new_ids = output_ids[0, input_ids.shape[1]:]
        text = self._tokenizer.decode(new_ids, skip_special_tokens=True)
        yield text

    def memory_usage_mb(self) -> dict:
        """Return current memory usage breakdown."""
        result = {}
        if TORCH_AVAILABLE:
            for i in range(torch.cuda.device_count()):
                result[f"cuda:{i}_allocated_mb"] = torch.cuda.memory_allocated(i) / (1024 ** 2)
                result[f"cuda:{i}_reserved_mb"]  = torch.cuda.memory_reserved(i)  / (1024 ** 2)
        import psutil, os
        proc = psutil.Process(os.getpid())
        result["ram_mb"] = proc.memory_info().rss / (1024 ** 2)
        return result


# ─────────────────────────────────────────────
# Null context manager fallback
# ─────────────────────────────────────────────

from contextlib import contextmanager

@contextmanager
def _null_ctx():
    yield
