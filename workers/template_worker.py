"""
workers/template_worker.py
--------------------------
Micro-Worker #8 — Chat Template Formatter

Responsibility: Format a (system_prompt, user_prompt) pair into the
correct chat-template string for a given model family, WITHOUT loading
the full tokenizer. Falls back to role-annotated plain text.

This is pure-Python with no ML weights. On systems with transformers
installed, it will use the tokenizer's apply_chat_template for accuracy.

RAM budget: ~20 MB (tokenizer only, no model weights)
"""

from __future__ import annotations

import traceback
from typing import Optional
from workers.base_worker import BaseWorker, WorkerResult


class TemplateWorker(BaseWorker):
    """
    Formats prompts using model-family chat templates.

    run() kwargs
    ------------
    family        : str  — model family: "llama" | "mistral" | "qwen" | "phi" | "gemma"
    user_prompt   : str  — user's message
    system_prompt : str  — optional system message (default "")
    hf_repo       : str  — optional HF repo id for tokenizer-based formatting
    use_tokenizer : bool — whether to attempt tokenizer-based formatting (default False)

    WorkerResult.data
    -----------------
    dict with keys:
      formatted_prompt : str  — ready-to-tokenize prompt string
      template_source  : str  — "tokenizer" | "builtin" | "fallback"
    """

    worker_id = "template"
    ram_budget_mb = 20.0
    requires_model = False

    # Built-in templates per model family
    _TEMPLATES = {
        "llama": (
            "<|begin_of_text|>"
            "{system}"
            "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
        "mistral": (
            "[INST] {system}{user} [/INST]"
        ),
        "qwen": (
            "<|im_start|>system\n{system_or_default}<|im_end|>\n"
            "<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "phi": (
            "<|system|>\n{system_or_default}<|end|>\n"
            "<|user|>\n{user}<|end|>\n"
            "<|assistant|>\n"
        ),
        "gemma": (
            "<start_of_turn>user\n{user}<end_of_turn>\n"
            "<start_of_turn>model\n"
        ),
    }

    _DEFAULT_SYSTEM = "You are a helpful AI assistant."

    def run(
        self,
        family: str = "llama",
        user_prompt: str = "",
        system_prompt: str = "",
        hf_repo: str = "",
        use_tokenizer: bool = False,
        **kwargs,
    ) -> WorkerResult:
        if not user_prompt:
            return self._err("TemplateWorker.run() requires 'user_prompt' kwarg.")

        try:
            # ── Try tokenizer-based formatting (optional, more accurate) ──
            if use_tokenizer and hf_repo:
                result = self._try_tokenizer(hf_repo, user_prompt, system_prompt)
                if result is not None:
                    return self._ok(
                        {"formatted_prompt": result, "template_source": "tokenizer"},
                        ram_mb=20.0,
                    )

            # ── Use built-in family templates ──────────────────────────────
            template = self._TEMPLATES.get(family.lower())
            if template:
                sys_text = system_prompt.strip() if system_prompt else ""
                sys_or_default = sys_text if sys_text else self._DEFAULT_SYSTEM

                # Build system block (some templates don't include system)
                if "{system}" in template:
                    if sys_text:
                        sys_block = (
                            f"<|start_header_id|>system<|end_header_id|>\n\n"
                            f"{sys_text}<|eot_id|>"
                        )
                    else:
                        sys_block = ""
                else:
                    sys_block = ""

                formatted = (
                    template
                    .replace("{system}", sys_block)
                    .replace("{system_or_default}", sys_or_default)
                    .replace("{user}", user_prompt)
                )
                return self._ok(
                    {"formatted_prompt": formatted, "template_source": "builtin"},
                    ram_mb=2.0,
                )

            # ── Fallback: plain role-annotated text ────────────────────────
            if system_prompt:
                fallback = f"System: {system_prompt}\n\nUser: {user_prompt}\nAssistant:"
            else:
                fallback = f"User: {user_prompt}\nAssistant:"

            return self._ok(
                {"formatted_prompt": fallback, "template_source": "fallback"},
                ram_mb=1.0,
            )

        except Exception as exc:
            return self._err(f"TemplateWorker error: {exc}\n{traceback.format_exc()}")

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _try_tokenizer(hf_repo: str, user_prompt: str, system_prompt: str) -> Optional[str]:
        """Attempt tokenizer-based apply_chat_template. Returns None on failure."""
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(
                hf_repo,
                trust_remote_code=True,
                use_fast=True,
            )
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            if hasattr(tok, "apply_chat_template"):
                return tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        except Exception:
            pass
        return None
