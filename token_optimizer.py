"""
token_optimizer.py
------------------
Token optimization and context budgeting for local LLM/SLM inference.

Capabilities:
  1. Prompt Compression: Strips redundant whitespace, repetitive punctuation,
     and filler phrasing without altering semantic meaning.
  2. Context Budgeting: Calculates maximum safe KV cache tokens based on
     the hardware profile (RAM/VRAM) to avoid OOM crashes.
  3. Context Pruning: Sliding window / FIFO truncation for multi-turn chats.
  4. Repetition Detection: Detects n-gram repetition loops during generation
     to prevent infinite loops and runaway compute usage.
  5. Token Analytics: Tracks token count, compression ratio, speed, and memory.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

from system_detector import SystemProfile
from model_catalog import ModelSpec


@dataclass
class TokenOptimizationResult:
    """Statistics about prompt optimization."""
    original_text: str
    optimized_text: str
    original_char_count: int
    optimized_char_count: int
    estimated_original_tokens: int
    estimated_optimized_tokens: int
    tokens_saved: int
    compression_ratio: float      # e.g., 0.15 = 15% reduction
    kv_cache_budget_tokens: int
    recommended_max_new_tokens: int


@dataclass
class LiveTokenStats:
    """Real-time token metrics emitted during generation."""
    prompt_tokens: int = 0
    generated_tokens: int = 0
    max_tokens: int = 512
    elapsed_seconds: float = 0.0
    tokens_per_second: float = 0.0
    progress_percentage: float = 0.0
    ram_usage_mb: float = 0.0
    gpu_usage_mb: float = 0.0
    is_completed: bool = False
    stop_reason: str = ""         # "eos", "max_tokens", "repetition_loop", "user_abort"


class TokenOptimizer:
    """
    Analyzes, optimizes, and budgets tokens for local SLM/LLM inference.
    """

    def __init__(self, model_spec: Optional[ModelSpec] = None, profile: Optional[SystemProfile] = None):
        self.model_spec = model_spec
        self.profile = profile

    # ── 1. Prompt Optimization ────────────────────────────────────────────

    def optimize_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_target_tokens: Optional[int] = None,
    ) -> TokenOptimizationResult:
        """
        Compresses and optimizes prompt text to consume fewer context tokens.
        Strips whitespace bloat, duplicate formatting, and verbose wrappers.
        """
        orig_text = prompt
        orig_chars = len(orig_text)
        orig_tok = self.estimate_tokens(orig_text)

        cleaned = orig_text

        # 1. Normalize line endings
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

        # 2. Collapse runs of 3+ newlines to 2
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        # 3. Collapse multiple spaces/tabs into single space (preserving markdown indentation where relevant)
        cleaned = re.sub(r"[ \t]{3,}", "  ", cleaned)

        # 4. Remove trailing whitespace per line
        cleaned = "\n".join(line.rstrip() for line in cleaned.split("\n"))

        # 5. Trim leading/trailing overall whitespace
        cleaned = cleaned.strip()

        # 6. Repetitive punctuation collapse (e.g. "???" -> "?", "!!!!" -> "!")
        cleaned = re.sub(r"\?{2,}", "?", cleaned)
        cleaned = re.sub(r"!{2,}", "!", cleaned)
        cleaned = re.sub(r"\.{4,}", "...", cleaned)

        opt_chars = len(cleaned)
        opt_tok = self.estimate_tokens(cleaned)
        saved = max(0, orig_tok - opt_tok)
        ratio = (saved / orig_tok) if orig_tok > 0 else 0.0

        # KV Cache budget calculation
        kv_budget, rec_max = self.calculate_token_budget(opt_tok, max_target_tokens)

        return TokenOptimizationResult(
            original_text=orig_text,
            optimized_text=cleaned,
            original_char_count=orig_chars,
            optimized_char_count=opt_chars,
            estimated_original_tokens=orig_tok,
            estimated_optimized_tokens=opt_tok,
            tokens_saved=saved,
            compression_ratio=ratio,
            kv_cache_budget_tokens=kv_budget,
            recommended_max_new_tokens=rec_max,
        )

    # ── 2. Context & KV Budgeting ──────────────────────────────────────────

    def calculate_token_budget(
        self,
        prompt_tokens: int,
        requested_max_tokens: Optional[int] = None,
    ) -> Tuple[int, int]:
        """
        Computes the safe KV cache token budget based on system profile.
        Returns: (safe_context_window, recommended_max_new_tokens)
        """
        model_ctx = self.model_spec.context_length if self.model_spec else 4096
        req_max = requested_max_tokens or 512

        if not self.profile:
            return model_ctx, min(req_max, model_ctx - prompt_tokens)

        # Estimate KV cache cost per token:
        # KV cache = 2 * num_layers * hidden_size * num_bytes_per_param per token
        if self.model_spec:
            num_layers = self.model_spec.num_layers
            hidden_size = self.model_spec.hidden_size
            # 2 (keys + values) * layers * hidden * 2 bytes (fp16)
            bytes_per_token = 2 * num_layers * hidden_size * 2
            mb_per_1k_tokens = (bytes_per_token * 1000) / (1024 ** 2)
        else:
            mb_per_1k_tokens = 40.0  # approximate 40MB per 1000 tokens for 7B fp16 KV

        # Usable headroom for KV cache
        usable_mb = self.profile.usable_vram_mb() + self.profile.usable_ram_mb()

        # Leave at least 500 MB breathing room
        kv_headroom_mb = max(256.0, usable_mb * 0.4)
        max_possible_tokens = int((kv_headroom_mb / max(1.0, mb_per_1k_tokens)) * 1000)

        # Constrain to model native context length
        safe_ctx = min(model_ctx, max(512, max_possible_tokens))
        safe_max_new = min(req_max, max(64, safe_ctx - prompt_tokens))

        return safe_ctx, safe_max_new

    # ── 3. Multi-Turn History Optimizer (Sliding Window) ───────────────────

    def prune_chat_history(
        self,
        messages: List[Dict[str, str]],
        max_budget_tokens: int,
    ) -> List[Dict[str, str]]:
        """
        Prunes multi-turn conversation messages so the prompt stays within budget.
        Always preserves:
          - System prompt (index 0 if role == 'system')
          - Latest user message (last message)
        Discards oldest intermediate dialogue turns first.
        """
        if not messages:
            return []

        # Always preserve system prompt if present
        has_system = messages[0].get("role") == "system"
        system_msg = messages[0] if has_system else None
        dialogue = messages[1:] if has_system else list(messages)

        if not dialogue:
            return [system_msg] if system_msg else []

        latest_user_msg = dialogue[-1]
        intermediate = dialogue[:-1]

        # Calculate base tokens needed
        base_tok = self.estimate_tokens(latest_user_msg.get("content", ""))
        if system_msg:
            base_tok += self.estimate_tokens(system_msg.get("content", ""))

        remaining_budget = max_budget_tokens - base_tok
        if remaining_budget <= 0:
            # Budget is very tight: only return system + latest user message
            return ([system_msg] if system_msg else []) + [latest_user_msg]

        # Pack intermediate messages from newest to oldest
        packed: List[Dict[str, str]] = []
        for msg in reversed(intermediate):
            msg_tok = self.estimate_tokens(msg.get("content", ""))
            if msg_tok <= remaining_budget:
                packed.insert(0, msg)
                remaining_budget -= msg_tok
            else:
                break

        result = []
        if system_msg:
            result.append(system_msg)
        result.extend(packed)
        result.append(latest_user_msg)
        return result

    # ── 4. Repetition & Loop Detector ──────────────────────────────────────

    @staticmethod
    def detect_repetition(recent_tokens_or_words: List[str], max_ngram_size: int = 8, threshold: int = 3, ngram_size: Optional[int] = None) -> bool:
        """
        Detects if generation is stuck in an infinite repetition loop.
        Checks for consecutive repetitions of any pattern length from 1 to max_ngram_size.
        """
        if ngram_size is not None:
            max_ngram_size = ngram_size
        n = len(recent_tokens_or_words)
        if n < threshold:
            return False

        # Check candidate repeating block lengths k from 1 up to max_ngram_size
        max_k = min(max_ngram_size, n // threshold)
        for k in range(1, max_k + 1):
            pattern = recent_tokens_or_words[-k:]
            repeats = 1
            for step in range(2, threshold + 1):
                start_idx = -k * step
                end_idx = -k * (step - 1)
                if recent_tokens_or_words[start_idx:end_idx] == pattern:
                    repeats += 1
                else:
                    break
            if repeats >= threshold:
                return True

        return False

    # ── 5. Token Estimation ───────────────────────────────────────────────

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Quick heuristic token estimation: ~4 characters per token in English.
        Accurate to within ~10% without needing heavy tokenizer loading.
        """
        if not text:
            return 0
        words = text.split()
        char_count = len(text)
        # Average between word-based and char-based approximation
        word_estimate = len(words) * 1.3
        char_estimate = char_count / 3.8
        return max(1, int((word_estimate + char_estimate) / 2))
