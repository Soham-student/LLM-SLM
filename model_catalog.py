"""
model_catalog.py
----------------
Catalogue of well-known open-source models with their specifications
(parameter count, layer count, memory requirements per quantization level).
Used by the partitioner to recommend and plan splitting strategies.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class QuantConfig:
    """Memory footprint for a particular quantization level."""
    bits: int                        # 2, 4, 8, 16, 32
    label: str                       # e.g. "Q4_K_M", "int8", "fp16"
    bytes_per_param: float           # average bytes per parameter
    quality_loss: str                # "none" | "minimal" | "moderate" | "significant"
    notes: str = ""


@dataclass
class ModelSpec:
    """Specifications for a known LLM."""
    name: str                         # Display name
    hf_repo: str                      # Hugging Face repo id
    family: str                       # "llama" | "mistral" | "qwen" | "phi" | "gemma" | etc.
    params_billions: float            # e.g. 7.0 for 7B
    num_layers: int                   # transformer block count
    hidden_size: int
    num_heads: int
    context_length: int
    quant_options: List[QuantConfig] = field(default_factory=list)
    license: str = "open"
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # ── derived ──────────────────────────────
    def vram_estimate_mb(self, bits: int = 4) -> float:
        """Rough VRAM estimate in MB for given quantization bits."""
        q = self._get_quant(bits)
        bpp = q.bytes_per_param if q else (bits / 8)
        # params * bytes + ~15% overhead for KV cache & activations
        base = self.params_billions * 1e9 * bpp / (1024 ** 2)
        return base * 1.15

    def ram_estimate_mb(self, bits: int = 4) -> float:
        """Estimate RAM needed when running on CPU only."""
        return self.vram_estimate_mb(bits) * 1.2   # extra for CPU overhead

    def _get_quant(self, bits: int) -> Optional[QuantConfig]:
        for q in self.quant_options:
            if q.bits == bits:
                return q
        return None

    def bytes_per_layer_mb(self, bits: int = 4) -> float:
        """Approximate memory per transformer layer in MB."""
        if self.num_layers == 0:
            return 0.0
        return self.vram_estimate_mb(bits) / self.num_layers


# ─────────────────────────────────────────────
# Standard quant configs
# ─────────────────────────────────────────────

_Q4 = QuantConfig(4, "Q4_K_M", 0.5,  "minimal",     "Good balance of size vs quality")
_Q8 = QuantConfig(8, "Q8_0",   1.0,  "none",        "Near-lossless, 2x size of Q4")
_F16 = QuantConfig(16,"fp16",  2.0,  "none",        "Full half precision — large")
_F32 = QuantConfig(32,"fp32",  4.0,  "none",        "Full precision — very large")
_Q2 = QuantConfig(2, "Q2_K",  0.25, "significant", "Very small, noticeable quality drop")
_INT8 = QuantConfig(8, "int8", 1.0,  "none",        "Transformers bitsandbytes int8")
_INT4 = QuantConfig(4, "int4", 0.5,  "minimal",     "Transformers bitsandbytes int4 (nf4)")


# ─────────────────────────────────────────────
# Model catalog
# ─────────────────────────────────────────────

CATALOG: Dict[str, ModelSpec] = {

    # ── Tiny / Phone-grade ────────────────────
    "phi-3.5-mini": ModelSpec(
        name="Phi-3.5 Mini Instruct",
        hf_repo="microsoft/Phi-3.5-mini-instruct",
        family="phi",
        params_billions=3.8,
        num_layers=32,
        hidden_size=3072,
        num_heads=32,
        context_length=128_000,
        quant_options=[_INT4, _INT8, _F16],
        license="MIT",
        description="Microsoft's compact 3.8B model, excellent for low-end hardware.",
        tags=["small", "instruct", "windows-friendly"],
    ),

    "qwen2.5-0.5b": ModelSpec(
        name="Qwen2.5 0.5B Instruct",
        hf_repo="Qwen/Qwen2.5-0.5B-Instruct",
        family="qwen",
        params_billions=0.5,
        num_layers=24,
        hidden_size=896,
        num_heads=14,
        context_length=32_768,
        quant_options=[_INT4, _INT8, _F16],
        license="Apache 2.0",
        description="Ultra-tiny 0.5B model, runs on almost any hardware.",
        tags=["tiny", "instruct", "cpu-friendly"],
    ),

    "qwen2.5-1.5b": ModelSpec(
        name="Qwen2.5 1.5B Instruct",
        hf_repo="Qwen/Qwen2.5-1.5B-Instruct",
        family="qwen",
        params_billions=1.5,
        num_layers=28,
        hidden_size=1536,
        num_heads=12,
        context_length=32_768,
        quant_options=[_INT4, _INT8, _F16],
        license="Apache 2.0",
        description="Good 1.5B model for ultra-low RAM setups.",
        tags=["tiny", "instruct", "cpu-friendly"],
    ),

    "qwen2.5-3b": ModelSpec(
        name="Qwen2.5 3B Instruct",
        hf_repo="Qwen/Qwen2.5-3B-Instruct",
        family="qwen",
        params_billions=3.0,
        num_layers=36,
        hidden_size=2048,
        num_heads=16,
        context_length=32_768,
        quant_options=[_INT4, _INT8, _F16],
        license="Apache 2.0",
        description="Solid 3B model that performs well under 8GB RAM.",
        tags=["small", "instruct"],
    ),

    "gemma-2-2b": ModelSpec(
        name="Gemma 2 2B Instruct",
        hf_repo="google/gemma-2-2b-it",
        family="gemma",
        params_billions=2.6,
        num_layers=26,
        hidden_size=2304,
        num_heads=8,
        context_length=8_192,
        quant_options=[_INT4, _INT8, _F16],
        license="Gemma",
        description="Google's efficient 2B model, great for reasoning tasks.",
        tags=["small", "instruct", "google"],
    ),

    # ── Mid-range / 7B class ──────────────────
    "llama-3.2-3b": ModelSpec(
        name="Llama 3.2 3B Instruct",
        hf_repo="meta-llama/Llama-3.2-3B-Instruct",
        family="llama",
        params_billions=3.2,
        num_layers=28,
        hidden_size=3072,
        num_heads=24,
        context_length=131_072,
        quant_options=[_INT4, _INT8, _F16],
        license="Llama 3.2",
        description="Meta's compact 3B model with long context (128K).",
        tags=["small", "instruct", "long-context"],
    ),

    "llama-3.1-8b": ModelSpec(
        name="Llama 3.1 8B Instruct",
        hf_repo="meta-llama/Meta-Llama-3.1-8B-Instruct",
        family="llama",
        params_billions=8.0,
        num_layers=32,
        hidden_size=4096,
        num_heads=32,
        context_length=131_072,
        quant_options=[_INT4, _INT8, _F16],
        license="Llama 3.1",
        description="Popular 8B model, good quality for mid-range hardware.",
        tags=["mid", "instruct", "long-context"],
    ),

    "mistral-7b": ModelSpec(
        name="Mistral 7B Instruct v0.3",
        hf_repo="mistralai/Mistral-7B-Instruct-v0.3",
        family="mistral",
        params_billions=7.3,
        num_layers=32,
        hidden_size=4096,
        num_heads=32,
        context_length=32_768,
        quant_options=[_Q4, _Q8, _INT4, _INT8, _F16],
        license="Apache 2.0",
        description="Efficient 7B model with sliding window attention.",
        tags=["mid", "instruct"],
    ),

    "qwen2.5-7b": ModelSpec(
        name="Qwen2.5 7B Instruct",
        hf_repo="Qwen/Qwen2.5-7B-Instruct",
        family="qwen",
        params_billions=7.6,
        num_layers=28,
        hidden_size=3584,
        num_heads=28,
        context_length=131_072,
        quant_options=[_INT4, _INT8, _F16],
        license="Apache 2.0",
        description="Strong 7B model with very long context window.",
        tags=["mid", "instruct", "long-context"],
    ),

    "gemma-2-9b": ModelSpec(
        name="Gemma 2 9B Instruct",
        hf_repo="google/gemma-2-9b-it",
        family="gemma",
        params_billions=9.2,
        num_layers=42,
        hidden_size=3584,
        num_heads=16,
        context_length=8_192,
        quant_options=[_INT4, _INT8, _F16],
        license="Gemma",
        description="Google's capable 9B model, surprisingly efficient.",
        tags=["mid", "instruct", "google"],
    ),

    # ── High-end / 13B+ ───────────────────────
    "llama-3.1-70b": ModelSpec(
        name="Llama 3.1 70B Instruct",
        hf_repo="meta-llama/Meta-Llama-3.1-70B-Instruct",
        family="llama",
        params_billions=70.0,
        num_layers=80,
        hidden_size=8192,
        num_heads=64,
        context_length=131_072,
        quant_options=[_INT4, _INT8, _F16],
        license="Llama 3.1",
        description="Large 70B model — needs high-end GPU or multi-GPU setup.",
        tags=["large", "instruct", "high-end"],
    ),

    "qwen2.5-14b": ModelSpec(
        name="Qwen2.5 14B Instruct",
        hf_repo="Qwen/Qwen2.5-14B-Instruct",
        family="qwen",
        params_billions=14.7,
        num_layers=48,
        hidden_size=5120,
        num_heads=40,
        context_length=131_072,
        quant_options=[_INT4, _INT8, _F16],
        license="Apache 2.0",
        description="Strong 14B model requiring ~24GB RAM (4-bit).",
        tags=["large", "instruct", "long-context"],
    ),

    "qwen2.5-32b": ModelSpec(
        name="Qwen2.5 32B Instruct",
        hf_repo="Qwen/Qwen2.5-32B-Instruct",
        family="qwen",
        params_billions=32.5,
        num_layers=64,
        hidden_size=5120,
        num_heads=40,
        context_length=131_072,
        quant_options=[_INT4, _INT8, _F16],
        license="Apache 2.0",
        description="Very capable 32B model — high VRAM or RAM needed.",
        tags=["large", "instruct", "long-context"],
    ),
}


def get_model(key: str) -> Optional[ModelSpec]:
    return CATALOG.get(key)


def list_models() -> List[ModelSpec]:
    return list(CATALOG.values())


def models_by_tag(tag: str) -> List[ModelSpec]:
    return [m for m in CATALOG.values() if tag in m.tags]


def recommend_models(tier: str) -> List[ModelSpec]:
    """Return models suitable for the given system tier."""
    tier_map = {
        "low":     ["tiny", "cpu-friendly"],
        "mid":     ["small", "mid"],
        "high":    ["mid", "large"],
        "extreme": ["large"],
    }
    tags = tier_map.get(tier, ["small"])
    seen = set()
    result = []
    for tag in tags:
        for m in models_by_tag(tag):
            if m.hf_repo not in seen:
                seen.add(m.hf_repo)
                result.append(m)
    return result
