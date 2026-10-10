"""
model_partitioner.py
--------------------
Core intelligence: given a SystemProfile and a ModelSpec, compute
an optimal partition plan (SLM shards) that fits within hardware limits.

Partitioning strategies:
  1. GPU-only        — all layers in VRAM (best performance)
  2. GPU+CPU hybrid  — some layers GPU, remainder CPU (good)
  3. CPU-only        — all layers in RAM (slow but works)
  4. Sequential disk — layers loaded from disk in chunks (very slow, fallback)

Each "shard" is a contiguous range of transformer layers with an assigned
device and quantization level.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from system_detector import SystemProfile
from model_catalog import ModelSpec


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────

@dataclass
class Shard:
    """A contiguous slice of transformer layers assigned to a device."""
    shard_id: int
    layer_start: int          # inclusive
    layer_end: int            # inclusive
    device: str               # "cuda:0", "cpu", "disk"
    quant_bits: int           # 4, 8, 16, 32
    estimated_memory_mb: float

    @property
    def num_layers(self) -> int:
        return self.layer_end - self.layer_start + 1

    @property
    def label(self) -> str:
        return f"Shard-{self.shard_id} [L{self.layer_start}-L{self.layer_end}] -> {self.device} ({self.quant_bits}-bit)"


@dataclass
class Warning:
    level: str    # "info" | "warning" | "error"
    message: str


@dataclass
class Suggestion:
    priority: int   # 1 = highest
    message: str


@dataclass
class PartitionPlan:
    model_name: str
    total_layers: int
    shards: List[Shard] = field(default_factory=list)
    strategy: str = ""              # "gpu_only" | "hybrid" | "cpu_only" | "disk_fallback"
    quant_bits: int = 4
    total_memory_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    cpu_memory_mb: float = 0.0
    warnings: List[Warning] = field(default_factory=list)
    suggestions: List[Suggestion] = field(default_factory=list)
    is_feasible: bool = True
    estimated_tokens_per_sec: float = 0.0

    # HuggingFace load_in_* flags
    load_in_4bit: bool = False
    load_in_8bit: bool = False

    # device_map for HF accelerate
    device_map: dict = field(default_factory=dict)

    def add_warning(self, level: str, message: str):
        self.warnings.append(Warning(level, message))

    def add_suggestion(self, priority: int, message: str):
        self.suggestions.append(Suggestion(priority, message))

    @property
    def gpu_layer_count(self) -> int:
        return sum(s.num_layers for s in self.shards if s.device.startswith("cuda"))

    @property
    def cpu_layer_count(self) -> int:
        return sum(s.num_layers for s in self.shards if s.device == "cpu")


# ─────────────────────────────────────────────
# Quantization selector
# ─────────────────────────────────────────────

def _choose_quant(
    model: ModelSpec,
    usable_memory_mb: float,
    prefer_quality: bool = False,
) -> int:
    """
    Choose the highest quality quantization that fits within usable_memory_mb.
    Returns bits: 16, 8, or 4.
    """
    for bits in ([16, 8, 4] if prefer_quality else [4, 8, 16]):
        needed = model.vram_estimate_mb(bits)
        if needed <= usable_memory_mb:
            return bits
    return 4  # worst-case fallback


# ─────────────────────────────────────────────
# Layer splitter helpers
# ─────────────────────────────────────────────

def _layers_fitting_in(memory_mb: float, bytes_per_layer_mb: float) -> int:
    """How many layers can we fit in `memory_mb` given per-layer cost?"""
    if bytes_per_layer_mb <= 0:
        return 0
    return max(0, int(memory_mb / bytes_per_layer_mb))


def _build_device_map(shards: List[Shard], model: ModelSpec) -> dict:
    """
    Build a HuggingFace accelerate device_map dict.
    Maps layer index → device string.
    """
    dmap: dict = {}
    # embedding and lm_head
    if shards:
        first_device = shards[0].device
        last_device = shards[-1].device
    else:
        first_device = last_device = "cpu"

    dmap["model.embed_tokens"] = first_device
    dmap["model.norm"] = last_device
    dmap["lm_head"] = last_device

    for shard in shards:
        for layer_idx in range(shard.layer_start, shard.layer_end + 1):
            dmap[f"model.layers.{layer_idx}"] = shard.device

    return dmap


# ─────────────────────────────────────────────
# Token speed estimator
# ─────────────────────────────────────────────

def _estimate_tokens_per_sec(
    plan: PartitionPlan,
    profile: SystemProfile,
) -> float:
    """
    Rough heuristic for expected generation speed (tokens/sec).
    Based on tier + strategy + quant level.
    """
    base_speed = {
        "gpu_only":      {"extreme": 80, "high": 45, "mid": 20, "low": 5},
        "hybrid":        {"extreme": 40, "high": 20, "mid": 10, "low": 2},
        "cpu_only":      {"extreme": 15, "high": 8,  "mid": 4,  "low": 1},
        "disk_fallback": {"extreme": 2,  "high": 1,  "mid": 0.5,"low": 0.2},
    }
    quant_multiplier = {4: 1.0, 8: 0.6, 16: 0.35, 32: 0.18}

    speed_table = base_speed.get(plan.strategy, base_speed["cpu_only"])
    base = speed_table.get(profile.tier, 1.0)
    mult = quant_multiplier.get(plan.quant_bits, 1.0)
    return round(base * mult, 1)


# ─────────────────────────────────────────────
# Main partitioner
# ─────────────────────────────────────────────

def partition_model(
    model: ModelSpec,
    profile: SystemProfile,
    prefer_quality: bool = False,
    force_cpu: bool = False,
) -> PartitionPlan:
    """
    Compute an optimal partition plan for running `model` on `profile`.

    Parameters
    ----------
    model          : ModelSpec from the catalog
    profile        : SystemProfile from system_detector
    prefer_quality : Prefer higher quant quality over speed
    force_cpu      : Skip GPU entirely (e.g., driver issues)
    """

    plan = PartitionPlan(
        model_name=model.name,
        total_layers=model.num_layers,
    )

    total_layers = model.num_layers
    usable_vram = 0.0 if force_cpu else profile.usable_vram_mb()
    usable_ram  = profile.usable_ram_mb()

    # ── 1. Choose quantization ────────────────────────────────────────────
    # We want the best quant that fits total memory (VRAM + RAM)
    total_available = usable_vram + usable_ram
    quant_bits = _choose_quant(model, total_available, prefer_quality)
    plan.quant_bits = quant_bits
    plan.load_in_4bit = (quant_bits == 4)
    plan.load_in_8bit = (quant_bits == 8)

    bytes_per_layer = model.bytes_per_layer_mb(quant_bits)
    total_model_mb  = bytes_per_layer * total_layers

    # ── 2. Feasibility check ──────────────────────────────────────────────
    if total_model_mb > total_available * 1.05:
        # Try 4-bit as last resort
        if quant_bits != 4:
            quant_bits = 4
            plan.quant_bits = 4
            plan.load_in_4bit = True
            plan.load_in_8bit = False
            bytes_per_layer = model.bytes_per_layer_mb(4)
            total_model_mb  = bytes_per_layer * total_layers

        if total_model_mb > total_available * 1.05:
            plan.is_feasible = False
            plan.strategy = "disk_fallback"
            plan.add_warning(
                "error",
                f"Model '{model.name}' requires ~{total_model_mb/1024:.1f} GB but only "
                f"~{total_available/1024:.1f} GB total memory is available (VRAM + RAM). "
                "Performance will be extremely slow using disk offloading."
            )
            plan.add_suggestion(1,
                f"Switch to a smaller model. Your system ({profile.tier} tier) "
                f"works best with models up to {_max_params_for_tier(profile.tier):.0f}B parameters."
            )
            # Still build a disk-based plan
            _build_disk_plan(plan, model, bytes_per_layer, usable_vram, usable_ram)
            plan.estimated_tokens_per_sec = _estimate_tokens_per_sec(plan, profile)
            _add_common_warnings(plan, model, profile)
            _finalize_device_map(plan, model)
            return plan

    # ── 3. GPU-only strategy ──────────────────────────────────────────────
    gpu_layer_capacity = _layers_fitting_in(usable_vram, bytes_per_layer)

    if not force_cpu and gpu_layer_capacity >= total_layers:
        # All layers fit in GPU
        plan.strategy = "gpu_only"
        shard = Shard(
            shard_id=0,
            layer_start=0,
            layer_end=total_layers - 1,
            device="cuda:0" if profile.has_cuda else ("mps" if profile.has_mps else "cpu"),
            quant_bits=quant_bits,
            estimated_memory_mb=total_model_mb,
        )
        plan.shards.append(shard)
        plan.gpu_memory_mb = total_model_mb

        if profile.has_cuda and len(profile.gpus) > 1:
            plan.add_suggestion(2,
                f"You have {len(profile.gpus)} GPUs. Consider enabling tensor parallelism "
                "for even faster inference."
            )

    # ── 4. Hybrid strategy (GPU + CPU) ────────────────────────────────────
    elif not force_cpu and gpu_layer_capacity > 0:
        plan.strategy = "hybrid"
        plan.add_warning("warning",
            f"Model too large for GPU alone. Splitting: "
            f"{gpu_layer_capacity} layers -> GPU, "
            f"{total_layers - gpu_layer_capacity} layers -> CPU (RAM)."
        )

        # GPU shard
        gpu_mem = gpu_layer_capacity * bytes_per_layer
        plan.shards.append(Shard(
            shard_id=0,
            layer_start=0,
            layer_end=gpu_layer_capacity - 1,
            device="cuda:0",
            quant_bits=quant_bits,
            estimated_memory_mb=gpu_mem,
        ))
        plan.gpu_memory_mb = gpu_mem

        # CPU shard(s) — split across logical RAM chunks if very large
        cpu_layers_remaining = total_layers - gpu_layer_capacity
        cpu_mem_needed = cpu_layers_remaining * bytes_per_layer

        if cpu_mem_needed <= usable_ram:
            plan.shards.append(Shard(
                shard_id=1,
                layer_start=gpu_layer_capacity,
                layer_end=total_layers - 1,
                device="cpu",
                quant_bits=quant_bits,
                estimated_memory_mb=cpu_mem_needed,
            ))
            plan.cpu_memory_mb = cpu_mem_needed
        else:
            # Even CPU doesn't fit entirely — split CPU shards and warn
            cpu_layer_capacity = _layers_fitting_in(usable_ram, bytes_per_layer)
            _build_chunked_cpu_shards(
                plan, gpu_layer_capacity, total_layers - 1,
                cpu_layer_capacity, bytes_per_layer
            )

    # ── 5. CPU-only strategy ──────────────────────────────────────────────
    else:
        cpu_layer_capacity = _layers_fitting_in(usable_ram, bytes_per_layer)

        if cpu_layer_capacity >= total_layers:
            plan.strategy = "cpu_only"
            plan.add_warning("info",
                "No GPU detected or GPU disabled. Running entirely on CPU — expect slower inference."
            )
            shard = Shard(
                shard_id=0,
                layer_start=0,
                layer_end=total_layers - 1,
                device="cpu",
                quant_bits=quant_bits,
                estimated_memory_mb=total_model_mb,
            )
            plan.shards.append(shard)
            plan.cpu_memory_mb = total_model_mb
        else:
            # CPU can't hold the full model either — chunked CPU
            plan.strategy = "disk_fallback"
            _build_disk_plan(plan, model, bytes_per_layer, 0, usable_ram)

    # ── 6. Warnings & suggestions ─────────────────────────────────────────
    _add_common_warnings(plan, model, profile)

    # Recompute totals
    plan.total_memory_mb = sum(s.estimated_memory_mb for s in plan.shards)
    plan.estimated_tokens_per_sec = _estimate_tokens_per_sec(plan, profile)

    # Build HF device map
    _finalize_device_map(plan, model)

    return plan


# ─────────────────────────────────────────────
# Helper builders
# ─────────────────────────────────────────────

def _build_chunked_cpu_shards(
    plan: PartitionPlan,
    layer_start: int,
    layer_end: int,
    cpu_layer_capacity: int,
    bytes_per_layer: float,
):
    """Build multiple CPU shards when CPU RAM is also limited."""
    remaining_start = layer_start
    shard_id = len(plan.shards)
    while remaining_start <= layer_end:
        end = min(remaining_start + cpu_layer_capacity - 1, layer_end)
        mem = (end - remaining_start + 1) * bytes_per_layer
        plan.shards.append(Shard(
            shard_id=shard_id,
            layer_start=remaining_start,
            layer_end=end,
            device="cpu",
            quant_bits=plan.quant_bits,
            estimated_memory_mb=mem,
        ))
        plan.cpu_memory_mb += mem
        remaining_start = end + 1
        shard_id += 1

    plan.add_warning("warning",
        "CPU RAM is also limited. Some layer shards will be swapped in/out. "
        "Performance will be significantly reduced."
    )


def _build_disk_plan(
    plan: PartitionPlan,
    model: ModelSpec,
    bytes_per_layer: float,
    usable_vram: float,
    usable_ram: float,
):
    """Build a disk-offload plan when neither VRAM nor RAM can hold the full model."""
    shard_id = 0
    layer_idx = 0
    total_layers = model.num_layers

    # Put as many as possible in GPU
    if usable_vram > 0:
        gpu_cap = _layers_fitting_in(usable_vram, bytes_per_layer)
        if gpu_cap > 0:
            end = min(gpu_cap - 1, total_layers - 1)
            plan.shards.append(Shard(
                shard_id=shard_id,
                layer_start=0,
                layer_end=end,
                device="cuda:0",
                quant_bits=plan.quant_bits,
                estimated_memory_mb=gpu_cap * bytes_per_layer,
            ))
            plan.gpu_memory_mb += gpu_cap * bytes_per_layer
            layer_idx = end + 1
            shard_id += 1

    # Then RAM
    if layer_idx < total_layers and usable_ram > 0:
        cpu_cap = _layers_fitting_in(usable_ram, bytes_per_layer)
        if cpu_cap > 0:
            end = min(layer_idx + cpu_cap - 1, total_layers - 1)
            plan.shards.append(Shard(
                shard_id=shard_id,
                layer_start=layer_idx,
                layer_end=end,
                device="cpu",
                quant_bits=plan.quant_bits,
                estimated_memory_mb=cpu_cap * bytes_per_layer,
            ))
            plan.cpu_memory_mb += cpu_cap * bytes_per_layer
            layer_idx = end + 1
            shard_id += 1

    # Remaining → disk
    if layer_idx < total_layers:
        end = total_layers - 1
        mem = (end - layer_idx + 1) * bytes_per_layer
        plan.shards.append(Shard(
            shard_id=shard_id,
            layer_start=layer_idx,
            layer_end=end,
            device="disk",
            quant_bits=plan.quant_bits,
            estimated_memory_mb=mem,
        ))
        shard_id += 1


def _add_common_warnings(plan: PartitionPlan, model: ModelSpec, profile: SystemProfile):
    """Add universal warnings and suggestions based on the plan and profile."""
    ram_gb = profile.ram_total_mb / 1024
    vram_gb = profile.total_vram_mb / 1024

    # RAM headroom
    ram_after_model = profile.ram_available_mb - plan.cpu_memory_mb
    if ram_after_model < 1024:
        plan.add_warning("warning",
            f"Only ~{ram_after_model/1024:.1f} GB RAM will remain after loading. "
            "Your system may become unresponsive. Close other applications first."
        )

    # Low VRAM
    if profile.has_cuda and vram_gb < 4:
        plan.add_warning("warning",
            f"Your GPU has only {vram_gb:.1f} GB VRAM. GPU layers will be limited. "
            "Consider upgrading or using a smaller model."
        )

    # CPU-only slow warning
    if plan.strategy in ("cpu_only", "disk_fallback"):
        plan.add_warning("info",
            f"CPU inference speed will be roughly {plan.estimated_tokens_per_sec} tok/s. "
            "For better performance, a discrete GPU is strongly recommended."
        )

    # Disk fallback
    if plan.strategy == "disk_fallback":
        plan.add_warning("error",
            "Disk offloading is extremely slow (often < 1 tok/s). "
            "This mode is not recommended for interactive use."
        )
        plan.add_suggestion(1,
            "Use a quantized (4-bit) version of a smaller model to avoid disk offloading."
        )

    # Hybrid performance note
    if plan.strategy == "hybrid":
        plan.add_suggestion(3,
            "Hybrid GPU/CPU mode has a data-transfer bottleneck. "
            "Increasing VRAM (e.g., upgrading GPU) would improve performance significantly."
        )

    # Quantization note
    if plan.quant_bits == 4:
        plan.add_suggestion(4,
            "4-bit quantization is active. Quality is close to full precision for most tasks. "
            "For coding or math, consider 8-bit if you have sufficient memory."
        )

    # Disk space check
    model_disk_gb = model.vram_estimate_mb(plan.quant_bits) / 1024
    if profile.disk_free_gb < model_disk_gb * 1.2:
        plan.add_warning("warning",
            f"Low disk space: {profile.disk_free_gb:.1f} GB free but model needs ~{model_disk_gb:.1f} GB. "
            "Ensure enough space for model download and cache."
        )

    # Context length warning
    if model.context_length > 32_000 and profile.ram_total_mb < 32_768:
        plan.add_warning("info",
            f"Model supports {model.context_length:,} context tokens, but long contexts "
            "require much more memory. Start with shorter prompts if you hit OOM errors."
        )


def _finalize_device_map(plan: PartitionPlan, model: ModelSpec):
    """Populate plan.device_map using the computed shards."""
    plan.device_map = _build_device_map(plan.shards, model)


def _max_params_for_tier(tier: str) -> float:
    return {"low": 3.0, "mid": 7.0, "high": 34.0, "extreme": 70.0}.get(tier, 7.0)


# ─────────────────────────────────────────────
# Pretty-print helper (for debugging)
# ─────────────────────────────────────────────

def summarize_plan(plan: PartitionPlan) -> str:
    lines = [
        f"Model   : {plan.model_name}",
        f"Strategy: {plan.strategy}  |  Quant: {plan.quant_bits}-bit",
        f"Layers  : {plan.total_layers} total -> {plan.gpu_layer_count} GPU / {plan.cpu_layer_count} CPU",
        f"Memory  : GPU {plan.gpu_memory_mb/1024:.2f} GB + CPU {plan.cpu_memory_mb/1024:.2f} GB",
        f"Speed   : ~{plan.estimated_tokens_per_sec} tok/s (estimated)",
        "",
        "Shards:",
    ]
    for s in plan.shards:
        lines.append(f"  {s.label}  (~{s.estimated_memory_mb/1024:.2f} GB)")
    if plan.warnings:
        lines.append("\nWarnings:")
        for w in plan.warnings:
            lines.append(f"  [{w.level.upper()}] {w.message}")
    if plan.suggestions:
        lines.append("\nSuggestions:")
        for s in sorted(plan.suggestions, key=lambda x: x.priority):
            lines.append(f"  [{s.priority}] {s.message}")
    return "\n".join(lines)
