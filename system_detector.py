"""
system_detector.py
------------------
Detects hardware capabilities: CPU, RAM, GPU/VRAM, Disk.
Returns a SystemProfile used by the model partitioner to decide
how to split an LLM into SLM shards.
"""

import os
import platform
import subprocess
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import psutil

# Optional GPU support via torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────

@dataclass
class GPUInfo:
    index: int
    name: str
    vram_total_mb: float
    vram_free_mb: float
    vram_used_mb: float
    compute_capability: Optional[Tuple[int, int]] = None  # (major, minor)
    is_cuda: bool = True


@dataclass
class SystemProfile:
    # CPU
    cpu_name: str
    cpu_physical_cores: int
    cpu_logical_cores: int
    cpu_freq_mhz: float                  # base / max freq in MHz
    cpu_arch: str                        # x86_64, arm64, etc.

    # RAM
    ram_total_mb: float
    ram_available_mb: float
    ram_used_mb: float

    # GPU(s)
    gpus: List[GPUInfo] = field(default_factory=list)
    has_cuda: bool = False
    has_mps: bool = False                # Apple Metal (not relevant on Windows but included)

    # Disk (where model cache lives)
    disk_free_gb: float = 0.0
    disk_total_gb: float = 0.0

    # OS
    os_name: str = ""
    os_version: str = ""
    python_bits: int = 64

    # Derived helper fields (filled after detection)
    total_vram_mb: float = 0.0
    free_vram_mb: float = 0.0
    gpu_count: int = 0

    # Tier computed after detection
    tier: str = "unknown"               # "low" | "mid" | "high" | "extreme"
    tier_description: str = ""

    def usable_ram_mb(self) -> float:
        """RAM that is safely usable (leave 10% or 1 GB headroom for OS, whichever is larger)."""
        headroom = max(1024, self.ram_available_mb * 0.10)
        return max(0, self.ram_available_mb - headroom)

    def usable_vram_mb(self) -> float:
        """VRAM that is safely usable (leave 512 MB headroom per GPU)."""
        headroom = 512 * max(1, self.gpu_count)
        return max(0, self.free_vram_mb - headroom)


# ─────────────────────────────────────────────
# Detection helpers
# ─────────────────────────────────────────────

def _get_cpu_name() -> str:
    """Return a human-readable CPU name."""
    system = platform.system()
    if system == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            winreg.CloseKey(key)
            if name and name.strip():
                return name.strip()
        except Exception:
            pass
        return platform.processor() or "Unknown CPU"
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        except Exception:
            pass
    elif system == "Darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            )
            return out.strip()
        except Exception:
            pass
    return platform.processor() or "Unknown CPU"


def _get_cpu_freq_mhz() -> float:
    """Return max CPU frequency in MHz, falling back to current."""
    freq = psutil.cpu_freq()
    if freq is None:
        return 0.0
    return freq.max if freq.max > 0 else freq.current


def _detect_gpus() -> Tuple[List[GPUInfo], bool, bool, float, float]:
    """
    Returns (gpu_list, has_cuda, has_mps, total_vram_mb, free_vram_mb).
    """
    gpus: List[GPUInfo] = []
    has_cuda = False
    has_mps = False
    total_vram = 0.0
    free_vram = 0.0

    if TORCH_AVAILABLE:
        if torch.cuda.is_available():
            has_cuda = True
            n = torch.cuda.device_count()
            for i in range(n):
                props = torch.cuda.get_device_properties(i)
                vram_total = props.total_memory / (1024 ** 2)
                # query free memory
                torch.cuda.set_device(i)
                reserved = torch.cuda.memory_reserved(i) / (1024 ** 2)
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 2)
                vram_free = vram_total - reserved
                vram_used = allocated

                cc = (props.major, props.minor)
                gpu = GPUInfo(
                    index=i,
                    name=props.name,
                    vram_total_mb=vram_total,
                    vram_free_mb=vram_free,
                    vram_used_mb=vram_used,
                    compute_capability=cc,
                    is_cuda=True,
                )
                gpus.append(gpu)
                total_vram += vram_total
                free_vram += vram_free

        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            has_mps = True
            # MPS shares system RAM; estimate
            mem = psutil.virtual_memory()
            estimated_vram = mem.total / (1024 ** 2) * 0.6
            gpu = GPUInfo(
                index=0,
                name="Apple Silicon MPS",
                vram_total_mb=estimated_vram,
                vram_free_mb=estimated_vram * 0.7,
                vram_used_mb=estimated_vram * 0.3,
                is_cuda=False,
            )
            gpus.append(gpu)
            total_vram = estimated_vram
            free_vram = estimated_vram * 0.7

    return gpus, has_cuda, has_mps, total_vram, free_vram


def _compute_tier(profile: SystemProfile) -> Tuple[str, str]:
    """
    Classify the system into a performance tier.
    Tiers:
      extreme  ≥ 24 GB VRAM  or ≥ 64 GB RAM
      high     ≥ 8 GB VRAM   or ≥ 32 GB RAM
      mid      ≥ 4 GB VRAM   or ≥ 16 GB RAM
      low      < 4 GB VRAM   and < 16 GB RAM
    """
    vram = profile.total_vram_mb / 1024   # GB
    ram  = profile.ram_total_mb  / 1024   # GB

    if vram >= 24 or ram >= 64:
        return "extreme", (
            f"Excellent hardware ({ram:.0f} GB RAM, {vram:.0f} GB VRAM). "
            "Can run 70B+ models with full precision."
        )
    elif vram >= 8 or ram >= 32:
        return "high", (
            f"High-end hardware ({ram:.0f} GB RAM, {vram:.0f} GB VRAM). "
            "Can run 13B–34B models comfortably."
        )
    elif vram >= 4 or ram >= 16:
        return "mid", (
            f"Mid-range hardware ({ram:.0f} GB RAM, {vram:.0f} GB VRAM). "
            "Best suited for 7B models with 4-bit quantization."
        )
    else:
        return "low", (
            f"Low-end hardware ({ram:.0f} GB RAM, {vram:.0f} GB VRAM). "
            "Recommended to use 1B–3B quantized models only."
        )


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def detect_system() -> SystemProfile:
    """Run full hardware detection and return a SystemProfile."""
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(os.path.expanduser("~"))

    gpus, has_cuda, has_mps, total_vram, free_vram = _detect_gpus()

    profile = SystemProfile(
        cpu_name=_get_cpu_name(),
        cpu_physical_cores=psutil.cpu_count(logical=False) or 1,
        cpu_logical_cores=psutil.cpu_count(logical=True) or 1,
        cpu_freq_mhz=_get_cpu_freq_mhz(),
        cpu_arch=platform.machine(),
        ram_total_mb=mem.total / (1024 ** 2),
        ram_available_mb=mem.available / (1024 ** 2),
        ram_used_mb=mem.used / (1024 ** 2),
        gpus=gpus,
        has_cuda=has_cuda,
        has_mps=has_mps,
        disk_free_gb=disk.free / (1024 ** 3),
        disk_total_gb=disk.total / (1024 ** 3),
        os_name=platform.system(),
        os_version=platform.version(),
        python_bits=64 if platform.architecture()[0] == "64bit" else 32,
        total_vram_mb=total_vram,
        free_vram_mb=free_vram,
        gpu_count=len(gpus),
    )

    tier, desc = _compute_tier(profile)
    profile.tier = tier
    profile.tier_description = desc
    return profile


if __name__ == "__main__":
    p = detect_system()
    print(f"Tier: {p.tier}")
    print(f"CPU: {p.cpu_name} ({p.cpu_physical_cores}p / {p.cpu_logical_cores}L cores)")
    print(f"RAM: {p.ram_total_mb/1024:.1f} GB total, {p.ram_available_mb/1024:.1f} GB free")
    for g in p.gpus:
        print(f"GPU[{g.index}]: {g.name} | VRAM {g.vram_total_mb/1024:.1f} GB total, {g.vram_free_mb/1024:.1f} GB free")
    print(f"Disk Free: {p.disk_free_gb:.1f} GB")
