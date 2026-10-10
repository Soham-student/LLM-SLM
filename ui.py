"""
ui.py
-----
Rich terminal UI helpers: system report, partition plan display,
warning panels, interactive prompts, and live generation output.
Windows CP1252 safe: all strings use ASCII characters only.
"""

from __future__ import annotations

import os
import sys

# Set UTF-8 mode env vars before Rich loads
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import time
from typing import List, Optional

from rich import box
from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.prompt import Prompt, Confirm
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from system_detector import SystemProfile, GPUInfo
from model_catalog import ModelSpec, CATALOG, recommend_models
from model_partitioner import PartitionPlan, Warning, Suggestion

# ─────────────────────────────────────────────
# Theme
# ─────────────────────────────────────────────

THEME = Theme({
    "header":       "bold cyan",
    "label":        "bold white",
    "value":        "bright_white",
    "good":         "bold green",
    "warn":         "bold yellow",
    "error":        "bold red",
    "info":         "bold blue",
    "dim_text":     "dim white",
    "gpu":          "bold magenta",
    "cpu":          "bold cyan",
    "disk_dev":     "bold red",
    "shard":        "bold bright_blue",
    "tier_low":     "bold red",
    "tier_mid":     "bold yellow",
    "tier_high":    "bold green",
    "tier_extreme": "bold bright_cyan",
})

# safe_box=True avoids Unicode box chars on legacy Windows consoles
console = Console(theme=THEME, highlight=False, safe_box=True)


# ─────────────────────────────────────────────
# Banner
# ─────────────────────────────────────────────

def print_banner():
    console.print()
    console.rule("[bold cyan]=== LLM -> SLM Partitioner ===[/bold cyan]", style="cyan")
    console.print("[bold bright_white]  Run large language models on ANY hardware, automatically.[/bold bright_white]")
    console.print("[dim_text]  Divides your LLM into SLM shards based on your system's CPU/RAM/GPU.[/dim_text]")
    console.rule(style="cyan")
    console.print()


# ─────────────────────────────────────────────
# System profile display
# ─────────────────────────────────────────────

def _tier_style(tier: str) -> str:
    return {
        "low":     "tier_low",
        "mid":     "tier_mid",
        "high":    "tier_high",
        "extreme": "tier_extreme",
    }.get(tier, "label")


def _tier_badge(tier: str) -> str:
    return {
        "low":     "[LOW]",
        "mid":     "[MID]",
        "high":    "[HIGH]",
        "extreme": "[MAX]",
    }.get(tier, "[?]")


def print_system_profile(profile: SystemProfile):
    console.rule("[header]System Hardware Profile[/header]", style="cyan")
    console.print()

    # ── Tier badge
    tier_style = _tier_style(profile.tier)
    tier_badge = _tier_badge(profile.tier)
    tier_panel = Panel(
        Align.center(
            f"[{tier_style}]{tier_badge}  {profile.tier.upper()} TIER[/{tier_style}]\n\n"
            f"[dim_text]{profile.tier_description}[/dim_text]"
        ),
        border_style=tier_style,
        padding=(1, 4),
    )
    console.print(tier_panel)
    console.print()

    # ── Hardware table
    table = Table(show_header=False, box=box.SIMPLE_HEAVY, border_style="cyan", padding=(0, 1))
    table.add_column("Key", style="label", width=24)
    table.add_column("Value", style="value")

    table.add_row("[CPU] CPU", profile.cpu_name)
    table.add_row("     Cores", f"{profile.cpu_physical_cores} physical / {profile.cpu_logical_cores} logical")
    table.add_row(
        "     Frequency",
        f"{profile.cpu_freq_mhz/1000:.2f} GHz" if profile.cpu_freq_mhz else "N/A"
    )
    table.add_row("     Architecture", profile.cpu_arch)

    ram_pct = profile.ram_available_mb / profile.ram_total_mb if profile.ram_total_mb else 0
    ram_style = "good" if ram_pct > 0.4 else ("warn" if ram_pct > 0.2 else "error")
    table.add_row("[RAM] Total RAM", f"{profile.ram_total_mb/1024:.1f} GB")
    table.add_row(
        "     Available",
        f"[{ram_style}]{profile.ram_available_mb/1024:.1f} GB[/{ram_style}]  ({ram_pct*100:.0f}% free)"
    )
    table.add_row(
        "     Usable (model)",
        f"{profile.usable_ram_mb()/1024:.1f} GB  [dim_text](after OS reserve)[/dim_text]"
    )

    if profile.gpus:
        for g in profile.gpus:
            vram_pct = g.vram_free_mb / g.vram_total_mb if g.vram_total_mb else 0
            vram_style = "good" if vram_pct > 0.4 else ("warn" if vram_pct > 0.2 else "error")
            table.add_row(f"[GPU] GPU [{g.index}]", f"[gpu]{g.name}[/gpu]")
            table.add_row("     VRAM Total", f"{g.vram_total_mb/1024:.1f} GB")
            table.add_row(
                "     VRAM Free",
                f"[{vram_style}]{g.vram_free_mb/1024:.1f} GB[/{vram_style}]  ({vram_pct*100:.0f}% free)"
            )
            if g.compute_capability:
                table.add_row(
                    "     Compute Cap.",
                    f"{g.compute_capability[0]}.{g.compute_capability[1]}"
                )
    else:
        table.add_row("[GPU] GPU", "[warn]None detected - CPU-only mode[/warn]")

    table.add_row("[DSK] Disk Free", f"{profile.disk_free_gb:.1f} GB / {profile.disk_total_gb:.1f} GB")
    table.add_row("[OS]  OS", profile.os_name)

    console.print(table)
    console.print()


# ─────────────────────────────────────────────
# Model list display
# ─────────────────────────────────────────────

def print_model_list(profile: Optional[SystemProfile] = None):
    console.rule("[header]Available Models[/header]", style="cyan")
    console.print()

    recommended_repos = set()
    if profile:
        recommended_repos = {m.hf_repo for m in recommend_models(profile.tier)}

    table = Table(
        show_header=True,
        header_style="bold cyan",
        box=box.SIMPLE_HEAVY,
        border_style="cyan",
        padding=(0, 1),
        expand=True,
    )
    table.add_column("#",         style="dim",        width=3)
    table.add_column("Key",       style="shard",      width=18)
    table.add_column("Model",     style="bold white", width=28)
    table.add_column("Params",    width=8,  justify="right")
    table.add_column("Layers",    width=7,  justify="right")
    table.add_column("VRAM(4b)",  width=12, justify="right")
    table.add_column("RAM(4b)",   width=11, justify="right")
    table.add_column("License",   width=11)
    table.add_column("Fit",       width=8)

    for i, (key, m) in enumerate(CATALOG.items(), 1):
        vram = m.vram_estimate_mb(4)
        ram  = m.ram_estimate_mb(4)
        is_rec = m.hf_repo in recommended_repos

        fit_icon = ""
        if profile:
            total_avail = profile.usable_vram_mb() + profile.usable_ram_mb()
            if vram <= profile.usable_vram_mb():
                fit_icon = "[good]OK-GPU[/good]"
            elif ram <= total_avail:
                fit_icon = "[warn]OK-CPU[/warn]"
            else:
                fit_icon = "[error]NO FIT[/error]"

        name_cell = ("(*) " if is_rec else "    ") + m.name

        table.add_row(
            str(i),
            key,
            name_cell,
            f"{m.params_billions:.1f}B",
            str(m.num_layers),
            f"{vram/1024:.1f} GB",
            f"{ram/1024:.1f} GB",
            m.license[:10],
            fit_icon,
        )

    console.print(table)
    if profile:
        console.print(
            "  [dim_text](*) = Recommended for your system  |  "
            "OK-GPU = fits in VRAM  |  OK-CPU = needs RAM  |  NO FIT = disk fallback[/dim_text]"
        )
    console.print()


# ─────────────────────────────────────────────
# Partition plan display
# ─────────────────────────────────────────────

STRATEGY_COLORS = {
    "gpu_only":      "good",
    "hybrid":        "warn",
    "cpu_only":      "warn",
    "disk_fallback": "error",
}

STRATEGY_LABELS = {
    "gpu_only":      "[FAST]  GPU-Only (Fastest)",
    "hybrid":        "[MIX]   GPU + CPU Hybrid",
    "cpu_only":      "[SLOW]  CPU-Only",
    "disk_fallback": "[VSLW]  Disk Offload (Very Slow)",
}


def print_partition_plan(plan: PartitionPlan):
    console.print()
    console.rule("[header]Partition Plan[/header]", style="cyan")
    console.print()

    strat_style = STRATEGY_COLORS.get(plan.strategy, "label")
    strat_label = STRATEGY_LABELS.get(plan.strategy, plan.strategy)

    # ── Summary panel
    summary_table = Table(show_header=False, box=None, padding=(0, 2))
    summary_table.add_column("k", style="label", width=22)
    summary_table.add_column("v", style="value")
    summary_table.add_row("Model",       plan.model_name)
    summary_table.add_row("Strategy",    f"[{strat_style}]{strat_label}[/{strat_style}]")
    summary_table.add_row("Quant",       f"{plan.quant_bits}-bit")
    summary_table.add_row("Layers",      f"{plan.total_layers} total")
    summary_table.add_row("  -> GPU",    f"[gpu]{plan.gpu_layer_count}[/gpu]")
    summary_table.add_row("  -> CPU",    f"[cpu]{plan.cpu_layer_count}[/cpu]")
    summary_table.add_row("GPU Memory",  f"{plan.gpu_memory_mb/1024:.2f} GB")
    summary_table.add_row("CPU Memory",  f"{plan.cpu_memory_mb/1024:.2f} GB")
    summary_table.add_row("Total Memory",f"{plan.total_memory_mb/1024:.2f} GB")
    summary_table.add_row("Est. Speed",  f"~{plan.estimated_tokens_per_sec} tok/s")

    console.print(Panel(summary_table, title="[header]Summary[/header]", border_style="cyan"))
    console.print()

    # ── Shard table
    shard_table = Table(
        show_header=True,
        header_style="bold cyan",
        box=box.SIMPLE_HEAVY,
        border_style="dim",
        padding=(0, 1),
    )
    shard_table.add_column("Shard",   width=7,  justify="center")
    shard_table.add_column("Layers",  width=14, justify="center")
    shard_table.add_column("Count",   width=7,  justify="right")
    shard_table.add_column("Device",  width=14)
    shard_table.add_column("Quant",   width=8,  justify="center")
    shard_table.add_column("Memory",  width=12, justify="right")
    shard_table.add_column("Usage %", width=16)

    total_layers = plan.total_layers or 1
    for s in plan.shards:
        if s.device.startswith("cuda"):
            dev_style = "gpu"
            dev_label = "[GPU]"
        elif s.device == "cpu":
            dev_style = "cpu"
            dev_label = "[CPU]"
        else:
            dev_style = "disk_dev"
            dev_label = "[DSK]"

        pct = s.num_layers / total_layers
        bar_fill = int(pct * 20)
        bar = "#" * bar_fill + "-" * (20 - bar_fill)

        shard_table.add_row(
            f"[shard]S-{s.shard_id}[/shard]",
            f"L{s.layer_start}-L{s.layer_end}",
            str(s.num_layers),
            f"[{dev_style}]{dev_label} {s.device}[/{dev_style}]",
            f"{s.quant_bits}-bit",
            f"{s.estimated_memory_mb/1024:.2f} GB",
            f"[{dev_style}][{bar}] {pct*100:.0f}%[/{dev_style}]",
        )

    console.print(shard_table)
    console.print()

    # ── Warnings
    if plan.warnings:
        console.rule("[warn]Warnings & Notices[/warn]", style="yellow")
        for w in plan.warnings:
            _print_warning(w)
        console.print()

    # ── Suggestions
    if plan.suggestions:
        console.rule("[info]Suggestions[/info]", style="blue")
        for s in sorted(plan.suggestions, key=lambda x: x.priority):
            _print_suggestion(s)
        console.print()


def _print_warning(w: Warning):
    icons = {"error": "[!!]", "warning": "[!] ", "info": "[i] "}
    styles = {"error": "error", "warning": "warn", "info": "info"}
    icon = icons.get(w.level, "[?]")
    style = styles.get(w.level, "label")
    console.print(f"  {icon} [{style}][{w.level.upper()}][/{style}]  {escape(w.message)}")


def _print_suggestion(s: Suggestion):
    console.print(f"  [>>] [info][Tip #{s.priority}][/info]  {escape(s.message)}")


# ─────────────────────────────────────────────
# Loading spinner
# ─────────────────────────────────────────────

def make_loading_progress() -> Progress:
    return Progress(
        SpinnerColumn("dots"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
        transient=True,
    )


# ─────────────────────────────────────────────
# Chat UI
# ─────────────────────────────────────────────

def print_chat_header(model_name: str, plan: PartitionPlan):
    console.print()
    console.rule(
        f"[header]Chat -- {model_name}  |  {plan.strategy}  |  {plan.quant_bits}-bit[/header]",
        style="cyan"
    )
    console.print(
        "  [dim_text]Type your message and press Enter.  "
        "Commands: [bold]/quit[/bold]  [bold]/info[/bold]  [bold]/plan[/bold]  [bold]/help[/bold][/dim_text]"
    )
    console.print()


def print_user_message(text: str):
    console.print(Panel(
        escape(text),
        title="[bold white]You[/bold white]",
        border_style="white",
        padding=(0, 1),
    ))


def print_assistant_header(model_name: str):
    console.print(f"[bold cyan][AI] {model_name}:[/bold cyan]")


def print_memory_usage(usage: dict):
    lines = [f"  {k}: {v:.0f} MB" for k, v in usage.items()]
    console.print(Panel(
        "\n".join(lines),
        title="[header]Memory Usage[/header]",
        border_style="dim"
    ))


def print_token_optimization_card(result):
    """
    Renders an ASCII-safe card showing prompt token savings & KV budget.
    """
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column(style="value")
    table.add_column(style="bold cyan")
    table.add_column(style="value")

    saved_pct = result.compression_ratio * 100
    save_color = "good" if result.tokens_saved > 0 else "dim_text"

    table.add_row(
        "Prompt Tokens:",
        f"{result.estimated_original_tokens} -> [bold white]{result.estimated_optimized_tokens}[/bold white]",
        "Saved:",
        f"[{save_color}]{result.tokens_saved} tok ({saved_pct:.1f}%)[/{save_color}]",
    )
    table.add_row(
        "KV Cache Budget:",
        f"{result.kv_cache_budget_tokens} tokens",
        "Recommended Max:",
        f"{result.recommended_max_new_tokens} tokens",
    )

    console.print(Panel(
        table,
        title="[header][OPT] Token Optimizer[/header]",
        border_style="cyan",
        padding=(0, 1)
    ))


class LiveTokenBar:
    """
    Renders live streaming response text while maintaining an animated
    real-time token count progress bar and performance metrics.
    CP1252-safe: pure ASCII indicators.
    """
    def __init__(self, model_name: str, max_tokens: int = 512, prompt_tokens: int = 0):
        self.model_name = model_name
        self.max_tokens = max(1, max_tokens)
        self.prompt_tokens = prompt_tokens
        self.generated_text = ""
        self.token_count = 0
        self.start_time = time.time()
        self.live = None

    def start(self):
        self.start_time = time.time()
        self.live = Live(self._renderable(), console=console, refresh_per_second=10)
        self.live.start()
        return self

    def update(self, chunk: str, token_count: Optional[int] = None):
        self.generated_text += chunk
        if token_count is not None:
            self.token_count = token_count
        else:
            self.token_count += max(1, len(chunk.split()))
        if self.live:
            self.live.update(self._renderable())

    def finish(self, stop_reason: str = ""):
        if self.live:
            self.live.update(self._renderable(final=True, stop_reason=stop_reason))
            self.live.stop()
            self.live = None

    def _renderable(self, final: bool = False, stop_reason: str = ""):
        elapsed = max(0.01, time.time() - self.start_time)
        tps = self.token_count / elapsed
        pct = min(1.0, self.token_count / self.max_tokens)

        # Build ASCII progress bar
        fill = int(pct * 25)
        bar = "=" * fill + ("-" * (25 - fill))

        header_text = escape(self.generated_text) if self.generated_text else "[dim]Generating initial tokens...[/dim]"

        if final:
            reason_str = f" ({stop_reason})" if stop_reason else ""
            status_line = (
                f"[good][DONE{reason_str}][/good] [bold white]{self.token_count}/{self.max_tokens} tokens[/bold white] "
                f"[{pct*100:.0f}%]  |  Speed: [bold yellow]{tps:.1f} tok/s[/bold yellow]  |  "
                f"Time: {elapsed:.1f}s"
            )
        else:
            status_line = (
                f"[bold cyan][GENERATING][/bold cyan] [bold white]{self.token_count}/{self.max_tokens} tokens[/bold white] "
                f"[[green]{bar}[/green]] {pct*100:.0f}%  |  "
                f"Speed: [bold yellow]{tps:.1f} tok/s[/bold yellow]  |  Time: {elapsed:.1f}s"
            )

        content = f"{header_text}\n\n[dim]--- Token Optimizer & Live Status ---[/dim]\n{status_line}"
        border_col = "green" if final else "cyan"
        title = f"[bold cyan][AI] {self.model_name}[/bold cyan]"
        return Panel(content, title=title, border_style=border_col, padding=(0, 1))


# ─────────────────────────────────────────────
# Interactive selectors
# ─────────────────────────────────────────────

def select_model(profile: Optional[SystemProfile] = None) -> Optional[str]:
    """Prompt user to choose a model key from the catalog."""
    keys = list(CATALOG.keys())
    print_model_list(profile)

    console.print("[label]Enter model number or key[/label] (or [bold]q[/bold] to quit):")
    choice = Prompt.ask("  Model", default="")

    if choice.lower() in ("q", "quit", "exit", ""):
        return None

    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(keys):
            return keys[idx]
        console.print("[error]Invalid number.[/error]")
        return None

    if choice in CATALOG:
        return choice

    matches = [k for k in keys if choice.lower() in k.lower()]
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        console.print(f"[warn]Multiple matches: {matches}. Be more specific.[/warn]")
    else:
        console.print(f"[error]Model '{choice}' not found.[/error]")
    return None


def ask_confirm(question: str, default: bool = True) -> bool:
    return Confirm.ask(f"  [label]{question}[/label]", default=default, console=console)


def ask_text(prompt: str, default: str = "") -> str:
    return Prompt.ask(f"  [label]{prompt}[/label]", default=default, console=console)
