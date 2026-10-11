"""
main.py
-------
Entry point for the LLM -> SLM Partitioner application.

Architecture: All tasks are routed through the Orchestrator, which
delegates each step to a specialized micro-SLM worker. No single worker
needs more than ~2 GB RAM — the system adapts to any hardware tier.

Usage:
    python main.py                   # Interactive wizard
    python main.py --scan            # Just scan hardware
    python main.py --list            # List models
    python main.py --model qwen2.5-3b --plan-only
    python main.py --model llama-3.1-8b --force-cpu
    python main.py --workers         # Show all workers and RAM budgets
"""

from __future__ import annotations

# ── Windows UTF-8 fix (must be before any rich/print imports) ─────────────────
import sys, os, io
if sys.platform == "win32":
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.environ["PYTHONIOENCODING"] = "utf-8"

import argparse
import time
import traceback
from typing import Optional, List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# ── Local modules ─────────────────────────────────────────────────────────────
from orchestrator import Orchestrator
from model_catalog import CATALOG
import ui

console = ui.console


# ─────────────────────────────────────────────
# CLI argument parser
# ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="slm-runner",
        description=(
            "Run large LLMs on ANY local hardware by auto-partitioning them "
            "into specialized micro-SLM workers."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                         Interactive wizard
  python main.py --scan                  Show hardware report only
  python main.py --list                  List available models
  python main.py --workers               Show micro-worker breakdown
  python main.py --model qwen2.5-3b     Load and chat with a model
  python main.py --model mistral-7b --plan-only
  python main.py --model llama-3.1-8b --force-cpu
  python main.py --model phi-3.5-mini --quality
        """,
    )
    p.add_argument("--scan",        action="store_true", help="Scan hardware and exit")
    p.add_argument("--list",        action="store_true", help="List available models and exit")
    p.add_argument("--workers",     action="store_true", help="Show micro-worker breakdown and exit")
    p.add_argument("--model",       type=str,            help="Model key (from catalog)")
    p.add_argument("--plan-only",   action="store_true", help="Show partition plan without loading")
    p.add_argument("--force-cpu",   action="store_true", help="Disable GPU and use CPU only")
    p.add_argument("--quality",     action="store_true", help="Prefer higher quality quantization")
    p.add_argument("--max-tokens",  type=int, default=512, help="Max new tokens per response")
    p.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    p.add_argument("--system",      type=str, default=None, help="System prompt text")
    p.add_argument("--no-stream",   action="store_true", help="Disable token streaming")
    p.add_argument("--no-optimize", action="store_true", help="Disable prompt token optimization")
    p.add_argument("--debug",       action="store_true", help="Show full tracebacks on errors")
    return p


# ─────────────────────────────────────────────
# Worker summary display
# ─────────────────────────────────────────────

def print_worker_summary(orch: Orchestrator):
    console.rule("[header]Micro-SLM Worker Breakdown[/header]", style="cyan")
    console.print()

    table = Table(
        show_header=True,
        header_style="bold cyan",
        box=box.SIMPLE_HEAVY,
        border_style="cyan",
        padding=(0, 1),
    )
    table.add_column("#",         style="dim",        width=3)
    table.add_column("Worker ID", style="bold white", width=18)
    table.add_column("RAM Budget", width=12, justify="right")
    table.add_column("Loads Model?", width=13, justify="center")
    table.add_column("Responsibility", width=55)

    descriptions = {
        "hardware":     "Detects CPU / RAM / GPU / Disk, computes tier",
        "catalog":      "Resolves model keys, lists & recommends models",
        "quant":        "Chooses optimal quantization (4/8/16-bit)",
        "plan":         "Splits model layers into GPU/CPU/Disk shards",
        "kv_budget":    "Calculates safe KV-cache token budget",
        "token":        "Compresses prompts to reduce token count",
        "repetition":   "Detects n-gram loops to stop runaway generation",
        "template":     "Formats prompts with model-family chat templates",
        "pruner":       "Prunes multi-turn history to fit context window",
        "download":     "Checks / pre-downloads model from HuggingFace Hub",
        "shard_loader": "Loads model weights per plan (GPU+CPU+Disk shards)",
        "chat":         "Runs inference, streams tokens, detects loops",
    }

    total_static_mb = 0.0
    for i, w in enumerate(orch.worker_summary, 1):
        model_col = "[good]YES[/good]" if w["model"] else "[dim_text]no[/dim_text]"
        ram_str = f"{w['ram_mb']:.0f} MB"
        if not w["model"]:
            total_static_mb += w["ram_mb"]

        table.add_row(
            str(i),
            w["id"],
            ram_str,
            model_col,
            descriptions.get(w["id"], ""),
        )

    console.print(table)
    console.print(
        f"  [dim_text]Total static worker overhead: ~{total_static_mb:.0f} MB  "
        f"(model worker RAM varies by model + quantization)[/dim_text]"
    )
    console.print()


# ─────────────────────────────────────────────
# Hardware scan wrapper
# ─────────────────────────────────────────────

def run_hardware_scan(orch: Orchestrator) -> bool:
    with console.status("[bold cyan]Scanning hardware…[/bold cyan]", spinner="dots"):
        res = orch.scan_hardware()
    if not res:
        console.print(f"[error]Hardware scan failed: {res.error}[/error]")
        return False
    ui.print_system_profile(orch.profile)
    return True


# ─────────────────────────────────────────────
# Plan display wrapper
# ─────────────────────────────────────────────

def run_plan(orch: Orchestrator, force_cpu: bool, prefer_quality: bool) -> bool:
    with console.status("[bold cyan]Computing partition plan…[/bold cyan]", spinner="dots"):
        res = orch.compute_plan(force_cpu=force_cpu, prefer_quality=prefer_quality)
    if not res:
        console.print(f"[error]Planning failed: {res.error}[/error]")
        return False
    ui.print_partition_plan(orch.plan)
    return True


# ─────────────────────────────────────────────
# Model load wrapper
# ─────────────────────────────────────────────

def load_model(orch: Orchestrator, debug: bool = False) -> bool:
    def on_progress(msg: str):
        console.print(f"  [dim_text]>  {msg}[/dim_text]")

    try:
        console.print()
        console.print("[bold cyan]Loading model into memory (via ShardLoaderWorker)…[/bold cyan]")
        t0 = time.time()
        res = orch.load_model(progress_callback=on_progress)
        elapsed = time.time() - t0

        if res:
            console.print(f"  [good]Model loaded in {elapsed:.1f}s[/good]")
            return True
        else:
            console.print(f"\n  [error]Failed to load model: {res.error}[/error]")
            _print_load_hints()
            return False
    except Exception as e:
        console.print(f"\n  [error]Unexpected error loading model: {e}[/error]")
        _print_load_hints()
        if debug:
            traceback.print_exc()
        return False


def _print_load_hints():
    console.print()
    console.print("[warn]Common fixes:[/warn]")
    console.print("  • Install dependencies:  [bold]pip install -r requirements.txt[/bold]")
    console.print("  • For 4-bit/8-bit:       [bold]pip install bitsandbytes[/bold]")
    console.print("  • For GPU:               [bold]pip install torch --index-url https://download.pytorch.org/whl/cu121[/bold]")
    console.print("  • Accept model license on HuggingFace Hub")
    console.print("  • Run:                   [bold]huggingface-cli login[/bold]")


# ─────────────────────────────────────────────
# Chat loop
# ─────────────────────────────────────────────

def run_chat(
    orch: Orchestrator,
    max_tokens: int = 512,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
    stream: bool = True,
    optimize_tokens: bool = True,
    debug: bool = False,
):
    ui.print_chat_header(orch.model.name, orch.plan)

    while True:
        try:
            user_input = input("  You > ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim_text]Interrupted. Goodbye![/dim_text]")
            break

        if not user_input:
            continue

        # ── Built-in commands ──────────────────────────────────────────
        cmd = user_input.lower()

        if cmd in ("/quit", "/exit", "/q"):
            console.print("[dim_text]Goodbye![/dim_text]")
            break

        if cmd == "/info":
            res = orch.memory_status()
            if res:
                ui.print_memory_usage(res.data)
            continue

        if cmd == "/plan":
            ui.print_partition_plan(orch.plan)
            continue

        if cmd == "/workers":
            print_worker_summary(orch)
            continue

        if cmd == "/tokens":
            # Show token budget advisory for a sample prompt
            budget_res = orch.get_kv_budget(prompt_tokens=20, requested_max=max_tokens)
            if budget_res:
                d = budget_res.data
                console.print(Panel(
                    f"  KV Cache Budget : [bold white]{d['safe_context_tokens']}[/bold white] tokens\n"
                    f"  Recommended Max : [bold white]{d['recommended_max_new']}[/bold white] tokens\n"
                    f"  KV Cost/1K tok  : [dim]{d['kv_mb_per_1k_tokens']:.2f} MB[/dim]\n"
                    f"  KV Headroom     : [dim]{d['kv_headroom_mb']:.0f} MB[/dim]",
                    title="[header][OPT] KV Budget Worker[/header]",
                    border_style="cyan",
                ))
            continue

        if cmd == "/system":
            console.print(f"  [dim_text]System prompt: {system_prompt or '(none)'}[/dim_text]")
            continue

        if cmd.startswith("/system "):
            system_prompt = user_input[8:].strip()
            console.print("  [good]System prompt updated.[/good]")
            continue

        if cmd == "/clear":
            orch.clear_history()
            console.print("  [good]Conversation history cleared.[/good]")
            continue

        if cmd == "/help":
            _print_chat_help()
            continue

        # ── Generate response ──────────────────────────────────────────
        console.print()
        ui.print_user_message(user_input)

        token_bar = ui.LiveTokenBar(
            model_name=orch.model.name,
            max_tokens=max_tokens,
            prompt_tokens=len(user_input.split()),
        )
        token_bar.start()

        def on_token(chunk: str):
            token_bar.update(chunk)

        try:
            res = orch.chat(
                user_input=user_input,
                system_prompt=system_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
                optimize_tokens=optimize_tokens,
                on_token=on_token if stream else None,
            )

            stop_reason = "completed"
            if res and res.data.get("stopped_early"):
                stop_reason = "loop_detected"
            if not stream and res:
                # For non-streaming, display text at once
                token_bar.update(res.data.get("generated_text", ""))

        except Exception as e:
            console.print(f"\n  [error]Generation error: {e}[/error]")
            if debug:
                traceback.print_exc()
            stop_reason = "error"
        finally:
            token_bar.finish(stop_reason=stop_reason)

        console.print()


def _print_chat_help():
    console.print(Panel(
        "  [bold]/quit[/bold]           Exit the chat\n"
        "  [bold]/info[/bold]           Show current memory usage\n"
        "  [bold]/plan[/bold]           Re-display the partition plan\n"
        "  [bold]/workers[/bold]        Show micro-worker breakdown & RAM budgets\n"
        "  [bold]/tokens[/bold]         Show KV-cache budget advisory\n"
        "  [bold]/system[/bold]         Show current system prompt\n"
        "  [bold]/system <text>[/bold]  Set a new system prompt\n"
        "  [bold]/clear[/bold]          Clear conversation history\n"
        "  [bold]/help[/bold]           Show this help",
        title="[header]Chat Commands[/header]",
        border_style="dim",
    ))


# ─────────────────────────────────────────────
# Interactive wizard
# ─────────────────────────────────────────────

def run_wizard(args, orch: Orchestrator):
    ui.print_banner()

    # 1. Hardware scan
    if not run_hardware_scan(orch):
        sys.exit(1)

    # 2. Model selection
    if args.model:
        res = orch.select_model(args.model)
        if not res:
            console.print(f"[error]{res.error}[/error]")
            console.print(f"  Valid keys: {', '.join(CATALOG.keys())}")
            sys.exit(1)
    else:
        model_key = ui.select_model(orch.profile)
        if model_key is None:
            console.print("[dim_text]No model selected. Exiting.[/dim_text]")
            return
        res = orch.select_model(model_key)
        if not res:
            console.print(f"[error]{res.error}[/error]")
            sys.exit(1)

    console.print(f"\n  [good]Selected:[/good] [bold]{orch.model.name}[/bold]")
    console.print(f"  [dim_text]{orch.model.description}[/dim_text]")
    console.print()

    # 3. Cache check
    with console.status("[bold cyan]Checking model cache…[/bold cyan]", spinner="dots"):
        cache_res = orch.check_cache()
    if cache_res and cache_res.data["is_cached"]:
        console.print(f"  [good]Model already cached[/good] ({cache_res.data['size_gb']:.1f} GB)")
    else:
        console.print("  [dim_text]Model not cached — will download when loading.[/dim_text]")

    # 4. Partition plan
    if not run_plan(orch, force_cpu=args.force_cpu, prefer_quality=args.quality):
        sys.exit(1)

    if args.plan_only:
        return

    # 5. Feasibility warning
    if not orch.plan.is_feasible:
        console.print(
            "\n  [error]⚠  This model may not run well on your hardware. "
            "Performance could be extremely poor.[/error]"
        )
        if not ui.ask_confirm("Proceed anyway?", default=False):
            console.print("[dim_text]Exiting.[/dim_text]")
            return

    # 6. Load confirmation
    mem_gb = orch.plan.total_memory_mb / 1024
    console.print(
        f"\n  This will use approximately [bold]{mem_gb:.1f} GB[/bold] of memory. "
        f"Estimated speed: [bold]~{orch.plan.estimated_tokens_per_sec} tok/s[/bold]."
    )

    if not ui.ask_confirm("Load model?", default=True):
        console.print("[dim_text]Exiting.[/dim_text]")
        return

    # 7. Load model
    if not load_model(orch, debug=args.debug):
        sys.exit(1)

    # 8. System prompt
    system_prompt = args.system
    if system_prompt is None and not args.model:
        if ui.ask_confirm("Set a system prompt?", default=False):
            system_prompt = ui.ask_text("System prompt")

    # 9. Chat
    try:
        run_chat(
            orch,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            system_prompt=system_prompt,
            stream=not args.no_stream,
            optimize_tokens=not args.no_optimize,
            debug=args.debug,
        )
    finally:
        res = orch.unload()
        if res:
            console.print("[dim_text]Model unloaded. Memory released.[/dim_text]")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    parser = build_parser()
    args   = parser.parse_args()

    orch = Orchestrator()

    # ── Scan only ─────────────────────────────
    if args.scan:
        ui.print_banner()
        run_hardware_scan(orch)
        return

    # ── List models only ───────────────────────
    if args.list:
        ui.print_banner()
        with console.status("[cyan]Scanning hardware…[/cyan]", spinner="dots"):
            orch.scan_hardware()
        ui.print_model_list(orch.profile)
        return

    # ── Worker summary only ────────────────────
    if args.workers:
        ui.print_banner()
        print_worker_summary(orch)
        return

    # ── Full interactive wizard ────────────────
    run_wizard(args, orch)


if __name__ == "__main__":
    main()
