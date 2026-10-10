"""
main.py
-------
Entry point for the LLM -> SLM Partitioner application.

Usage:
    python main.py                   # Interactive mode
    python main.py --model qwen2.5-7b --chat
    python main.py --scan            # Just scan hardware
    python main.py --list            # List models
    python main.py --model llama-3.1-8b --plan-only
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
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.text import Text

# ── Local modules ─────────────────────────────────────────────────────────────
from system_detector import detect_system, SystemProfile
from model_catalog import CATALOG, ModelSpec, get_model
from model_partitioner import partition_model, PartitionPlan
from model_runner import ModelRunner
from token_optimizer import TokenOptimizer, TokenOptimizationResult
import ui

console = ui.console


# ─────────────────────────────────────────────
# CLI argument parser
# ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="slm-runner",
        description="Run large LLMs on local hardware by auto-partitioning them into SLM shards.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                         Interactive wizard
  python main.py --scan                  Show hardware report only
  python main.py --list                  List available models
  python main.py --model qwen2.5-3b     Load and chat with a model
  python main.py --model mistral-7b --plan-only
  python main.py --model llama-3.1-8b --force-cpu
  python main.py --model phi-3.5-mini --quality
        """
    )
    p.add_argument("--scan",        action="store_true", help="Scan hardware and exit")
    p.add_argument("--list",        action="store_true", help="List available models and exit")
    p.add_argument("--model",       type=str,            help="Model key (from catalog)")
    p.add_argument("--plan-only",   action="store_true", help="Show partition plan without loading")
    p.add_argument("--force-cpu",   action="store_true", help="Disable GPU and use CPU only")
    p.add_argument("--quality",     action="store_true", help="Prefer higher quality quantization")
    p.add_argument("--max-tokens",  type=int, default=512, help="Max new tokens per response")
    p.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    p.add_argument("--system",      type=str, default=None, help="System prompt text")
    p.add_argument("--no-stream",   action="store_true", help="Disable token streaming")
    p.add_argument("--no-optimize", action="store_true", help="Disable prompt and context token optimization")
    return p


# ─────────────────────────────────────────────
# Hardware scan
# ─────────────────────────────────────────────

def run_hardware_scan() -> SystemProfile:
    with console.status("[bold cyan]Scanning hardware…[/bold cyan]", spinner="dots"):
        profile = detect_system()
    ui.print_system_profile(profile)
    return profile


# ─────────────────────────────────────────────
# Plan display
# ─────────────────────────────────────────────

def run_plan(
    model: ModelSpec,
    profile: SystemProfile,
    force_cpu: bool = False,
    prefer_quality: bool = False,
) -> PartitionPlan:
    with console.status("[bold cyan]Computing partition plan…[/bold cyan]", spinner="dots"):
        plan = partition_model(model, profile, prefer_quality=prefer_quality, force_cpu=force_cpu)
    ui.print_partition_plan(plan)
    return plan


# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────

def load_model(model: ModelSpec, plan: PartitionPlan) -> Optional[ModelRunner]:
    runner = ModelRunner(model, plan)
    messages = []

    def on_progress(msg: str):
        messages.append(msg)
        console.print(f"  [dim_text]▶  {msg}[/dim_text]")

    try:
        console.print()
        console.print("[bold cyan]Loading model into memory…[/bold cyan]")
        t0 = time.time()
        runner.load(progress_callback=on_progress)
        elapsed = time.time() - t0
        console.print(f"  [good]✓ Model loaded in {elapsed:.1f}s[/good]")
        return runner
    except Exception as e:
        console.print(f"\n  [error]✗ Failed to load model: {e}[/error]")
        console.print()
        console.print("[warn]Common fixes:[/warn]")
        console.print("  • Install dependencies:  [bold]pip install -r requirements.txt[/bold]")
        console.print("  • For 4-bit/8-bit:       [bold]pip install bitsandbytes[/bold]")
        console.print("  • For GPU:               [bold]pip install torch --index-url https://download.pytorch.org/whl/cu121[/bold]")
        console.print("  • Accept model license on HuggingFace Hub")
        console.print("  • Run:                   [bold]huggingface-cli login[/bold]")
        if "--debug" in sys.argv:
            traceback.print_exc()
        return None


# ─────────────────────────────────────────────
# Chat loop
# ─────────────────────────────────────────────

def run_chat(
    runner: ModelRunner,
    model: ModelSpec,
    plan: PartitionPlan,
    profile: Optional[SystemProfile] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
    stream: bool = True,
    optimize_tokens: bool = True,
):
    ui.print_chat_header(model.name, plan)
    optimizer = TokenOptimizer(model, profile)

    while True:
        try:
            user_input = input("  You > ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim_text]Interrupted. Goodbye![/dim_text]")
            break

        if not user_input:
            continue

        # ── Commands ──────────────────────────────────
        if user_input.lower() in ("/quit", "/exit", "/q"):
            console.print("[dim_text]Goodbye![/dim_text]")
            break

        if user_input.lower() == "/info":
            usage = runner.memory_usage_mb()
            ui.print_memory_usage(usage)
            continue

        if user_input.lower() == "/plan":
            ui.print_partition_plan(plan)
            continue

        if user_input.lower() == "/tokens":
            # Display current token optimizer metrics
            opt_res = optimizer.optimize_prompt(user_input or "Hello", system_prompt, max_target_tokens=max_tokens)
            ui.print_token_optimization_card(opt_res)
            continue

        if user_input.lower() == "/system":
            console.print(f"  [dim_text]System prompt: {system_prompt or '(none)'}[/dim_text]")
            continue

        if user_input.lower().startswith("/system "):
            system_prompt = user_input[8:].strip()
            console.print(f"  [good]System prompt updated.[/good]")
            continue

        if user_input.lower() == "/help":
            _print_chat_help()
            continue

        # ── Token Optimization ────────────────────────
        effective_prompt = user_input
        effective_max_tokens = max_tokens
        prompt_tokens_est = optimizer.estimate_tokens(user_input)

        if optimize_tokens:
            opt_res = optimizer.optimize_prompt(user_input, system_prompt, max_target_tokens=max_tokens)
            effective_prompt = opt_res.optimized_text
            effective_max_tokens = opt_res.recommended_max_new_tokens
            prompt_tokens_est = opt_res.estimated_optimized_tokens

            # Show token optimizer summary if tokens were saved or budget was constrained
            if opt_res.tokens_saved > 0 or effective_max_tokens < max_tokens:
                ui.print_token_optimization_card(opt_res)

        # ── Generate with Live Token Count Bar ─────────
        console.print()
        ui.print_user_message(user_input)

        token_bar = ui.LiveTokenBar(
            model_name=model.name,
            max_tokens=effective_max_tokens,
            prompt_tokens=prompt_tokens_est,
        )

        recent_words: List[str] = []
        loop_detected = False

        try:
            token_bar.start()

            gen = runner.generate(
                prompt=effective_prompt,
                max_new_tokens=effective_max_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
                stream=stream,
            )

            for chunk in gen:
                token_bar.update(chunk)

                # Check for repetitive loops to stop early and save tokens
                chunk_words = chunk.split()
                if chunk_words:
                    recent_words.extend(chunk_words)
                    if len(recent_words) > 30:
                        recent_words = recent_words[-30:]
                        if TokenOptimizer.detect_repetition(recent_words, ngram_size=4, threshold=3):
                            loop_detected = True
                            break

        except Exception as e:
            console.print(f"\n  [error]Generation error: {e}[/error]")
            if "--debug" in sys.argv:
                traceback.print_exc()
            continue
        finally:
            stop_reason = "loop_detected" if loop_detected else "completed"
            token_bar.finish(stop_reason=stop_reason)

        console.print()


def _print_chat_help():
    console.print(Panel(
        "  [bold]/quit[/bold]           Exit the chat\n"
        "  [bold]/info[/bold]           Show current memory usage\n"
        "  [bold]/plan[/bold]           Re-display the partition plan\n"
        "  [bold]/tokens[/bold]         Show token optimizer & KV cache budget\n"
        "  [bold]/system[/bold]         Show current system prompt\n"
        "  [bold]/system <text>[/bold]  Set a new system prompt\n"
        "  [bold]/help[/bold]           Show this help",
        title="[header]Chat Commands[/header]",
        border_style="dim",
    ))


# ─────────────────────────────────────────────
# Interactive wizard
# ─────────────────────────────────────────────

def run_wizard(args):
    ui.print_banner()

    # 1. Hardware scan
    profile = run_hardware_scan()

    # 2. Model selection
    if args.model:
        model_key = args.model
        model = get_model(model_key)
        if model is None:
            console.print(f"[error]Unknown model key: '{model_key}'[/error]")
            console.print(f"  Valid keys: {', '.join(CATALOG.keys())}")
            sys.exit(1)
    else:
        model_key = ui.select_model(profile)
        if model_key is None:
            console.print("[dim_text]No model selected. Exiting.[/dim_text]")
            return
        model = get_model(model_key)

    console.print(f"\n  [good]Selected:[/good] [bold]{model.name}[/bold]")
    console.print(f"  [dim_text]{model.description}[/dim_text]")
    console.print()

    # 3. Partition plan
    plan = run_plan(model, profile, force_cpu=args.force_cpu, prefer_quality=args.quality)

    if args.plan_only:
        return

    # 4. Feasibility warning
    if not plan.is_feasible:
        console.print(
            "\n  [error]⚠  This model may not run well on your hardware. "
            "Performance could be extremely poor.[/error]"
        )
        if not ui.ask_confirm("Proceed anyway?", default=False):
            console.print("[dim_text]Exiting.[/dim_text]")
            return

    # 5. Load confirmation
    mem_gb = plan.total_memory_mb / 1024
    console.print(
        f"\n  This will use approximately [bold]{mem_gb:.1f} GB[/bold] of memory. "
        f"Estimated speed: [bold]~{plan.estimated_tokens_per_sec} tok/s[/bold]."
    )

    if not ui.ask_confirm("Load model?", default=True):
        console.print("[dim_text]Exiting.[/dim_text]")
        return

    # 6. Load model
    runner = load_model(model, plan)
    if runner is None:
        sys.exit(1)

    # 7. System prompt
    system_prompt = args.system
    if system_prompt is None and not args.model:
        if ui.ask_confirm("Set a system prompt?", default=False):
            system_prompt = ui.ask_text("System prompt")

    # 8. Chat
    try:
        run_chat(
            runner, model, plan,
            profile=profile,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            system_prompt=system_prompt,
            stream=not args.no_stream,
            optimize_tokens=not args.no_optimize,
        )
    finally:
        runner.unload()
        console.print("[dim_text]Model unloaded. Memory released.[/dim_text]")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    parser = build_parser()
    args = parser.parse_args()

    # ── Scan only ──────────────────────────────
    if args.scan:
        ui.print_banner()
        run_hardware_scan()
        return

    # ── List only ──────────────────────────────
    if args.list:
        ui.print_banner()
        profile = None
        try:
            with console.status("[cyan]Scanning hardware…[/cyan]", spinner="dots"):
                profile = detect_system()
        except Exception:
            pass
        ui.print_model_list(profile)
        return

    # ── Full wizard ────────────────────────────
    run_wizard(args)


if __name__ == "__main__":
    main()
