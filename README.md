# 🧠 LLM → SLM Partitioner & Runner

Run Large Language Models locally on **any consumer hardware** by automatically partitioning the model into Small Language Model (SLM) layer shards tailored to your machine's CPU, RAM, GPU, and VRAM.

---

## 🚀 Key Features

1. **Automatic Hardware Profiling (`system_detector.py`)**
   - Detects CPU physical/logical cores, frequency, architecture.
   - Detects total & available system RAM.
   - Detects NVIDIA CUDA / Apple Silicon MPS GPUs & dedicated VRAM.
   - Measures free disk storage for model caching.
   - Categorizes your PC into hardware tiers: `Low`, `Mid`, `High`, or `Extreme`.

2. **Intelligent Model Partitioning (`model_partitioner.py`)**
   - Breaks an LLM (e.g., 7B, 14B, 70B) into **SLM shards** (contiguous transformer layer slices).
   - Dynamic Strategy Selection:
     - ⚡ **GPU-Only**: When all layers fit into VRAM.
     - 🔀 **GPU + CPU Hybrid**: Allocates maximum feasible layers to GPU VRAM and shards remainder into CPU RAM.
     - 🐢 **CPU-Only**: Fully runs in system RAM with CPU quantization.
     - 🐌 **Disk-Offload Fallback**: Swaps layer shards from disk when RAM/VRAM are constrained.
   - Automatic Quantization Selection (4-bit NF4, 8-bit, 16-bit) to minimize memory footprint.

3. **Built-in Model Catalog (`model_catalog.py`)**
   - Supports leading open-weight models:
     - `qwen2.5-0.5b`, `qwen2.5-1.5b`, `qwen2.5-3b`, `qwen2.5-7b`, `qwen2.5-14b`, `qwen2.5-32b`
     - `llama-3.2-3b`, `llama-3.1-8b`, `llama-3.1-70b`
     - `mistral-7b`
     - `phi-3.5-mini`
     - `gemma-2-2b`, `gemma-2-9b`

4. **Warnings & Adaptive Suggestions**
   - RAM exhaustion alerts with headroom guarantees.
   - VRAM bottleneck warnings.
   - Context length memory consumption guidance.
   - Suggestions tailored to system tier.

4. **Token Optimizer (`token_optimizer.py`)**
   - **Prompt Compression**: Cleans redundant whitespace and formatting bloat to reduce prompt tokens.
   - **Dynamic KV Cache Budgeting**: Analyzes available RAM/VRAM to set safe context limits and prevent OOM crashes.
   - **Sliding-Window Dialogue Pruning**: Retains system prompts and recent context while shedding older turns.
   - **Repetition Loop Detection**: Detects looping output patterns and terminates early to save hardware cycles.

5. **Live Token Count Bar (`ui.py`)**
   - **Real-Time Progress Bar**: Displays `[====>----] X/Y tokens (Z%)` live during generation.
   - **Live Speed**: Shows token generation rate in real-time (`tok/s`).
   - **Token Optimization Card**: Pre-generation breakdown of original tokens, compressed tokens, and tokens saved.

6. **Streaming Interactive Chat (`model_runner.py` & `main.py`)**
   - Token streaming directly to your terminal.
   - Real-time token generation speed (tok/s).
   - In-chat diagnostics: `/tokens`, `/info`, `/plan`, `/system`, `/quit`.

---

## 🛠️ Installation & Setup

Double-click `install.bat` or run:

```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### 1. Interactive Mode (Recommended)
Run the full interactive wizard:
```bash
python main.py
```
*(Or double-click `run.bat`)*

### 2. Quick Hardware Scan
Inspect your hardware classification and RAM/VRAM capacity:
```bash
python main.py --scan
```

### 3. List Models & Hardware Compatibility
Check which models fit in your system's VRAM or RAM:
```bash
python main.py --list
```

### 4. Partition Plan Simulation (Dry Run)
Inspect how any model will be divided without downloading or loading it:
```bash
python main.py --model qwen2.5-3b --plan-only
python main.py --model mistral-7b --plan-only
python main.py --model llama-3.1-70b --plan-only
```

### 5. Chat with a Model
Load and chat with a specific model directly:
```bash
python main.py --model qwen2.5-0.5b
```

---

## 🧩 Architectural Overview

```
                      [ User Hardware ]
             (CPU Cores, RAM, GPU VRAM, Disk)
                             │
                             ▼
                  ┌──────────────────────┐
                  │  system_detector.py  │
                  └──────────┬───────────┘
                             │ SystemProfile
                             ▼
  [ Model Catalog ] ──▶ ┌──────────────────────┐
   (Layers, Params,     │ model_partitioner.py │
    Quant Footprint)    └──────────┬───────────┘
                                   │ PartitionPlan (SLM Shards, Device Map, Quant)
                                   ▼
                        ┌──────────────────────┐
                        │   model_runner.py    │ ◀─── HuggingFace / Accelerate
                        └──────────┬───────────┘
                                   │ Streaming Tokens
                                   ▼
                        ┌──────────────────────┐
                        │       ui.py          │ ◀─── Rich Terminal Interface
                        └──────────────────────┘
```
