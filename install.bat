@echo off
setlocal enabledelayedexpansion
title LLM -^> SLM Partitioner — Installer

echo.
echo ============================================================
echo   LLM -^> SLM Partitioner — Micro-Worker Architecture
echo   Install Script
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [OK] Found %PYVER%

:: Check pip
pip --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pip not found.
    pause
    exit /b 1
)
echo [OK] pip found

:: Core requirements
echo.
echo [1/4] Installing core requirements...
pip install rich psutil colorama requests tabulate --quiet
if errorlevel 1 goto :error
echo      Done.

:: HuggingFace stack
echo [2/4] Installing HuggingFace stack...
pip install huggingface-hub transformers accelerate sentencepiece safetensors tokenizers --quiet
if errorlevel 1 goto :error
echo      Done.

:: PyTorch - detect CUDA
echo [3/4] Installing PyTorch...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo      No CUDA GPU detected — installing CPU-only PyTorch...
    pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
) else (
    echo      CUDA GPU detected — installing CUDA 12.1 PyTorch...
    pip install torch --index-url https://download.pytorch.org/whl/cu121 --quiet
)
if errorlevel 1 goto :error
echo      Done.

:: Optional: bitsandbytes for 4/8-bit quantization
echo [4/4] Installing bitsandbytes (4-bit / 8-bit quantization)...
pip install bitsandbytes --quiet
if errorlevel 1 (
    echo      [WARN] bitsandbytes install failed — 4-bit/8-bit quant unavailable.
    echo             This is OK for fp16 CPU inference.
) else (
    echo      Done.
)

:: Run smoke test
echo.
echo [TEST] Running worker smoke tests...
python smoke_test.py
if errorlevel 1 (
    echo [WARN] Some smoke tests failed. Check output above.
) else (
    echo [OK]   All workers healthy.
)

echo.
echo ============================================================
echo   Installation complete!
echo   Run:  python main.py          (interactive wizard)
echo   Run:  python main.py --scan   (hardware scan)
echo   Run:  python main.py --list   (list models)
echo   Run:  python main.py --workers (show worker breakdown)
echo ============================================================
echo.
pause
exit /b 0

:error
echo.
echo [ERROR] An installation step failed. See output above.
pause
exit /b 1
