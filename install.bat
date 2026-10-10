@echo off
echo.
echo ================================================================
echo   LLM --^> SLM Partitioner - Setup Script
echo ================================================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! Install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

echo [1/4] Installing core requirements...
pip install rich psutil requests colorama tabulate huggingface-hub safetensors sentencepiece accelerate

echo.
echo [2/4] Installing PyTorch (CPU version)...
pip install torch torchvision torchaudio

echo.
echo [3/4] Installing HuggingFace Transformers...
pip install transformers>=4.35.0

echo.
echo [4/4] Optional: Install bitsandbytes for 4-bit/8-bit quantization...
echo      (Required for running large models on limited VRAM/RAM)
echo      Attempting bitsandbytes install (Windows)...
pip install bitsandbytes

echo.
echo ================================================================
echo   Setup complete! You can now run:
echo     python main.py           -- Interactive wizard
echo     python main.py --scan    -- Hardware scan only
echo     python main.py --list    -- List models
echo ================================================================
echo.
pause
