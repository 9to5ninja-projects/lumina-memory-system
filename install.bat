@echo off
REM Quick installation script for Lumina Memory System (Windows)

echo 🚀 Installing Lumina Memory System...

REM Check Python version
python --version
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.9+
    exit /b 1
)

REM Install core package
echo 📦 Installing core dependencies...
pip install -e .

REM Install SpaCy model
echo 🔥 Installing SpaCy model for lexical attribution...
python -m spacy download en_core_web_sm

REM Optional installations
set /p install_full="🤔 Install full dependencies (ML, database, API)? [y/N]: "
if /i "%install_full%"=="y" (
    echo 📚 Installing full dependencies...
    pip install -e ".[full]"
)

set /p install_dev="🛠️  Install development dependencies? [y/N]: "
if /i "%install_dev%"=="y" (
    echo 🔧 Installing development dependencies...
    pip install -e ".[dev]"
)

echo.
echo ✅ Installation complete!
echo.
echo 🎯 Quick Start:
echo    python -c "from lumina_memory import MemorySystem; print('✅ Lumina Memory ready!')"
echo.
echo 📖 Documentation:
echo    https://lumina-memory.readthedocs.io
echo.
echo 🚀 Happy memory building!
pause
