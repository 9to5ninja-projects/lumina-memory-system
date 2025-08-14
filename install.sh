#!/bin/bash
# Quick installation script for Lumina Memory System

echo "🚀 Installing Lumina Memory System..."

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "📋 Python version: $python_version"

if [[ $(echo "$python_version < 3.9" | bc -l) ]]; then
    echo "❌ Python 3.9+ required. Please upgrade Python."
    exit 1
fi

# Install core package
echo "📦 Installing core dependencies..."
pip install -e .

# Install SpaCy model for fast lexical attribution
echo "🔥 Installing SpaCy model for lexical attribution..."
python -m spacy download en_core_web_sm

# Optional: Install full dependencies
read -p "🤔 Install full dependencies (ML, database, API)? [y/N]: " install_full
if [[ $install_full == "y" || $install_full == "Y" ]]; then
    echo "📚 Installing full dependencies..."
    pip install -e ".[full]"
fi

# Optional: Install development dependencies
read -p "🛠️  Install development dependencies? [y/N]: " install_dev
if [[ $install_dev == "y" || $install_dev == "Y" ]]; then
    echo "🔧 Installing development dependencies..."
    pip install -e ".[dev]"
fi

echo ""
echo "✅ Installation complete!"
echo ""
echo "🎯 Quick Start:"
echo "   python -c \"from lumina_memory import MemorySystem; print('✅ Lumina Memory ready!')\""
echo ""
echo "📖 Documentation:"
echo "   https://lumina-memory.readthedocs.io"
echo ""
echo "🚀 Happy memory building!"
