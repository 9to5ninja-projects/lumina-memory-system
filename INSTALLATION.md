# 🚀 Installation & Setup Guide

**Complete setup instructions for Lumina Memory System with SpaCy integration**

## 📋 **Prerequisites**

### **System Requirements**
- **Python**: 3.8+ (3.13+ recommended for best performance)
- **Memory**: 4GB+ RAM (8GB+ recommended for large models)
- **Storage**: 2GB+ free space (for SpaCy models and dependencies)
- **OS**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)

### **Hardware Recommendations**
- **CPU**: Modern multi-core processor (SpaCy benefits from multiple cores)
- **GPU**: Optional - CUDA-compatible GPU for PyTorch acceleration
- **Network**: Internet connection required for model downloads

## 🔧 **Quick Installation**

### **1. Clone Repository**
```bash
git clone https://github.com/9to5ninja-projects/lumina-memory-system.git
cd lumina-memory-system
```

### **2. Install Dependencies**
```bash
# Core dependencies
pip install -r requirements.txt

# Development dependencies (optional)
pip install -r requirements-dev.txt
```

### **3. Setup SpaCy Models**
```bash
# Automated setup with verification
python setup_dependencies.py

# Or manual setup
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md  # Optional, larger model with vectors
```

### **4. Verify Installation**
```python
# Quick verification script
from lumina_memory import MemorySystem
import spacy

# Test SpaCy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Lumina memory system is working!")
print(f"✅ SpaCy: {len(doc)} tokens processed")

# Test core system
memory = MemorySystem()
print("✅ Lumina Memory System initialized")
```

## 🧬 **Advanced Installation**

### **GPU Acceleration (Optional)**
```bash
# For NVIDIA GPUs with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install faiss-gpu

# Replace faiss-cpu with faiss-gpu in requirements.txt
```

### **Development Environment**
```bash
# Install development tools
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install

# Verify code quality tools
black --version
mypy --version
ruff --version
```

### **Notebook Environment**
```bash
# Install Jupyter ecosystem
pip install jupyter jupyterlab ipywidgets

# Launch development environment
jupyter lab notebooks/
```

## 📦 **Dependency Details**

### **Core ML/NLP Stack**
```txt
spacy>=3.8.7              # Production NLP pipeline
transformers>=4.55.0      # Hugging Face transformers
sentence-transformers>=5.1.0  # Sentence embeddings
torch>=2.8.0              # PyTorch framework
faiss-cpu>=1.11.0         # Vector similarity search
```

### **Scientific Computing**
```txt
numpy>=2.3.2              # Numerical computing
scipy>=1.16.1             # Scientific algorithms
scikit-learn>=1.7.1       # Machine learning utilities
pandas>=2.3.1             # Data manipulation
```

### **Visualization & Graphics**
```txt
matplotlib>=3.10.5        # Plotting and visualization
networkx>=3.4.2           # Graph structures and network analysis
```

### **Security & Utilities**
```txt
cryptography>=45.0.6      # Cryptographic operations
blake3>=1.0.5             # Fast hashing
pydantic>=2.11.7          # Data validation
tqdm>=4.67.1              # Progress bars
```

## 🔍 **Troubleshooting**

### **Common Installation Issues**

#### **SpaCy Model Download Fails**
```bash
# Manual download
python -m spacy download en_core_web_sm --direct

# Alternative: Download and install manually
wget https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
pip install en_core_web_sm-3.8.0-py3-none-any.whl
```

#### **PyTorch Installation Issues**
```bash
# CPU-only version (smaller, no GPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Check PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

#### **FAISS Import Errors**
```bash
# Ensure correct FAISS version
pip uninstall faiss faiss-cpu faiss-gpu
pip install faiss-cpu  # or faiss-gpu for GPU support

# Test FAISS
python -c "import faiss; print(f'FAISS: {faiss.__version__}')"
```

#### **Memory Issues**
```bash
# Reduce SpaCy model memory usage
python -c "
import spacy
from spacy.util import prefer_gpu
prefer_gpu()  # Use GPU if available
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])  # Disable heavy components
"
```

### **Version Compatibility**

#### **Python Version Issues**
- **Python 3.7**: Not supported (missing required features)
- **Python 3.8-3.10**: Fully supported
- **Python 3.11-3.13**: Recommended (best performance)

#### **SpaCy Model Compatibility**
```bash
# Check model compatibility
python -m spacy validate

# Update models if needed
python -m spacy download en_core_web_sm --upgrade
```

## 🧪 **Verification Tests**

### **Complete System Test**
```python
# Run comprehensive verification
python setup_dependencies.py

# Expected output:
# ✅ SpaCy: Production NLP pipeline ready
# ✅ FAISS: Vector search operational  
# ✅ PyTorch: ML framework loaded
# ✅ All dependencies verified successfully!
```

### **Performance Benchmark**
```python
# Quick performance test
from lumina_memory.attribution import HybridLexicalAttributor
import time

attributor = HybridLexicalAttributor()
text = "The quantum holographic memory system processes information efficiently."

start = time.time()
result = attributor.compute_attribution(text)
duration = (time.time() - start) * 1000

print(f"✅ Attribution computed in {duration:.3f}ms")
# Target: <0.1ms for simple texts
```

### **Integration Test**
```python
# Test notebook execution
jupyter nbconvert --execute --to notebook notebooks/xp_core_design.ipynb
# Should execute without errors
```

## 🔄 **Updates & Maintenance**

### **Regular Updates**
```bash
# Update dependencies (monthly recommended)
pip install --upgrade -r requirements.txt

# Update SpaCy models
python -m spacy download en_core_web_sm --upgrade

# Verify after updates
python setup_dependencies.py
```

### **Security Updates**
```bash
# Check for security vulnerabilities
pip-audit

# Update security-sensitive packages
pip install --upgrade cryptography pydantic
```

## 🆘 **Getting Help**

### **Documentation**
- **[README.md](README.md)**: Main project overview
- **[DEPENDENCIES.md](DEPENDENCIES.md)**: Detailed dependency information
- **[CHANGELOG.md](CHANGELOG.md)**: Version history and changes

### **Community Support**
- **Issues**: [GitHub Issues](https://github.com/9to5ninja-projects/lumina-memory-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/9to5ninja-projects/lumina-memory-system/discussions)

### **Debugging Mode**
```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with debug output
python -v setup_dependencies.py
```

---

*Installation guide last updated: August 14, 2025*  
*For the latest installation instructions, see [README.md](README.md)*
