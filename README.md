# 🧠 Lumina Memory System

**Industrial-Strength Holographic Memory with Production NLP Integration**

[![Version](https://img.shields.io/badge/version-v0.3.0--alpha-blue.svg)](./CHANGELOG.md)
[![Python](https://img.shields.io/badge/python-3.8%2B-green.svg)](https://python.org)
[![SpaCy](https://img.shields.io/badge/SpaCy-3.8.7%2B-orange.svg)](https://spacy.io)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](./LICENSE)

## 🔥 **Latest: Production NLP Integration (v0.3.0-alpha)**

- **SpaCy 3.8.7+ Integration**: Complete production NLP pipeline
- **59 Classes Mapped**: 15 SpaCy classes + 5 bridge integrations
- **4 Conflict Resolutions**: SpaCy-Lumina integration strategies
- **Industrial ML Stack**: Transformers, FAISS, PyTorch integration

## 🚀 **Quick Start**

### **Installation**
```bash
git clone https://github.com/9to5ninja-projects/lumina-memory-system.git
cd lumina-memory-system
pip install -r requirements.txt
python setup_dependencies.py  # Install SpaCy models + verification
```

### **Basic Usage**
```python
from lumina_memory import MemorySystem
from lumina_memory.nlp import HybridLexicalAttributor

# Initialize system with SpaCy integration
memory = MemorySystem()
attributor = HybridLexicalAttributor()

# Process text with production NLP
text = "The quantum holographic memory system uses lexical attribution."
attribution = attributor.compute_attribution(text)
memory.store(text, attribution)
```

## 🧬 **Architecture Overview**

### **Core Components**
- **🔬 XP Core**: Mathematical foundation with ultra-fast attribution (0.025ms)
- **🌉 Bridge System**: Integration layer between components
- **🧠 Memory System**: Holographic memory with decay mathematics
- **🔥 NLP Integration**: Production SpaCy pipeline with custom extensions

### **Class Architecture (59 Total)**
```
📦 Lumina Memory System
├── 🏗️ Core System (7 classes)
│   ├── MemorySystem, VectorStore, HRROperations
│   └── UnifiedMemory, UnifiedConfig, UnifiedKernel
│
├── 🔬 XP Core (32 classes)  
│   ├── HolographicShapeComputer, LexicalAttributor
│   ├── 15 SpaCy Classes (Doc, Token, Pipeline...)
│   └── 5 Bridge Classes (SpacyMemoryBridge...)
│
└── 🌉 Integration Layer (18 classes)
    ├── XPCoreBridge, SuperpositionSpaceBridge
    └── Configuration & Mock classes
```

## 🔬 **Scientific Foundation**

### **Holographic Distributed Representations (HDR)**
- **Circular Convolution**: Binding operation for concept association
- **Vector Superposition**: Additive composition of holographic memories
- **Decay Mathematics**: Temporal forgetting with consolidation scoring

### **Lexical Attribution System**
- **Ultra-Fast Processing**: 0.025ms per text (200,000x improvement)
- **SpaCy Integration**: Production NLP with linguistic features
- **Hybrid Architecture**: Simple + advanced attribution methods

## 🧪 **Development & Testing**

### **Notebook-Driven Development**
```bash
jupyter lab notebooks/
# Key notebooks:
# - xp_core_design.ipynb: Mathematical foundation + SpaCy integration
# - unit_space_kernel_bridge.ipynb: Integration architecture  
# - hd_kernel_xp_spec.ipynb: Specifications and interfaces
```

### **Testing**
```bash
pytest tests/                    # Run test suite
pytest --benchmark-only          # Performance benchmarks
python setup_dependencies.py    # Verify all dependencies
```

## 📦 **Dependencies**

### **Production Stack**
- **NLP**: SpaCy (3.8.7+), Transformers (4.55.0+)
- **ML**: PyTorch (2.8.0+), SentenceTransformers (5.1.0+) 
- **Vector Search**: FAISS (1.11.0+)
- **Scientific**: NumPy (2.3.2+), SciPy (1.16.1+)
- **Security**: Cryptography (45.0.6+), BLAKE3 (1.0.5+)

### **Language Models**
```bash
python -m spacy download en_core_web_sm  # Required
python -m spacy download en_core_web_md  # Optional (with vectors)
```

## 🔄 **Versioning & Releases**

### **Current Milestones**
- **v0.1.0-alpha**: Mathematical foundation (XP Core)
- **v0.2.0-alpha**: Ultra-fast attribution (0.025ms)
- **v0.3.0-alpha**: Production NLP integration (SpaCy)
- **v0.4.0-alpha**: *Planned - Unit Space Kernel*

### **Release Process**
See [VERSIONING_STRATEGY.md](VERSIONING_STRATEGY.md) for our 5-step semantic versioning process.

## 📚 **Documentation**

- **[CHANGELOG.md](CHANGELOG.md)**: Complete version history
- **[DEPENDENCIES.md](DEPENDENCIES.md)**: Comprehensive dependency management
- **[COMPLETE_CLASS_TREE.md](docs/COMPLETE_CLASS_TREE.md)**: Full class architecture
- **[VERSIONING_STRATEGY.md](VERSIONING_STRATEGY.md)**: Release methodology

## 🤝 **Contributing**

### **Development Setup**
```bash
pip install -r requirements-dev.txt
pre-commit install
```

### **Code Quality**
```bash
black src/                 # Format code
isort src/                 # Sort imports  
mypy src/                  # Type checking
ruff src/                  # Fast linting
```

## ⚖️ **License**

MIT License - see [LICENSE](LICENSE) for details.

## 🔗 **Links**

- **Repository**: [GitHub](https://github.com/9to5ninja-projects/lumina-memory-system)
- **Issues**: [Bug Reports](https://github.com/9to5ninja-projects/lumina-memory-system/issues)
- **Discussions**: [Feature Requests](https://github.com/9to5ninja-projects/lumina-memory-system/discussions)

---

*Built with 🧠 by the Lumina Memory Team*
