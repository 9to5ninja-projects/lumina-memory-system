# 📦 Lumina Memory System - Dependencies & Requirements

## 🎯 **Core Dependencies**

### **Production Dependencies (requirements.txt)**
```txt
# Core Scientific Computing
numpy>=2.3.2
scipy>=1.16.1
scikit-learn>=1.7.1

# Machine Learning & NLP
spacy>=3.8.7
transformers>=4.55.0
sentence-transformers>=5.1.0
torch>=2.8.0

# Vector Operations & Search
faiss-cpu>=1.11.0

# Data Processing
pandas>=2.3.1
joblib>=1.5.1

# Cryptography & Security
cryptography>=45.0.6
blake3>=1.0.5

# Utilities
pydantic>=2.11.7
tqdm>=4.67.1
```

### **Development Dependencies (requirements-dev.txt)**
```txt
# Testing Framework
pytest>=8.4.1
pytest-benchmark>=5.1.0
pytest-cov>=6.2.1
pytest-mock>=3.14.1
hypothesis>=6.137.3

# Code Quality
black>=25.1.0
isort>=6.0.1
mypy>=1.17.1
ruff>=0.12.8
pre-commit>=4.3.0

# Notebook Development
jupyter>=1.1.1
jupyterlab>=4.4.5
ipywidgets>=8.1.7
matplotlib>=3.10.5
seaborn>=0.13.2

# Documentation & Profiling
coverage>=7.10.3
py-cpuinfo>=9.0.0
```

## 🧬 **SpaCy Language Models**

### **Required Models**
```bash
# English language model (required for lexical attribution)
python -m spacy download en_core_web_sm

# Additional models (optional, for multilingual support)
python -m spacy download en_core_web_md  # Medium model with vectors
python -m spacy download en_core_web_lg  # Large model with more vectors
```

### **SpaCy Model Integration Classes**
- **Core SpaCy Classes**: 15 mapped classes
- **Lumina Integration Classes**: 5 custom bridge classes
- **Pipeline Components**: Tokenizer, Tagger, Parser, EntityRecognizer

## 🔬 **Class Dependencies Mapping**

### **External Library Classes**
```python
EXTERNAL_CLASSES = {
    # SpaCy Classes
    "spacy.lang.en.English": "Main NLP pipeline",
    "spacy.tokens.Doc": "Document container with tokens",
    "spacy.tokens.Token": "Individual token with linguistic features",
    "spacy.tokens.Span": "Text span with metadata",
    "spacy.vocab.Vocab": "Vocabulary and string-to-hash mapping",
    
    # Transformers Classes
    "transformers.AutoModel": "Hugging Face transformer models",
    "transformers.AutoTokenizer": "Tokenization for transformers",
    
    # FAISS Classes
    "faiss.IndexFlatL2": "L2 distance index for vector search",
    "faiss.IndexIVFFlat": "Inverted file index",
    
    # PyTorch Classes
    "torch.nn.Module": "Neural network modules",
    "torch.Tensor": "Multi-dimensional arrays"
}
```

### **Integration Bridge Classes**
```python
BRIDGE_CLASSES = {
    # SpaCy-Lumina Bridges
    "SpacyLexicalAttributor": "Lexical attribution using SpaCy",
    "SpacyMemoryBridge": "Connect SpaCy analysis to MemoryUnit",
    "SpacyHologramConnector": "Bridge SpaCy features to holographic shapes",
    
    # Transformer-Lumina Bridges  
    "TransformerEmbeddingBridge": "Connect transformer embeddings",
    "TransformerMemoryBridge": "Bridge transformer outputs to memory",
    
    # FAISS-Lumina Bridges
    "FAISSVectorBridge": "Connect FAISS indices to vector store"
}
```

## 🚀 **Installation Instructions**

### **1. Basic Installation**
```bash
cd lumina_memory_package
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### **2. SpaCy Model Setup**
```bash
python -m spacy download en_core_web_sm
```

### **3. Development Environment**
```bash
# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Format code
black src/
isort src/
```

### **4. Optional GPU Support**
```bash
# For GPU acceleration (CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install faiss-gpu
```

## ⚠️ **Dependency Conflicts & Resolutions**

### **Known Conflicts**
1. **SpaCy vs Transformers**: Different tokenization approaches
   - **Resolution**: Use `HybridLexicalAttributor` to bridge methods
   
2. **FAISS CPU vs GPU**: Cannot install both simultaneously
   - **Resolution**: Choose based on hardware availability
   
3. **PyTorch Version Compatibility**: Version conflicts with other ML libraries
   - **Resolution**: Pin specific compatible versions

### **Version Pinning Strategy**
- **Core ML Libraries**: Pin major.minor versions
- **Utility Libraries**: Allow patch version updates
- **Development Tools**: Latest stable versions preferred

## 📋 **Dependency Verification**

### **Health Check Commands**
```python
# SpaCy Health Check
import spacy
nlp = spacy.load("en_core_web_sm")
print(f"✅ SpaCy {spacy.__version__} with {nlp.meta['name']} model")

# Transformers Health Check  
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
print(f"✅ Transformers with tokenizer loaded")

# FAISS Health Check
import faiss
print(f"✅ FAISS {faiss.__version__} available")
```

## 🔄 **Update Strategy**

### **Regular Updates**
- **Monthly**: Security patches and minor updates
- **Quarterly**: Feature updates and major version evaluations
- **As Needed**: Critical security fixes

### **Testing Protocol**
1. Update in development branch
2. Run full test suite
3. Benchmark performance impact
4. Update documentation
5. Deploy to staging
6. Production rollout

---

*Last Updated: August 14, 2025*
*Next Review: September 14, 2025*
