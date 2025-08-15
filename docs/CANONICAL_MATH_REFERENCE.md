# Lumina Memory System - Canonical Mathematical Reference
## Universal Documentation for All Environments

This document serves as the canonical reference for all mathematical formulas, dependencies, imports, variables, and functions used across the Lumina Memory System. **All implementations must reference this document to ensure consistency.**

---

## 📋 **REQUIRED DEPENDENCIES**

### Core Dependencies (Production)
```python
# Core numerical and scientific computing
numpy>=2.3.2           # Mathematical operations, FFT for HRR
time                    # Temporal calculations (built-in)
typing                  # Type annotations (built-in)

# Natural language processing  
spacy>=3.8.7           # Semantic embeddings, token similarity
en_core_web_sm         # English language model for spaCy

# Optional but recommended
torch>=2.8.0           # Advanced ML operations
networkx>=3.4.2        # Graph operations for memory networks
```

### Development Dependencies
```python
# Testing and validation
pytest>=8.3.3         # Unit testing framework
pytest-cov>=6.0.0     # Coverage reporting

# Code quality
black>=24.8.0          # Code formatting
flake8>=7.1.1          # Linting
mypy>=1.11.2           # Type checking
```

---

## 🧮 **CANONICAL MATHEMATICAL FORMULAS**

All formulas below are extracted from `xp_core_design.ipynb` cells 1-3 and validated in production.

### HRR (Holographic Reduced Representations) Operations

#### 1. Circular Convolution (Binding)
**Formula:** `a ⊛ b = IFFT(FFT(a) * FFT(b))`
```python
def circular_convolution(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """HRR binding operation - binds role and filler vectors"""
    return np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=len(a))
```
- **Source:** `xp_core_design.ipynb` Cell 2
- **Purpose:** Bind role-filler pairs in holographic memory
- **Input:** Two vectors of same dimension
- **Output:** Bound vector (same dimension)

#### 2. Circular Correlation (Unbinding)  
**Formula:** `a ⊕ b = IFFT(FFT(a) * CONJ(FFT(b)))`
```python
def circular_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """HRR unbinding operation - retrieves filler from bound vector"""
    return np.fft.irfft(np.fft.rfft(a) * np.conj(np.fft.rfft(b)), n=len(a))
```
- **Source:** `xp_core_design.ipynb` Cell 2
- **Purpose:** Unbind to retrieve original filler
- **Input:** Bound vector and role vector
- **Output:** Retrieved filler (approximate)

#### 3. Vector Normalization
**Formula:** `norm(v) = v / ||v||₂ if ||v||₂ > ε else 0`
```python
def normalize_vector(v: np.ndarray, epsilon: float = 1e-9) -> np.ndarray:
    """L2 normalize vector with numerical stability"""
    norm = np.linalg.norm(v)
    if norm < epsilon:
        return np.zeros_like(v)
    return v / norm
```
- **Source:** `xp_core_design.ipynb` Cell 2
- **Purpose:** Unit normalization for stable operations
- **Stability:** Uses epsilon to prevent division by zero

### Memory Unit Scoring

#### 4. Semantic Similarity (Cosine)
**Formula:** `sim(a,b) = (a·b) / (||a||₂ * ||b||₂)`
```python
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between normalized vectors"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
```

#### 5. Memory Unit Score (Complete Formula)
**Formula:** `score = (w_sem * sim_sem + w_emo * sim_emo) * exp(-r * t) * importance`
```python
def memory_unit_score(query_semantic: np.ndarray, memory_semantic: np.ndarray,
                     query_emotion: Optional[np.ndarray] = None, 
                     memory_emotion: Optional[np.ndarray] = None,
                     age_hours: float = 0.0, decay_rate: float = 0.1,
                     importance: float = 1.0,
                     w_semantic: float = 0.7, w_emotion: float = 0.3) -> float:
    # Semantic similarity
    semantic_sim = np.dot(query_semantic, memory_semantic) / (
        np.linalg.norm(query_semantic) * np.linalg.norm(memory_semantic)
    )
    
    if query_emotion is not None and memory_emotion is not None:
        emotion_sim = np.dot(query_emotion, memory_emotion) / (
            np.linalg.norm(query_emotion) * np.linalg.norm(memory_emotion)
        )
        total_score = w_semantic * semantic_sim + w_emotion * emotion_sim
    else:
        total_score = semantic_sim
    
    # Temporal decay
    decay_factor = np.exp(-decay_rate * age_hours)
    
    # Final score
    final_score = total_score * decay_factor * importance
    return float(np.clip(final_score, 0.0, 1.0))
```
- **Source:** `MemoryUnit.score()` in `xp_core_design.ipynb` Cell 2
- **Components:**
  - `w_semantic = 0.7` (semantic weight)
  - `w_emotion = 0.3` (emotion weight) 
  - `decay_rate = 0.1` (temporal decay rate)
  - `exp(-r * t)` (exponential temporal decay)

#### 6. Mathematical Coherence
**Formula:** `coherence = 0.6 * hrr_sim + 0.4 * sem_sim`
```python
def mathematical_coherence(hrr1: np.ndarray, hrr2: np.ndarray,
                          sem1: np.ndarray, sem2: np.ndarray) -> float:
    hrr_similarity = np.dot(hrr1, hrr2) / (np.linalg.norm(hrr1) * np.linalg.norm(hrr2))
    semantic_similarity = np.dot(sem1, sem2) / (np.linalg.norm(sem1) * np.linalg.norm(sem2))
    coherence = 0.6 * hrr_similarity + 0.4 * semantic_similarity
    return float(np.clip(coherence, 0.0, 1.0))
```
- **Source:** `MemoryUnit.mathematical_coherence()` in Cell 2
- **Weights:** HRR=0.6, Semantic=0.4 (validated in notebook)

### Lexical Attribution

#### 7. Instant Salience (Jaccard Similarity)
**Formula:** `salience = |intersection| / |union|`
```python
def instant_salience(text: str, concept: str) -> float:
    if not text or not concept:
        return 0.0
    
    text_words = set(text.lower().split())
    concept_words = set(concept.lower().replace('_', ' ').split())
    
    intersection = len(text_words & concept_words)
    union = len(text_words | concept_words)
    
    return intersection / union if union > 0 else 0.0
```
- **Source:** `instant_salience()` function in Cell 2
- **Method:** Jaccard similarity on word sets

#### 8. Hybrid Lexical Attribution
**Formula:** `final = 0.6 * spacy_sim + 0.4 * math_sim`
```python
def hybrid_lexical_attribution(text: str, concept: str, spacy_similarity: float = 0.0) -> dict:
    math_salience = instant_salience(text, concept)
    
    if spacy_similarity > 0:
        final_salience = 0.6 * spacy_similarity + 0.4 * math_salience
        confidence = 0.9
        method = 'hybrid'
    else:
        final_salience = math_salience
        confidence = 0.3
        method = 'mathematical'
    
    return {'salience': final_salience, 'confidence': confidence, 'method': method}
```
- **Source:** `HybridLexicalAttributor.compute_attribution()` in Cell 2
- **Weights:** SpaCy=0.6, Mathematical=0.4

---

## 📊 **CANONICAL VARIABLES AND CONSTANTS**

### Vector Dimensions
```python
SEMANTIC_DIM = 384      # spaCy sentence transformer dimension
EMOTION_DIM = 8         # Emotion vector dimension
HRR_DIM = 384          # HRR vector dimension (matches semantic)
HOLOGRAPHIC_DIM = 512   # Full holographic shape dimension
```

### Temporal Parameters
```python
DEFAULT_DECAY_RATE = 0.1        # Per hour decay rate
DEFAULT_IMPORTANCE = 1.0        # Maximum importance
DEFAULT_W_SEMANTIC = 0.7        # Semantic weight in scoring
DEFAULT_W_EMOTION = 0.3         # Emotion weight in scoring
```

### Mathematical Constants
```python
EPSILON = 1e-9                  # Numerical stability threshold
COHERENCE_HRR_WEIGHT = 0.6     # HRR weight in coherence
COHERENCE_SEM_WEIGHT = 0.4     # Semantic weight in coherence
HYBRID_SPACY_WEIGHT = 0.6      # SpaCy weight in hybrid attribution
HYBRID_MATH_WEIGHT = 0.4       # Math weight in hybrid attribution
```

---

## 🔧 **CANONICAL IMPORTS**

### Standard Imports (All Environments)
```python
import numpy as np
import time
from typing import Optional, Union, Dict, List, Tuple
from dataclasses import dataclass, field
```

### SpaCy Integration
```python
import spacy
try:
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except OSError:
    SPACY_AVAILABLE = False
    nlp = None
```

### Optional Advanced Dependencies
```python
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
```

---

## 🏗️ **CANONICAL DATA STRUCTURES**

### MemoryUnit (from Cell 2)
```python
@dataclass
class MemoryUnit:
    id: str
    content: str
    semantic_vector: np.ndarray
    emotion_vector: np.ndarray
    hrr_vector: Optional[np.ndarray] = None
    holographic_vector: Optional[np.ndarray] = None
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    importance: float = 1.0
    decay_rate: float = 0.1
    
    def score(self, query_semantic: np.ndarray, query_emotion: Optional[np.ndarray] = None, 
             current_time: Optional[float] = None) -> float:
        # Implementation using canonical formula #5 above
        pass
    
    def mathematical_coherence(self, other: 'MemoryUnit') -> float:
        # Implementation using canonical formula #6 above
        pass
```

---

## 🌍 **UNIVERSAL USAGE PATTERNS**

### In Notebooks
```python
# Cell 1: Always start with canonical imports
import sys
import os
sys.path.append(os.path.join(os.getcwd(), '..', 'src'))

# Cell 2: Import canonical functions
from lumina_memory.math_foundation import (
    circular_convolution, circular_correlation, normalize_vector,
    bind_role_filler, unbind_role_filler, memory_unit_score,
    mathematical_coherence, instant_salience
)
```

### In Production Scripts
```python
# Always use absolute imports from installed package
from lumina_memory.math_foundation import *
from lumina_memory.core import MemoryUnit
```

### In Tests
```python
# Use canonical reference for validation
from lumina_memory.math_foundation import circular_convolution
import numpy as np

def test_hrr_binding():
    a = np.random.randn(384)
    b = np.random.randn(384)
    result = circular_convolution(a, b)
    assert result.shape == (384,)  # Canonical dimension
```

---

## ✅ **VALIDATION CHECKLIST**

Before using any mathematical function in any environment:

1. ✅ **Formula Source:** Verified against `xp_core_design.ipynb` cells 1-3
2. ✅ **Import Path:** Uses canonical import from `lumina_memory.math_foundation`  
3. ✅ **Dependencies:** All required packages listed and version-pinned
4. ✅ **Constants:** Uses canonical constants (not hardcoded values)
5. ✅ **Type Annotations:** Proper typing for all parameters and returns
6. ✅ **Error Handling:** Numerical stability (epsilon, clipping, etc.)
7. ✅ **Documentation:** Function docstring references this canonical document

---

## 📚 **CROSS-ENVIRONMENT COMPATIBILITY**

### Jupyter Notebooks
- Use `%pip install` for missing dependencies
- Always check `SPACY_AVAILABLE` before using NLP features
- Import path: `../src/lumina_memory/math_foundation`

### Python Scripts  
- Use virtual environments with `requirements.txt`
- Import path: `lumina_memory.math_foundation` (installed package)
- Handle optional dependencies with try/except

### Production Servers
- Use container with pinned dependencies
- Environment variables for optional features
- Logging for dependency availability status

### Testing Environments
- Mock unavailable dependencies in tests
- Use canonical constants for expected values
- Validate formulas against reference implementation

---

**📌 This document is the single source of truth for all mathematical operations in the Lumina Memory System. All implementations must reference and comply with these canonical specifications.**
