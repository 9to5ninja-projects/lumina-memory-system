"""
Mathematical Foundation - ACTUAL FORMULAS from xp_core_design.ipynb
==================================================================

These are the real mathematical functions extracted from the working notebook.
All formulas are validated and production-ready.

Source: xp_core_design.ipynb cells 1-3
Documentation: docs/CANONICAL_MATH_REFERENCE.md
Constants: lumina_memory.constants (canonical values)

Author: Lumina Memory Team  
License: MIT
"""

import numpy as np
import time
from typing import Optional

# Import canonical constants
from .constants import (
    EPSILON, NORMALIZATION_EPSILON, DEFAULT_W_SEMANTIC, DEFAULT_W_EMOTION,
    COHERENCE_HRR_WEIGHT, COHERENCE_SEM_WEIGHT, HYBRID_SPACY_WEIGHT, 
    HYBRID_MATH_WEIGHT, CONFIDENCE_HYBRID, CONFIDENCE_MATH_ONLY,
    MIN_SCORE, MAX_SCORE, VECTOR_DTYPE
)

# =============================================================================
# HRR OPERATIONS - ACTUAL FORMULAS FROM NOTEBOOK CELL 2
# =============================================================================

def circular_convolution(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """HRR binding operation - ACTUAL formula from notebook cell 2"""
    return np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=len(a))

def circular_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """HRR unbinding operation - ACTUAL formula from notebook cell 2"""
    return np.fft.irfft(np.fft.rfft(a) * np.conj(np.fft.rfft(b)), n=len(a))

def normalize_vector(v: np.ndarray, epsilon: float = NORMALIZATION_EPSILON) -> np.ndarray:
    """Normalize vector with stability - ACTUAL formula from notebook"""
    norm = np.linalg.norm(v)
    if norm < epsilon:
        return np.zeros_like(v)
    return v / norm

def bind_role_filler(role: np.ndarray, filler: np.ndarray) -> np.ndarray:
    """Bind role-filler pair - ACTUAL formula from notebook cell 2"""
    return normalize_vector(circular_convolution(role, filler))

def unbind_role_filler(bound_vector: np.ndarray, role: np.ndarray) -> np.ndarray:
    """Unbind to get filler - ACTUAL formula from notebook cell 2"""
    return normalize_vector(circular_correlation(bound_vector, role))

# =============================================================================
# MEMORY SCORING - ACTUAL FORMULAS FROM MEMORY UNIT IN CELL 2
# =============================================================================

def memory_unit_score(query_semantic: np.ndarray, memory_semantic: np.ndarray,
                     query_emotion: Optional[np.ndarray] = None, 
                     memory_emotion: Optional[np.ndarray] = None,
                     age_hours: float = 0.0, decay_rate: float = 0.1,
                     importance: float = 1.0,
                     w_semantic: float = DEFAULT_W_SEMANTIC, 
                     w_emotion: float = DEFAULT_W_EMOTION) -> float:
    """
    ACTUAL memory scoring formula from MemoryUnit.score() in notebook cell 2
    
    Formula:
    1. semantic_sim = cosine(query_semantic, memory_semantic)  
    2. emotion_sim = cosine(query_emotion, memory_emotion) [if provided]
    3. total_score = w_semantic * semantic_sim + w_emotion * emotion_sim
    4. decay_factor = exp(-decay_rate * age_hours)
    5. final_score = total_score * decay_factor * importance
    """
    # Semantic similarity (cosine)
    semantic_sim = np.dot(query_semantic, memory_semantic) / (
        np.linalg.norm(query_semantic) * np.linalg.norm(memory_semantic)
    )
    
    if query_emotion is not None and memory_emotion is not None:
        # Emotional similarity 
        emotion_sim = np.dot(query_emotion, memory_emotion) / (
            np.linalg.norm(query_emotion) * np.linalg.norm(memory_emotion)
        )
        # Weighted combination
        total_score = w_semantic * semantic_sim + w_emotion * emotion_sim
    else:
        total_score = semantic_sim
    
    # Temporal decay - ACTUAL formula from notebook
    decay_factor = np.exp(-decay_rate * age_hours)
    
    # Final score with importance weighting
    final_score = total_score * decay_factor * importance
    
    return float(np.clip(final_score, MIN_SCORE, MAX_SCORE))

def mathematical_coherence(hrr1: np.ndarray, hrr2: np.ndarray,
                          sem1: np.ndarray, sem2: np.ndarray) -> float:
    """
    ACTUAL coherence formula from MemoryUnit.mathematical_coherence() in notebook cell 2
    
    Formula:
    1. hrr_similarity = cosine(hrr1, hrr2)
    2. semantic_similarity = cosine(sem1, sem2)  
    3. coherence = 0.6 * hrr_similarity + 0.4 * semantic_similarity
    """
    # HRR similarity
    hrr_similarity = np.dot(hrr1, hrr2) / (np.linalg.norm(hrr1) * np.linalg.norm(hrr2))
    
    # Semantic similarity  
    semantic_similarity = np.dot(sem1, sem2) / (np.linalg.norm(sem1) * np.linalg.norm(sem2))
    
    # Combined coherence - ACTUAL weights from notebook
    coherence = COHERENCE_HRR_WEIGHT * hrr_similarity + COHERENCE_SEM_WEIGHT * semantic_similarity
    return float(np.clip(coherence, MIN_SCORE, MAX_SCORE))

# =============================================================================
# LEXICAL ATTRIBUTION - ACTUAL FORMULAS FROM NOTEBOOK CELL 2
# =============================================================================

def instant_salience(text: str, concept: str) -> float:
    """
    ACTUAL instant salience formula from notebook cell 2
    
    Formula: Jaccard similarity = |intersection| / |union|
    """
    if not text or not concept:
        return 0.0
    
    text_words = set(text.lower().split())
    concept_words = set(concept.lower().replace('_', ' ').split())
    
    # Jaccard similarity - ACTUAL formula from notebook
    intersection = len(text_words & concept_words)
    union = len(text_words | concept_words)
    
    return intersection / union if union > 0 else 0.0

def hybrid_lexical_attribution(text: str, concept: str, 
                             spacy_similarity: float = 0.0,
                             use_spacy: bool = False) -> dict:
    """
    ACTUAL hybrid attribution from HybridLexicalAttributor in notebook cell 2
    
    Formula: Uses canonical weights from constants module
    """
    math_salience = instant_salience(text, concept)
    
    if use_spacy and spacy_similarity > 0:
        # Hybrid combination - ACTUAL weights from notebook (now canonical)
        final_salience = HYBRID_SPACY_WEIGHT * spacy_similarity + HYBRID_MATH_WEIGHT * math_salience
        confidence = CONFIDENCE_HYBRID
        method = 'hybrid'
    else:
        final_salience = math_salience
        confidence = CONFIDENCE_MATH_ONLY
        method = 'mathematical'
    
    return {
        'salience': final_salience,
        'confidence': confidence,
        'method': method
    }

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_current_timestamp() -> float:
    """Get current timestamp in seconds."""
    return time.time()

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# =============================================================================
# EXPORT THE ACTUAL WORKING FORMULAS
# =============================================================================

__all__ = [
    # HRR operations - from notebook cell 2
    'circular_convolution', 'circular_correlation', 'normalize_vector',
    'bind_role_filler', 'unbind_role_filler',
    
    # Memory scoring - from MemoryUnit class in cell 2
    'memory_unit_score', 'mathematical_coherence',
    
    # Lexical attribution - from notebook cell 2
    'instant_salience', 'hybrid_lexical_attribution',
    
    # Utilities
    'get_current_timestamp', 'cosine_similarity'
]
