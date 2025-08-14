"""
Holographic Reduced Representations (HRR) for semantic vector operations.

This module provides reference_vector() for generating deterministic
high-dimensional vectors from content and metadata for binding operations.
"""

import hashlib
import json
import numpy as np
from typing import Dict, Any, Optional, Union

def _canonicalize_for_hrr(content: Optional[str], metadata: Dict[str, Any]) -> str:
    """
    Canonicalize content and metadata for deterministic HRR vector generation.
    
    Args:
        content: Content text (can be None or empty)
        metadata: Metadata dictionary
        
    Returns:
        Canonical string representation
    """
    # Handle None/empty content
    canonical_content = content or ""
    
    # Create deterministic representation
    hrr_data = {
        "content": canonical_content,
        "metadata": metadata or {}
    }
    
    # Sort keys for deterministic JSON
    return json.dumps(hrr_data, sort_keys=True, separators=(',', ':'), ensure_ascii=True)

def _seed_from_string(text: str) -> int:
    """Generate deterministic seed from string."""
    hash_obj = hashlib.sha256(text.encode('utf-8'))
    # Use first 8 bytes of hash as seed (convert to int)
    seed_bytes = hash_obj.digest()[:8]
    return int.from_bytes(seed_bytes, byteorder='big') % (2**32)

def reference_vector(
    content: Optional[str], 
    metadata: Dict[str, Any], 
    dim: int = 256
) -> np.ndarray:
    """
    Generate deterministic HRR reference vector from content and metadata.
    
    Args:
        content: Content text (None/empty allowed)
        metadata: Metadata dictionary
        dim: Vector dimension
        
    Returns:
        Unit-normalized numpy array of specified dimension
    """
    # Create canonical representation
    canonical = _canonicalize_for_hrr(content, metadata)
    
    # Generate deterministic seed
    seed = _seed_from_string(canonical)
    
    # Create random number generator with deterministic seed
    rng = np.random.RandomState(seed)
    
    # Generate random vector from normal distribution
    vector = rng.normal(0, 1, dim)
    
    # Normalize to unit vector
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    else:
        # Handle edge case where norm is 0 (very unlikely)
        vector = np.ones(dim) / np.sqrt(dim)
    
    return vector

def bind_vectors(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """
    Bind two HRR vectors using circular convolution.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Bound vector (circular convolution)
    """
    if vec1.shape != vec2.shape:
        raise ValueError(f"Vector dimensions must match: {vec1.shape} vs {vec2.shape}")
    
    # Use FFT for efficient circular convolution
    fft1 = np.fft.fft(vec1)
    fft2 = np.fft.fft(vec2)
    
    # Element-wise multiplication in frequency domain = convolution in time domain
    bound_fft = fft1 * fft2
    
    # Convert back to time domain (real part only for real vectors)
    bound = np.fft.ifft(bound_fft).real
    
    return bound

def unbind_vectors(bound: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """
    Unbind HRR vectors using circular correlation.
    
    If bound = vec1  vec2 (where  is circular convolution),
    then vec1  bound  vec2* (where  is circular correlation, vec2* is inverted)
    
    Args:
        bound: Bound vector (result of previous binding)
        vec2: Vector to unbind
        
    Returns:
        Approximation of original first vector
    """
    if bound.shape != vec2.shape:
        raise ValueError(f"Vector dimensions must match: {bound.shape} vs {vec2.shape}")
    
    # Invert vec2 (reverse order except first element for circular correlation)
    vec2_inv = np.concatenate([[vec2[0]], vec2[1:][::-1]])
    
    # Circular correlation is circular convolution with inverted vector
    return bind_vectors(bound, vec2_inv)

def similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity (-1 to 1)
    """
    if vec1.shape != vec2.shape:
        raise ValueError(f"Vector dimensions must match: {vec1.shape} vs {vec2.shape}")
    
    # Compute dot product (both should be unit vectors)
    dot_product = np.dot(vec1, vec2)
    
    # Clamp to [-1, 1] to handle numerical errors
    return np.clip(dot_product, -1.0, 1.0)

def superpose_vectors(vectors: list, weights: Optional[list] = None) -> np.ndarray:
    """
    Superpose (add) multiple HRR vectors with optional weights.
    
    Args:
        vectors: List of numpy arrays to superpose
        weights: Optional list of weights (defaults to equal weights)
        
    Returns:
        Superposed vector
    """
    if not vectors:
        raise ValueError("Cannot superpose empty list of vectors")
    
    if weights is None:
        weights = [1.0] * len(vectors)
    
    if len(weights) != len(vectors):
        raise ValueError(f"Number of weights ({len(weights)}) must match number of vectors ({len(vectors)})")
    
    # Check all vectors have same dimension
    dim = vectors[0].shape[0]
    for i, vec in enumerate(vectors[1:], 1):
        if vec.shape[0] != dim:
            raise ValueError(f"Vector {i} has dimension {vec.shape[0]}, expected {dim}")
    
    # Weighted sum
    result = np.zeros(dim)
    for vec, weight in zip(vectors, weights):
        result += weight * vec
    
    return result

def create_concept_vector(
    concept_name: str,
    properties: Dict[str, Any],
    dim: int = 256
) -> np.ndarray:
    """
    Create a concept vector for a named concept with properties.
    
    Args:
        concept_name: Name of the concept
        properties: Properties/attributes of the concept
        dim: Vector dimension
        
    Returns:
        HRR reference vector for the concept
    """
    metadata = {
        "concept": concept_name,
        "properties": properties
    }
    
    return reference_vector(concept_name, metadata, dim)

def create_relation_vector(
    relation_name: str,
    relation_type: str = "generic",
    dim: int = 256
) -> np.ndarray:
    """
    Create a relation vector for binding concepts together.
    
    Args:
        relation_name: Name of the relation
        relation_type: Type/category of relation
        dim: Vector dimension
        
    Returns:
        HRR reference vector for the relation
    """
    metadata = {
        "relation": relation_name,
        "type": relation_type
    }
    
    return reference_vector(relation_name, metadata, dim)

# Utility functions for HRR operations

def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length."""
    norm = np.linalg.norm(vector)
    if norm > 0:
        return vector / norm
    else:
        return vector

def vector_magnitude(vector: np.ndarray) -> float:
    """Get magnitude (L2 norm) of vector."""
    return np.linalg.norm(vector)

def is_unit_vector(vector: np.ndarray, tolerance: float = 1e-6) -> bool:
    """Check if vector is approximately unit length."""
    magnitude = vector_magnitude(vector)
    return abs(magnitude - 1.0) < tolerance
