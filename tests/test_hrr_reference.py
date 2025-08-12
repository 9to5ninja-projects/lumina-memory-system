"""
Tests for HRR (Holographic Reduced Representations) reference vectors.

These tests ensure that:
1. Reference vectors are deterministic
2. Dict key ordering doesn't affect results
3. Different inputs produce different vectors
4. Vector operations work correctly
"""

import pytest
import numpy as np
from lumina_memory.hrr import reference_vector

def test_reference_is_deterministic():
    """Same inputs should produce same reference vector."""
    r1 = reference_vector("abc", {"topic": "x", "src": "s"}, dim=256)
    r2 = reference_vector("abc", {"src": "s", "topic": "x"}, dim=256)
    assert np.allclose(r1, r2)

def test_different_content_different_vector():
    """Different content should produce different reference vectors."""
    r1 = reference_vector("abc", {"topic": "x"}, dim=256)
    r2 = reference_vector("def", {"topic": "x"}, dim=256)
    
    # Vectors should be different
    assert not np.allclose(r1, r2)
    
    # But both should be unit vectors
    assert np.isclose(np.linalg.norm(r1), 1.0)
    assert np.isclose(np.linalg.norm(r2), 1.0)

def test_different_metadata_different_vector():
    """Different metadata should produce different reference vectors."""
    r1 = reference_vector("abc", {"topic": "x"}, dim=256)
    r2 = reference_vector("abc", {"topic": "y"}, dim=256)
    
    assert not np.allclose(r1, r2)

def test_vector_dimension_respected():
    """Vector should have requested dimension."""
    for dim in [64, 128, 256, 512, 1024]:
        r = reference_vector("test", {"key": "val"}, dim=dim)
        assert r.shape == (dim,)
        assert np.isclose(np.linalg.norm(r), 1.0)  # Unit vector

def test_empty_metadata_works():
    """Empty metadata should work."""
    r = reference_vector("test content", {}, dim=128)
    assert r.shape == (128,)
    assert np.isclose(np.linalg.norm(r), 1.0)

def test_none_content_works():
    """None or empty content should work."""
    r1 = reference_vector("", {"key": "val"}, dim=128)
    r2 = reference_vector(None, {"key": "val"}, dim=128)
    
    assert r1.shape == (128,)
    assert r2.shape == (128,)
    assert np.isclose(np.linalg.norm(r1), 1.0)
    assert np.isclose(np.linalg.norm(r2), 1.0)

def test_complex_metadata_deterministic():
    """Complex nested metadata should be deterministic."""
    metadata1 = {
        "source": "web", 
        "tags": ["ai", "memory"],
        "nested": {"level": 2, "values": [1, 2, 3]},
        "timestamp": 1234567890
    }
    metadata2 = {
        "timestamp": 1234567890,
        "nested": {"values": [1, 2, 3], "level": 2},
        "tags": ["ai", "memory"],
        "source": "web"
    }
    
    r1 = reference_vector("content", metadata1, dim=256)
    r2 = reference_vector("content", metadata2, dim=256)
    
    assert np.allclose(r1, r2)

def test_vector_binding_operation():
    """Test HRR circular convolution binding."""
    # This might be in a separate function, but test the concept
    r1 = reference_vector("concept1", {"type": "entity"}, dim=128)
    r2 = reference_vector("concept2", {"type": "relation"}, dim=128)
    
    # Binding should produce another unit vector
    bound = np.fft.irfft(np.fft.rfft(r1) * np.fft.rfft(r2))
    
    assert bound.shape == (128,)
    # Note: bound vector is not necessarily unit length after binding
    # but should be roughly the same magnitude
    assert 0.5 < np.linalg.norm(bound) < 2.0

def test_unicode_content_works():
    """Unicode content should work consistently."""
    content = "Hello 世界 "
    r1 = reference_vector(content, {"lang": "mixed"}, dim=256)
    r2 = reference_vector(content, {"lang": "mixed"}, dim=256)
    
    assert np.allclose(r1, r2)
    assert np.isclose(np.linalg.norm(r1), 1.0)

def test_large_content_works():
    """Large content should work efficiently."""
    large_content = "word " * 1000  # ~5KB of text
    metadata = {"size": "large", "type": "document"}
    
    r = reference_vector(large_content, metadata, dim=512)
    
    assert r.shape == (512,)
    assert np.isclose(np.linalg.norm(r), 1.0)

def test_numerical_stability():
    """Vector generation should be numerically stable."""
    content = "test content"
    metadata = {"key": "value"}
    
    # Generate same vector multiple times
    vectors = [reference_vector(content, metadata, dim=256) for _ in range(5)]
    
    # All should be identical
    for i in range(1, len(vectors)):
        assert np.allclose(vectors[0], vectors[i])
        assert np.isclose(np.linalg.norm(vectors[i]), 1.0)
