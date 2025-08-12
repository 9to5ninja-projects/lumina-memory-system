"""
Tests for Content-Addressable Cryptographic IDs

Test suite for crypto_ids.py module covering:
- Content hash determinism and collision resistance
- Memory content ID generation and verification
- Content-addressable index operations
- Integrity verification and conflict detection
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from lumina_memory.crypto_ids import (
    normalize_content,
    compute_content_hash,
    memory_content_id,
    composite_memory_id,
    event_content_id,
    verify_content_integrity,
    detect_content_collision,
    ContentAddressableIndex,
    global_content_index
)


class TestContentNormalization:
    """Test content normalization for consistent hashing."""
    
    def test_normalize_string(self):
        """Test string normalization."""
        assert normalize_content("hello") == "hello"
        assert normalize_content("  hello  ") == "hello"
        assert normalize_content("") == ""
    
    def test_normalize_numbers(self):
        """Test number normalization."""
        assert normalize_content(42) == "42"
        assert normalize_content(3.14) == "3.14"
        assert normalize_content(True) == "True"
        assert normalize_content(False) == "False"
    
    def test_normalize_collections(self):
        """Test collection normalization."""
        # Lists should be sorted if all elements are comparable
        assert normalize_content([3, 1, 2]) == "[1,2,3]"
        assert normalize_content((3, 1, 2)) == "[1,2,3]"
        
        # Dictionaries should have sorted keys
        result = normalize_content({"b": 2, "a": 1})
        expected = '{"a":1,"b":2}'
        assert result == expected
    
    def test_normalize_numpy_array(self):
        """Test NumPy array normalization."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = normalize_content(arr)
        
        # Should be deterministic hex representation
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Same array should produce same result
        arr2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert normalize_content(arr2) == result


class TestContentHashing:
    """Test content hashing functions."""
    
    def test_compute_content_hash_deterministic(self):
        """Test that same content produces same hash."""
        content = "test content"
        embedding = np.array([0.1, 0.2, 0.3])
        
        hash1 = compute_content_hash(content, embedding)
        hash2 = compute_content_hash(content, embedding)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex length
    
    def test_compute_content_hash_different_content(self):
        """Test that different content produces different hashes."""
        embedding = np.array([0.1, 0.2, 0.3])
        
        hash1 = compute_content_hash("content1", embedding)
        hash2 = compute_content_hash("content2", embedding)
        
        assert hash1 != hash2
    
    def test_compute_content_hash_with_metadata(self):
        """Test hashing with metadata."""
        content = "test"
        embedding = np.array([0.1, 0.2])
        metadata = {"key": "value"}
        
        hash_with_meta = compute_content_hash(content, embedding, metadata)
        hash_without_meta = compute_content_hash(content, embedding)
        
        assert hash_with_meta != hash_without_meta
    
    def test_memory_content_id(self):
        """Test memory content ID generation."""
        content = "memory content"
        embedding = np.array([1.0, 2.0, 3.0])
        
        memory_id = memory_content_id(content, embedding)
        
        assert isinstance(memory_id, str)
        assert len(memory_id) == 64
        
        # Same inputs should produce same ID
        memory_id2 = memory_content_id(content, embedding)
        assert memory_id == memory_id2
    
    def test_composite_memory_id(self):
        """Test composite memory ID generation."""
        memory_ids = ["id1", "id2", "id3"]
        
        composite_id = composite_memory_id(memory_ids)
        
        assert isinstance(composite_id, str)
        assert len(composite_id) == 64
        
        # Order should not matter (IDs are sorted internally)
        composite_id2 = composite_memory_id(["id3", "id1", "id2"])
        assert composite_id == composite_id2
    
    def test_event_content_id(self):
        """Test event content ID generation."""
        event_data = {
            'event_type': 'ingest',
            'timestamp': 1692000000.0,
            'sequence': 42,
            'payload': {'memory_id': 'test'},
            'schema_version': 'v1.0'
        }
        
        event_id = event_content_id(event_data)
        
        assert isinstance(event_id, str)
        assert len(event_id) == 64
        
        # Same event data should produce same ID
        event_id2 = event_content_id(event_data)
        assert event_id == event_id2


class TestIntegrityVerification:
    """Test integrity verification functions."""
    
    def test_verify_content_integrity_valid(self):
        """Test verification of valid content."""
        content = "test content"
        embedding = np.array([1.0, 2.0])
        
        # Generate ID and verify it
        content_id = memory_content_id(content, embedding)
        is_valid = verify_content_integrity(content_id, content, embedding)
        
        assert is_valid
    
    def test_verify_content_integrity_invalid(self):
        """Test verification of invalid content."""
        content = "test content"
        embedding = np.array([1.0, 2.0])
        
        # Generate ID for different content
        content_id = memory_content_id("different content", embedding)
        is_valid = verify_content_integrity(content_id, content, embedding)
        
        assert not is_valid
    
    def test_detect_content_collision_none(self):
        """Test collision detection with identical content."""
        content = "same content"
        embedding = np.array([1.0, 2.0])
        
        existing_content = {
            'content': content,
            'embedding': embedding,
            'metadata': {}
        }
        
        new_content = {
            'content': content,
            'embedding': embedding,
            'metadata': {}
        }
        
        content_id = memory_content_id(content, embedding)
        collision = detect_content_collision(content_id, existing_content, new_content)
        
        assert collision is None  # No collision for identical content
    
    def test_detect_content_collision_invalid_id(self):
        """Test collision detection with invalid ID."""
        content = "test content"
        embedding = np.array([1.0, 2.0])
        
        existing_content = {
            'content': content,
            'embedding': embedding,
            'metadata': {}
        }
        
        new_content = {
            'content': content,
            'embedding': embedding,
            'metadata': {}
        }
        
        # Use wrong content ID
        wrong_id = "0" * 64
        collision = detect_content_collision(wrong_id, existing_content, new_content)
        
        assert collision is not None
        assert collision['type'] == 'invalid_id'


class TestContentAddressableIndex:
    """Test content-addressable index operations."""
    
    def test_add_memory_new(self):
        """Test adding new memory to index."""
        index = ContentAddressableIndex()
        
        content = "test memory"
        embedding = np.array([1.0, 2.0, 3.0])
        
        content_id, is_duplicate = index.add_memory(content, embedding)
        
        assert isinstance(content_id, str)
        assert len(content_id) == 64
        assert not is_duplicate
        assert content_id in index.index
    
    def test_add_memory_duplicate(self):
        """Test adding duplicate memory to index."""
        index = ContentAddressableIndex()
        
        content = "test memory"
        embedding = np.array([1.0, 2.0, 3.0])
        
        # Add memory first time
        content_id1, is_dup1 = index.add_memory(content, embedding)
        assert not is_dup1
        
        # Add same memory again
        content_id2, is_dup2 = index.add_memory(content, embedding)
        assert is_dup2
        assert content_id1 == content_id2
        assert index.access_count[content_id1] == 2
    
    def test_get_memory(self):
        """Test retrieving memory from index."""
        index = ContentAddressableIndex()
        
        content = "test memory"
        embedding = np.array([1.0, 2.0, 3.0])
        
        content_id, _ = index.add_memory(content, embedding)
        retrieved = index.get_memory(content_id)
        
        assert retrieved is not None
        assert retrieved['content'] == content
        assert np.array_equal(retrieved['embedding'], embedding)
    
    def test_verify_index_integrity(self):
        """Test index integrity verification."""
        index = ContentAddressableIndex()
        
        # Add valid memory
        content = "test memory"
        embedding = np.array([1.0, 2.0, 3.0])
        content_id, _ = index.add_memory(content, embedding)
        
        # Verify integrity
        violations = index.verify_index_integrity()
        assert len(violations) == 0
        
        # Corrupt the index by modifying content
        index.index[content_id]['content'] = "corrupted content"
        
        # Check for violations
        violations = index.verify_index_integrity()
        assert len(violations) == 1
        assert violations[0] == content_id
    
    def test_get_stats(self):
        """Test index statistics."""
        index = ContentAddressableIndex()
        
        # Initially empty
        stats = index.get_stats()
        assert stats['total_unique_memories'] == 0
        assert stats['total_access_count'] == 0
        
        # Add memory
        content = "test memory"
        embedding = np.array([1.0, 2.0, 3.0])
        index.add_memory(content, embedding)
        
        # Check stats
        stats = index.get_stats()
        assert stats['total_unique_memories'] == 1
        assert stats['total_access_count'] == 1


# Property-based tests
@given(st.text(), st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=100))
def test_memory_id_deterministic(content, embedding_list):
    """Property test: Same content should always produce same ID."""
    assume(len(content) > 0)  # Non-empty content
    
    embedding = np.array(embedding_list, dtype=np.float32)
    
    id1 = memory_content_id(content, embedding)
    id2 = memory_content_id(content, embedding)
    
    assert id1 == id2
    assert len(id1) == 64


@given(st.text(), st.text(), st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=100))
def test_different_content_different_ids(content1, content2, embedding_list):
    """Property test: Different content should produce different IDs."""
    assume(len(content1) > 0 and len(content2) > 0)
    assume(content1 != content2)
    
    embedding = np.array(embedding_list, dtype=np.float32)
    
    id1 = memory_content_id(content1, embedding)
    id2 = memory_content_id(content2, embedding)
    
    assert id1 != id2


@given(st.lists(st.text(min_size=1), min_size=2, max_size=10, unique=True))
def test_composite_id_order_invariant(memory_ids):
    """Property test: Composite ID should be invariant to input order."""
    id1 = composite_memory_id(memory_ids)
    id2 = composite_memory_id(memory_ids[::-1])  # Reverse order
    
    assert id1 == id2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
