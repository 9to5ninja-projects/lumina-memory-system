"""
Tests for Event Hashing and Hash Chain Verification

Test suite for event_hashing.py module covering:
- Event hash computation and determinism
- Hash chain creation and linking
- Chain integrity verification
- Merkle tree construction and proof verification
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume
import sys
import os
import time

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from lumina_memory.event_hashing import (
    HashChainEntry,
    compute_event_hash,
    create_genesis_entry,
    add_event_to_chain,
    verify_hash_chain,
    MerkleTree,
    HashChainManager,
    global_hash_chain
)


class TestEventHashing:
    """Test event hash computation."""
    
    def test_compute_event_hash_deterministic(self):
        """Test that same event produces same hash."""
        event_data = {
            'event_type': 'ingest',
            'timestamp': 1692000000.0,
            'sequence': 1,
            'payload': {'memory_id': 'test123'},
            'metadata': {'source': 'test'},
            'schema_version': 'v1.0'
        }
        
        hash1 = compute_event_hash(event_data)
        hash2 = compute_event_hash(event_data)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex length
        assert isinstance(hash1, str)
    
    def test_compute_event_hash_different_events(self):
        """Test that different events produce different hashes."""
        event1 = {
            'event_type': 'ingest',
            'timestamp': 1692000000.0,
            'sequence': 1,
            'payload': {'memory_id': 'test123'},
            'metadata': {},
            'schema_version': 'v1.0'
        }
        
        event2 = {
            'event_type': 'recall',
            'timestamp': 1692000000.0,
            'sequence': 1,
            'payload': {'memory_id': 'test123'},
            'metadata': {},
            'schema_version': 'v1.0'
        }
        
        hash1 = compute_event_hash(event1)
        hash2 = compute_event_hash(event2)
        
        assert hash1 != hash2
    
    def test_compute_event_hash_missing_fields(self):
        """Test hash computation with missing optional fields."""
        event_data = {
            'event_type': 'test',
            'timestamp': time.time(),
            'sequence': 0
            # payload, metadata, schema_version missing
        }
        
        event_hash = compute_event_hash(event_data)
        assert isinstance(event_hash, str)
        assert len(event_hash) == 64


class TestHashChainEntry:
    """Test hash chain entry operations."""
    
    def test_hash_chain_entry_creation(self):
        """Test creating hash chain entry."""
        entry = HashChainEntry(
            event_id="test123",
            event_hash="a" * 64,
            previous_hash="b" * 64,
            chain_hash="c" * 64,
            sequence_number=5,
            timestamp=1692000000.0
        )
        
        assert entry.event_id == "test123"
        assert entry.sequence_number == 5
        assert isinstance(entry, HashChainEntry)
    
    def test_compute_chain_hash(self):
        """Test chain hash computation."""
        entry = HashChainEntry(
            event_id="test123",
            event_hash="a" * 64,
            previous_hash="b" * 64,
            chain_hash="",  # Will be computed
            sequence_number=5,
            timestamp=1692000000.0
        )
        
        computed_hash = entry.compute_chain_hash()
        assert isinstance(computed_hash, str)
        assert len(computed_hash) == 64
        
        # Same computation should yield same result
        computed_hash2 = entry.compute_chain_hash()
        assert computed_hash == computed_hash2
    
    def test_verify_integrity(self):
        """Test entry integrity verification."""
        entry = HashChainEntry(
            event_id="test123",
            event_hash="a" * 64,
            previous_hash="b" * 64,
            chain_hash="",
            sequence_number=5,
            timestamp=1692000000.0
        )
        
        # Compute correct chain hash
        correct_hash = entry.compute_chain_hash()
        valid_entry = HashChainEntry(
            event_id=entry.event_id,
            event_hash=entry.event_hash,
            previous_hash=entry.previous_hash,
            chain_hash=correct_hash,
            sequence_number=entry.sequence_number,
            timestamp=entry.timestamp
        )
        
        assert valid_entry.verify_integrity()
        
        # Test with wrong hash
        invalid_entry = HashChainEntry(
            event_id=entry.event_id,
            event_hash=entry.event_hash,
            previous_hash=entry.previous_hash,
            chain_hash="wrong_hash",
            sequence_number=entry.sequence_number,
            timestamp=entry.timestamp
        )
        
        assert not invalid_entry.verify_integrity()
    
    def test_verify_chain_link_genesis(self):
        """Test chain link verification for genesis entry."""
        genesis = HashChainEntry(
            event_id="genesis",
            event_hash="a" * 64,
            previous_hash="0" * 64,
            chain_hash="b" * 64,
            sequence_number=0,
            timestamp=1692000000.0
        )
        
        assert genesis.verify_chain_link(None)
        
        # Invalid genesis (wrong previous hash)
        invalid_genesis = HashChainEntry(
            event_id="genesis",
            event_hash="a" * 64,
            previous_hash="1" * 64,  # Should be zeros
            chain_hash="b" * 64,
            sequence_number=0,
            timestamp=1692000000.0
        )
        
        assert not invalid_genesis.verify_chain_link(None)
    
    def test_verify_chain_link_regular(self):
        """Test chain link verification for regular entries."""
        previous = HashChainEntry(
            event_id="prev",
            event_hash="a" * 64,
            previous_hash="0" * 64,
            chain_hash="b" * 64,
            sequence_number=0,
            timestamp=1692000000.0
        )
        
        # Valid next entry
        current = HashChainEntry(
            event_id="current",
            event_hash="c" * 64,
            previous_hash="b" * 64,  # Should match previous chain_hash
            chain_hash="d" * 64,
            sequence_number=1,  # Should be previous + 1
            timestamp=1692000001.0
        )
        
        assert current.verify_chain_link(previous)
        
        # Invalid - wrong previous hash
        invalid_current = HashChainEntry(
            event_id="current",
            event_hash="c" * 64,
            previous_hash="wrong" * 16,
            chain_hash="d" * 64,
            sequence_number=1,
            timestamp=1692000001.0
        )
        
        assert not invalid_current.verify_chain_link(previous)


class TestHashChainOperations:
    """Test hash chain creation and verification."""
    
    def test_create_genesis_entry(self):
        """Test genesis entry creation."""
        genesis = create_genesis_entry()
        
        assert genesis.event_id == "genesis"
        assert genesis.sequence_number == 0
        assert genesis.previous_hash == "0" * 64
        assert genesis.verify_integrity()
        assert genesis.verify_chain_link(None)
    
    def test_add_event_to_chain(self):
        """Test adding event to chain."""
        genesis = create_genesis_entry()
        
        event_data = {
            'event_id': 'event1',
            'event_type': 'test',
            'timestamp': 1692000001.0,
            'sequence': 1,
            'payload': {'data': 'test'},
            'metadata': {}
        }
        
        new_entry = add_event_to_chain(event_data, genesis)
        
        assert new_entry.event_id == 'event1'
        assert new_entry.sequence_number == 1
        assert new_entry.previous_hash == genesis.chain_hash
        assert new_entry.verify_integrity()
        assert new_entry.verify_chain_link(genesis)
    
    def test_verify_hash_chain_valid(self):
        """Test verification of valid hash chain."""
        # Create chain with genesis + 2 events
        genesis = create_genesis_entry()
        
        event1_data = {
            'event_id': 'event1',
            'event_type': 'test',
            'timestamp': 1692000001.0,
            'sequence': 1,
            'payload': {'data': 'test1'},
            'metadata': {}
        }
        
        event2_data = {
            'event_id': 'event2',
            'event_type': 'test',
            'timestamp': 1692000002.0,
            'sequence': 2,
            'payload': {'data': 'test2'},
            'metadata': {}
        }
        
        entry1 = add_event_to_chain(event1_data, genesis)
        entry2 = add_event_to_chain(event2_data, entry1)
        
        chain = [genesis, entry1, entry2]
        is_valid, error = verify_hash_chain(chain)
        
        assert is_valid
        assert error is None
    
    def test_verify_hash_chain_empty(self):
        """Test verification of empty chain."""
        is_valid, error = verify_hash_chain([])
        
        assert not is_valid
        assert "Empty chain" in error
    
    def test_verify_hash_chain_corrupted(self):
        """Test verification of corrupted chain."""
        genesis = create_genesis_entry()
        
        # Create corrupted entry with wrong sequence number
        corrupted_entry = HashChainEntry(
            event_id="corrupted",
            event_hash="a" * 64,
            previous_hash=genesis.chain_hash,
            chain_hash="b" * 64,
            sequence_number=5,  # Should be 1
            timestamp=1692000001.0
        )
        
        chain = [genesis, corrupted_entry]
        is_valid, error = verify_hash_chain(chain)
        
        assert not is_valid
        assert "chain link verification failed" in error


class TestMerkleTree:
    """Test Merkle tree operations."""
    
    def test_merkle_tree_creation(self):
        """Test Merkle tree creation."""
        hashes = ["a" * 64, "b" * 64, "c" * 64, "d" * 64]
        tree = MerkleTree(hashes)
        
        assert tree.root_hash is not None
        assert len(tree.root_hash) == 64
        assert tree.leaves == hashes
    
    def test_merkle_tree_empty_error(self):
        """Test error on empty hash list."""
        with pytest.raises(ValueError, match="empty list"):
            MerkleTree([])
    
    def test_merkle_tree_single_hash(self):
        """Test Merkle tree with single hash."""
        hashes = ["a" * 64]
        tree = MerkleTree(hashes)
        
        assert tree.root_hash == hashes[0]
    
    def test_merkle_proof_generation(self):
        """Test Merkle proof generation."""
        hashes = ["a" * 64, "b" * 64, "c" * 64, "d" * 64]
        tree = MerkleTree(hashes)
        
        # Get proof for first hash
        proof = tree.get_proof(hashes[0])
        
        assert proof is not None
        assert isinstance(proof, list)
        assert len(proof) >= 1  # Should have at least one sibling
    
    def test_merkle_proof_verification(self):
        """Test Merkle proof verification."""
        hashes = ["a" * 64, "b" * 64, "c" * 64, "d" * 64]
        tree = MerkleTree(hashes)
        
        # Test proof for each hash
        for hash_val in hashes:
            proof = tree.get_proof(hash_val)
            is_valid = tree.verify_proof(hash_val, proof)
            assert is_valid
    
    def test_merkle_proof_invalid_hash(self):
        """Test proof generation for non-existent hash."""
        hashes = ["a" * 64, "b" * 64, "c" * 64]
        tree = MerkleTree(hashes)
        
        proof = tree.get_proof("nonexistent" + "0" * 48)
        assert proof is None
    
    def test_merkle_proof_verification_invalid(self):
        """Test verification with invalid proof."""
        hashes = ["a" * 64, "b" * 64, "c" * 64, "d" * 64]
        tree = MerkleTree(hashes)
        
        # Create invalid proof
        invalid_proof = ["wrong" + "0" * 59]
        is_valid = tree.verify_proof(hashes[0], invalid_proof)
        assert not is_valid


class TestHashChainManager:
    """Test hash chain manager."""
    
    def test_hash_chain_manager_init(self):
        """Test hash chain manager initialization."""
        manager = HashChainManager()
        
        assert len(manager.chain) == 1  # Genesis entry
        assert manager.chain[0].event_id == "genesis"
        assert manager.chain[0].sequence_number == 0
    
    def test_add_event(self):
        """Test adding event to managed chain."""
        manager = HashChainManager()
        
        event_data = {
            'event_id': 'test_event',
            'event_type': 'test',
            'timestamp': 1692000001.0,
            'payload': {'data': 'test'},
            'metadata': {}
        }
        
        new_entry = manager.add_event(event_data)
        
        assert new_entry.event_id == 'test_event'
        assert len(manager.chain) == 2
        assert manager.chain[-1] == new_entry
    
    def test_verify_integrity(self):
        """Test chain integrity verification."""
        manager = HashChainManager()
        
        # Add some events
        for i in range(3):
            event_data = {
                'event_id': f'event_{i}',
                'event_type': 'test',
                'timestamp': 1692000000.0 + i,
                'payload': {'data': f'test{i}'},
                'metadata': {}
            }
            manager.add_event(event_data)
        
        is_valid, error = manager.verify_integrity()
        assert is_valid
        assert error is None
    
    def test_get_chain_head(self):
        """Test getting chain head."""
        manager = HashChainManager()
        
        head = manager.get_chain_head()
        assert head.event_id == "genesis"
        
        # Add event and check new head
        event_data = {
            'event_id': 'new_event',
            'event_type': 'test',
            'timestamp': 1692000001.0,
            'payload': {},
            'metadata': {}
        }
        manager.add_event(event_data)
        
        new_head = manager.get_chain_head()
        assert new_head.event_id == 'new_event'
    
    def test_get_merkle_tree(self):
        """Test Merkle tree generation from chain."""
        manager = HashChainManager()
        
        # Add some events
        for i in range(3):
            event_data = {
                'event_id': f'event_{i}',
                'event_type': 'test',
                'timestamp': 1692000000.0 + i,
                'payload': {'data': f'test{i}'},
                'metadata': {}
            }
            manager.add_event(event_data)
        
        merkle_tree = manager.get_merkle_tree()
        assert merkle_tree.root_hash is not None
        assert len(merkle_tree.leaves) == len(manager.chain)
    
    def test_verify_event_inclusion(self):
        """Test event inclusion verification."""
        manager = HashChainManager()
        
        event_data = {
            'event_id': 'test_event',
            'event_type': 'test',
            'timestamp': 1692000001.0,
            'payload': {'data': 'test'},
            'metadata': {}
        }
        
        new_entry = manager.add_event(event_data)
        
        # Event should be included
        assert manager.verify_event_inclusion(new_entry.event_hash)
        
        # Random hash should not be included
        assert not manager.verify_event_inclusion("random" + "0" * 58)
    
    def test_get_chain_stats(self):
        """Test chain statistics."""
        manager = HashChainManager()
        
        stats = manager.get_chain_stats()
        assert stats['total_events'] == 1  # Just genesis
        assert stats['chain_length'] == 0  # Exclude genesis
        assert stats['integrity_valid'] is True
        
        # Add event and check stats
        event_data = {
            'event_id': 'test_event',
            'event_type': 'test',
            'timestamp': 1692000001.0,
            'payload': {},
            'metadata': {}
        }
        manager.add_event(event_data)
        
        new_stats = manager.get_chain_stats()
        assert new_stats['total_events'] == 2
        assert new_stats['chain_length'] == 1


# Property-based tests
@given(st.dictionaries(
    st.sampled_from(['event_type', 'timestamp', 'sequence', 'payload']),
    st.one_of(st.text(), st.floats(allow_nan=False, allow_infinity=False), st.integers(), st.dictionaries(st.text(), st.text()))
))
def test_event_hash_deterministic_property(event_data):
    """Property test: Same event data should produce same hash."""
    hash1 = compute_event_hash(event_data)
    hash2 = compute_event_hash(event_data)
    
    assert hash1 == hash2
    assert len(hash1) == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
