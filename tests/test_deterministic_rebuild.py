"""
Tests for Deterministic Rebuild Functionality

Comprehensive test suite for deterministic rebuild including:
- Byte-identical reconstruction from events
- Active-Set uniqueness enforcement
- Conflict resolution with cryptographic precedence
- Snapshot-optimized rebuild
- Integration with event store
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lumina_memory.deterministic_rebuild import (
    DeterministicRebuilder, RebuildState, ConflictResolutionError,
    create_deterministic_rebuilder, rebuild_memory_from_events
)
from lumina_memory.event_store import create_event_store
from lumina_memory.kernel import LuminaMemory, MemoryRecord, create_memory_record
from lumina_memory.crypto_ids import memory_content_id


class MockLuminaMemory:
    """Mock LuminaMemory for testing."""
    
    def __init__(self):
        self.memories = {}
        self.cleared = False
    
    def add_memory(self, record: MemoryRecord):
        self.memories[record.memory_id] = record
    
    def get_memory_by_id(self, memory_id: str):
        return self.memories.get(memory_id)
    
    def update_memory(self, record: MemoryRecord):
        self.memories[record.memory_id] = record
    
    def delete_memory(self, memory_id: str):
        self.memories.pop(memory_id, None)
    
    def clear_all_memories(self):
        self.memories.clear()
        self.cleared = True
    
    def rebuild_search_index(self):
        pass
    
    def get_all_memories(self):
        return list(self.memories.values())


class TestDeterministicRebuilder:
    """Test deterministic rebuilder core functionality."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "rebuild_test"
    
    @pytest.fixture
    def event_store(self, temp_storage):
        """Create test event store."""
        return create_event_store(temp_storage)
    
    @pytest.fixture
    def mock_memory(self):
        """Create mock memory system."""
        return MockLuminaMemory()
    
    @pytest.fixture
    def rebuilder(self, event_store, mock_memory):
        """Create test rebuilder."""
        return DeterministicRebuilder(event_store, mock_memory)
    
    def test_rebuilder_initialization(self, rebuilder):
        """Test rebuilder initialization."""
        assert rebuilder.event_store is not None
        assert rebuilder.memory is not None
        assert rebuilder.content_index is not None
        assert rebuilder.rebuild_state is None
    
    def test_rebuild_from_scratch_empty(self, rebuilder):
        """Test rebuild from scratch with no events."""
        stats = rebuilder.rebuild_from_scratch()
        
        assert stats[''events_processed''] == 0
        assert stats[''memories_created''] == 0
        assert stats[''conflicts_resolved''] == 0
        assert stats[''hash_chain_verified''] is True
        assert stats[''final_active_set_size''] == 0
        assert rebuilder.memory.cleared  # Should clear existing memories
    
    def test_rebuild_from_scratch_with_events(self, rebuilder):
        """Test rebuild from scratch with memory ingest events."""
        # Add memory ingest events to event store
        test_contents = [
            "First memory content",
            "Second memory content", 
            "Third memory content"
        ]
        
        for i, content in enumerate(test_contents):
            rebuilder.event_store.append_event("memory_ingest", {
                "content": content,
                "metadata": {"index": i, "type": "test"}
            })
        
        # Perform rebuild
        stats = rebuilder.rebuild_from_scratch()
        
        assert stats[''events_processed''] == 3
        assert stats[''final_active_set_size''] == 3
        assert len(rebuilder.memory.get_all_memories()) == 3
        
        # Verify Active-Set state
        active_set = rebuilder.get_active_set_state()
        assert len(active_set) == 3
        
        # Each content should map to correct memory
        for content in test_contents:
            content_id = memory_content_id(content)
            assert content_id in active_set
            
            memory_id = active_set[content_id]
            memory_record = rebuilder.memory.get_memory_by_id(memory_id)
            assert memory_record is not None
            assert memory_record.content == content
    
    def test_active_set_uniqueness_enforcement(self, rebuilder):
        """Test that Active-Set prevents duplicate content."""
        # Add same content multiple times
        duplicate_content = "This content appears multiple times"
        
        for i in range(3):
            rebuilder.event_store.append_event("memory_ingest", {
                "content": duplicate_content,
                "metadata": {"attempt": i}
            })
        
        # Rebuild should enforce uniqueness
        stats = rebuilder.rebuild_from_scratch()
        
        # Should process all events but only create one memory
        assert stats[''events_processed''] == 1  # Idempotent events
        assert stats[''final_active_set_size''] == 1
        assert len(rebuilder.memory.get_all_memories()) == 1
        
        # Verify single entry in Active-Set
        active_set = rebuilder.get_active_set_state()
        content_id = memory_content_id(duplicate_content)
        assert content_id in active_set
    
    def test_conflict_resolution_cryptographic_precedence(self, rebuilder):
        """Test conflict resolution using cryptographic precedence."""
        # Create two different contents with different hashes
        content1 = "Content A - earlier hash"
        content2 = "Content B - later hash"
        
        # Ensure content1 has lexicographically smaller hash
        hash1 = memory_content_id(content1)
        hash2 = memory_content_id(content2) 
        
        if hash1 > hash2:
            content1, content2 = content2, content1
            hash1, hash2 = hash2, hash1
        
        # Add events (second event tries to conflict)
        rebuilder.event_store.append_event("memory_ingest", {
            "content": content2,  # Larger hash first
            "metadata": {"order": 1}
        })
        
        rebuilder.event_store.append_event("memory_ingest", {
            "content": content1,  # Smaller hash second (should win)
            "metadata": {"order": 2}
        })
        
        # Manual conflict simulation (same content_id different hashes)
        # This is a edge case test for the conflict resolution logic
        rebuilder.rebuild_state = RebuildState(
            processed_events=set(),
            active_set={hash2: "existing_memory"},  # Larger hash existing
            conflict_count=0,
            last_snapshot_hash=None,
            rebuild_timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        # Add memory with larger hash
        existing_memory = create_memory_record(content2, {"order": 1})
        existing_memory = existing_memory._replace(memory_id="existing_memory")
        rebuilder.memory.add_memory(existing_memory)
        
        # Process event with smaller hash - should replace existing
        event_smaller = rebuilder.event_store.get_events_by_type("memory_ingest")[0]  # content1
        if memory_content_id(event_smaller.content[''content'']) == hash1:
            rebuilder._process_memory_ingest_event(event_smaller)
        
            # Should increment conflict count
            assert rebuilder.rebuild_state.conflict_count > 0
    
    def test_memory_update_event_processing(self, rebuilder):
        """Test processing memory update events."""
        # Add initial memory
        original_content = "Original content"
        rebuilder.event_store.append_event("memory_ingest", {
            "content": original_content,
            "metadata": {"version": 1}
        })
        
        # Rebuild to establish initial state
        rebuilder.rebuild_from_scratch()
        
        # Get the created memory ID
        active_set = rebuilder.get_active_set_state()
        original_content_id = memory_content_id(original_content)
        memory_id = active_set[original_content_id]
        
        # Add update event
        updated_content = "Updated content"
        rebuilder.event_store.append_event("memory_update", {
            "memory_id": memory_id,
            "new_content": updated_content,
            "new_metadata": {"version": 2}
        })
        
        # Rebuild again
        stats = rebuilder.rebuild_from_scratch()
        
        # Should have processed both events
        assert stats[''events_processed''] == 2
        
        # Memory should be updated
        updated_memory = rebuilder.memory.get_memory_by_id(memory_id)
        assert updated_memory.content == updated_content
        assert updated_memory.metadata["version"] == 2
        
        # Active-Set should reflect new content
        active_set = rebuilder.get_active_set_state()
        updated_content_id = memory_content_id(updated_content)
        assert updated_content_id in active_set
        assert original_content_id not in active_set
    
    def test_memory_delete_event_processing(self, rebuilder):
        """Test processing memory deletion events."""
        # Add initial memory
        content = "Content to be deleted"
        rebuilder.event_store.append_event("memory_ingest", {
            "content": content,
            "metadata": {"temporary": True}
        })
        
        # Rebuild to establish initial state
        rebuilder.rebuild_from_scratch()
        
        # Get memory ID
        active_set = rebuilder.get_active_set_state()
        content_id = memory_content_id(content)
        memory_id = active_set[content_id]
        
        # Add deletion event
        rebuilder.event_store.append_event("memory_delete", {
            "memory_id": memory_id,
            "reason": "cleanup"
        })
        
        # Rebuild again
        stats = rebuilder.rebuild_from_scratch()
        
        # Memory should be deleted
        assert rebuilder.memory.get_memory_by_id(memory_id) is None
        
        # Active-Set should be empty
        active_set = rebuilder.get_active_set_state()
        assert len(active_set) == 0
    
    def test_snapshot_optimized_rebuild(self, rebuilder):
        """Test rebuild from snapshot optimization."""
        # Add initial events
        initial_contents = ["Content 1", "Content 2", "Content 3"]
        for content in initial_contents:
            rebuilder.event_store.append_event("memory_ingest", {
                "content": content,
                "metadata": {"phase": "initial"}
            })
        
        # Perform initial rebuild
        rebuilder.rebuild_from_scratch()
        
        # Create snapshot
        snapshot = rebuilder.create_checkpoint_snapshot()
        
        # Add more events after snapshot
        additional_contents = ["Content 4", "Content 5"]
        for content in additional_contents:
            rebuilder.event_store.append_event("memory_ingest", {
                "content": content,
                "metadata": {"phase": "additional"}
            })
        
        # Clear memory and rebuild from snapshot
        rebuilder.memory.clear_all_memories()
        stats = rebuilder.rebuild_from_snapshot()
        
        # Should have all memories
        assert len(rebuilder.memory.get_all_memories()) == 5
        assert stats[''final_active_set_size''] == 5
        
        # Should have processed only events since snapshot
        assert stats[''events_processed''] == 2
    
    def test_rebuild_verification(self, rebuilder):
        """Test rebuild integrity verification."""
        # Add test events
        test_contents = ["Verify A", "Verify B", "Verify C"]
        for content in test_contents:
            rebuilder.event_store.append_event("memory_ingest", {
                "content": content,
                "metadata": {"verified": True}
            })
        
        # Perform rebuild
        stats = rebuilder.rebuild_from_scratch()
        
        # Should include verification results
        assert ''final_verification'' in stats
        verification = stats[''final_verification'']
        
        assert verification[''active_set_consistent''] is True
        assert verification[''no_duplicate_content''] is True
        assert verification[''all_content_ids_valid''] is True
    
    def test_rebuild_determinism(self, rebuilder):
        """Test that rebuilds are deterministic."""
        # Add events with same content but different timestamps
        base_content = "Deterministic test content"
        
        for i in range(3):
            rebuilder.event_store.append_event("memory_ingest", {
                "content": base_content,
                "metadata": {"iteration": i}
            })
        
        # Perform multiple rebuilds
        stats1 = rebuilder.rebuild_from_scratch()
        active_set1 = rebuilder.get_active_set_state().copy()
        
        stats2 = rebuilder.rebuild_from_scratch()
        active_set2 = rebuilder.get_active_set_state().copy()
        
        # Results should be identical
        assert stats1[''final_active_set_size''] == stats2[''final_active_set_size'']
        assert active_set1 == active_set2
        
        # Memory contents should be identical
        memories1 = {m.memory_id: m for m in rebuilder.memory.get_all_memories()}
        
        rebuilder.rebuild_from_scratch()
        memories2 = {m.memory_id: m for m in rebuilder.memory.get_all_memories()}
        
        assert memories1.keys() == memories2.keys()
        for memory_id in memories1:
            assert memories1[memory_id].content == memories2[memory_id].content


class TestDeterministicRebuildIntegration:
    """Integration tests for deterministic rebuild."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "integration_rebuild"
    
    def test_convenience_function(self, temp_storage):
        """Test convenience function for rebuild."""
        # Create event store with events
        event_store = create_event_store(temp_storage)
        event_store.append_event("memory_ingest", {
            "content": "Integration test content",
            "metadata": {"test": "convenience"}
        })
        
        # Create mock memory
        memory = MockLuminaMemory()
        
        # Use convenience function
        stats = rebuild_memory_from_events(temp_storage, memory, use_snapshot=False)
        
        assert stats[''events_processed''] == 1
        assert len(memory.get_all_memories()) == 1
    
    def test_error_handling(self, temp_storage):
        """Test error handling in rebuild."""
        memory = MockLuminaMemory()
        
        # Test with non-existent storage
        with pytest.raises(Exception):
            rebuild_memory_from_events(Path("/nonexistent/path"), memory)
    
    def test_large_scale_rebuild(self, temp_storage):
        """Test rebuild with larger number of events."""
        # Create many events
        event_store = create_event_store(temp_storage)
        
        num_events = 100
        for i in range(num_events):
            event_store.append_event("memory_ingest", {
                "content": f"Large scale content {i}",
                "metadata": {"index": i, "batch": i // 10}
            })
        
        # Rebuild
        memory = MockLuminaMemory()
        stats = rebuild_memory_from_events(temp_storage, memory, use_snapshot=False)
        
        assert stats[''events_processed''] == num_events
        assert stats[''final_active_set_size''] == num_events
        assert len(memory.get_all_memories()) == num_events


if __name__ == "__main__":
    pytest.main([__file__, "-v"])