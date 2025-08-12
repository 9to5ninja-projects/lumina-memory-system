"""
Tests for M4: Event Store + Deterministic Rebuild

Comprehensive test suite for event store functionality including:
- Event storage and retrieval with cryptographic guarantees
- Hash chain integrity verification
- Snapshot creation and restoration
- Deterministic rebuild from events
- Active-Set uniqueness enforcement
"""

import pytest
import tempfile
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lumina_memory.event_store import (
    EventStore, StoredEvent, EventStoreSnapshot, create_event_store, EventStoreError
)
from lumina_memory.crypto_ids import memory_content_id, verify_content_integrity
from lumina_memory.event_hashing import create_genesis_entry, verify_hash_chain
from lumina_memory.kernel import MemoryRecord
import numpy as np


class TestEventStore:
    """Test event store core functionality."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_event_store"
    
    @pytest.fixture
    def event_store(self, temp_storage):
        """Create test event store."""
        return create_event_store(temp_storage)
    
    @pytest.fixture
    def sample_memory_records(self):
        """Create sample memory records for testing."""
        records = []
        for i in range(5):
            record = MemoryRecord(
                memory_id=f"test_mem_{i}",
                content=f"Test content {i}",
                embedding=np.random.rand(384).astype(np.float32),
                metadata={"test_id": i, "type": "test"}
            )
            records.append(record)
        return records
    
    def test_event_store_initialization(self, temp_storage):
        """Test event store initialization creates required files."""
        store = create_event_store(temp_storage)
        
        # Check storage directory exists
        assert temp_storage.exists()
        
        # Check required files exist
        assert (temp_storage / "events.jsonl").exists()
        assert (temp_storage / "snapshots.jsonl").exists()
        assert (temp_storage / "event_index.db").exists()
        
        # Check SQLite schema
        with sqlite3.connect(temp_storage / "event_index.db") as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type=''table''")
            tables = [row[0] for row in cursor.fetchall()]
            assert "events" in tables
            assert "snapshots" in tables
    
    def test_append_event_basic(self, event_store):
        """Test basic event appending."""
        content = {"action": "test", "data": "sample"}
        
        stored_event = event_store.append_event("test_event", content)
        
        assert stored_event.event_type == "test_event"
        assert stored_event.content == content
        assert stored_event.content_hash == memory_content_id(content)
        assert len(stored_event.event_id) == 64  # SHA-256 hex length
        assert len(stored_event.chain_hash) == 64
        
        # Verify timestamp format
        timestamp = datetime.fromisoformat(stored_event.timestamp.replace(''Z'', ''+00:00''))
        assert timestamp.tzinfo == timezone.utc
    
    def test_event_idempotency(self, event_store):
        """Test that identical content produces identical events."""
        content = {"action": "duplicate_test", "value": 42}
        
        event1 = event_store.append_event("test_type", content)
        event2 = event_store.append_event("test_type", content)
        
        # Should return same event (idempotent)
        assert event1.event_id == event2.event_id
        assert event1.content_hash == event2.content_hash
        assert event1.chain_hash == event2.chain_hash
    
    def test_event_retrieval(self, event_store):
        """Test event retrieval by ID."""
        content = {"test": "retrieval"}
        stored_event = event_store.append_event("retrieval_test", content)
        
        # Retrieve event
        retrieved_event = event_store.get_event(stored_event.event_id)
        
        assert retrieved_event is not None
        assert retrieved_event.event_id == stored_event.event_id
        assert retrieved_event.content == content
        
        # Test non-existent event
        fake_id = "0" * 64
        assert event_store.get_event(fake_id) is None
    
    def test_events_by_type(self, event_store):
        """Test retrieving events by type."""
        # Add events of different types
        event_store.append_event("type_a", {"data": 1})
        event_store.append_event("type_a", {"data": 2})
        event_store.append_event("type_b", {"data": 3})
        
        type_a_events = event_store.get_events_by_type("type_a")
        assert len(type_a_events) == 2
        
        type_b_events = event_store.get_events_by_type("type_b")
        assert len(type_b_events) == 1
        
        # Test with limit
        limited_events = event_store.get_events_by_type("type_a", limit=1)
        assert len(limited_events) == 1
    
    def test_events_since_timestamp(self, event_store):
        """Test retrieving events since timestamp."""
        # Add first event
        event1 = event_store.append_event("time_test", {"order": 1})
        
        # Get timestamp after first event
        since_timestamp = event1.timestamp
        
        # Add more events
        event_store.append_event("time_test", {"order": 2})
        event_store.append_event("time_test", {"order": 3})
        
        # Get events since timestamp
        recent_events = event_store.get_events_since(since_timestamp)
        assert len(recent_events) == 2  # Should not include first event
        
        # Verify order
        orders = [event.content["order"] for event in recent_events]
        assert orders == [2, 3]
    
    def test_hash_chain_integrity(self, event_store):
        """Test hash chain integrity verification."""
        # Add several events
        events = []
        for i in range(5):
            event = event_store.append_event("chain_test", {"index": i})
            events.append(event)
        
        # Verify chain integrity
        assert event_store.verify_integrity()
        
        # Check chain linking
        for i in range(1, len(events)):
            # Each event should have different chain hash
            assert events[i].chain_hash != events[i-1].chain_hash
    
    def test_content_integrity_verification(self, event_store):
        """Test content integrity verification."""
        content = {"integrity": "test", "data": [1, 2, 3]}
        event = event_store.append_event("integrity_test", content)
        
        # Verify content integrity
        assert verify_content_integrity(event.content, event.content_hash)
        
        # Test with modified content
        modified_content = content.copy()
        modified_content["data"] = [1, 2, 4]  # Changed last element
        assert not verify_content_integrity(modified_content, event.content_hash)
    
    def test_snapshot_creation(self, event_store, sample_memory_records):
        """Test snapshot creation and persistence."""
        active_set = {
            memory_content_id(record.content): record.memory_id
            for record in sample_memory_records
        }
        
        snapshot = event_store.create_snapshot(sample_memory_records, active_set)
        
        assert snapshot.snapshot_id
        assert len(snapshot.memory_records) == len(sample_memory_records)
        assert snapshot.active_set_state == active_set
        
        # Verify snapshot content hash
        snapshot_content_id = memory_content_id({
            ''timestamp'': snapshot.timestamp,
            ''last_event_hash'': snapshot.last_event_hash,
            ''memory_records'': snapshot.memory_records,
            ''active_set_state'': snapshot.active_set_state
        })
        assert snapshot.snapshot_id == snapshot_content_id
    
    def test_snapshot_retrieval(self, event_store, sample_memory_records):
        """Test snapshot retrieval."""
        active_set = {
            memory_content_id(record.content): record.memory_id
            for record in sample_memory_records
        }
        
        # Create snapshot
        created_snapshot = event_store.create_snapshot(sample_memory_records, active_set)
        
        # Retrieve latest snapshot
        retrieved_snapshot = event_store.get_latest_snapshot()
        
        assert retrieved_snapshot is not None
        assert retrieved_snapshot.snapshot_id == created_snapshot.snapshot_id
        assert len(retrieved_snapshot.memory_records) == len(sample_memory_records)
    
    def test_event_store_statistics(self, event_store):
        """Test event store statistics."""
        # Initial statistics
        stats = event_store.get_statistics()
        assert stats[''total_events''] == 0  # Genesis entry doesn''t count as event
        assert stats[''total_snapshots''] == 0
        assert stats[''chain_verified''] is True
        
        # Add some events
        event_store.append_event("stats_test", {"data": 1})
        event_store.append_event("stats_test", {"data": 2})
        event_store.append_event("other_type", {"data": 3})
        
        # Updated statistics
        stats = event_store.get_statistics()
        assert stats[''total_events''] == 3
        assert stats[''event_counts_by_type''][''stats_test''] == 2
        assert stats[''event_counts_by_type''][''other_type''] == 1
        assert stats[''hash_chain_length''] > 0
    
    def test_rebuild_iterator(self, event_store):
        """Test deterministic rebuild iterator."""
        # Add events in specific order
        test_data = [
            {"type": "memory_ingest", "content": "First memory"},
            {"type": "memory_ingest", "content": "Second memory"},
            {"type": "memory_update", "memory_id": "test_1", "new_content": "Updated"},
            {"type": "index_rebuild", "reason": "periodic"}
        ]
        
        added_events = []
        for data in test_data:
            event = event_store.append_event(data["type"], data)
            added_events.append(event)
        
        # Test rebuild iterator
        rebuilt_events = list(event_store.rebuild_from_events())
        
        # Should get same events in same order
        assert len(rebuilt_events) == len(added_events)
        for original, rebuilt in zip(added_events, rebuilt_events):
            assert original.event_id == rebuilt.event_id
            assert original.content == rebuilt.content
    
    def test_storage_persistence(self, temp_storage):
        """Test that event store persists across restarts."""
        # Create event store and add events
        store1 = create_event_store(temp_storage)
        event1 = store1.append_event("persist_test", {"data": "persistent"})
        
        # Close and reopen
        del store1
        store2 = create_event_store(temp_storage)
        
        # Event should still exist
        retrieved_event = store2.get_event(event1.event_id)
        assert retrieved_event is not None
        assert retrieved_event.content == {"data": "persistent"}
        
        # Statistics should match
        stats = store2.get_statistics()
        assert stats[''total_events''] == 1
    
    def test_jsonl_format_validation(self, event_store):
        """Test that events are stored in valid JSONL format."""
        # Add some events
        events = []
        for i in range(3):
            event = event_store.append_event("jsonl_test", {"index": i})
            events.append(event)
        
        # Read JSONL file directly
        events_file = event_store.events_file
        with open(events_file, ''r'', encoding=''utf-8'') as f:
            lines = f.readlines()
        
        # Should have one line per event
        assert len([line for line in lines if line.strip()]) == len(events)
        
        # Each line should be valid JSON
        for line in lines:
            if line.strip():
                event_data = json.loads(line)
                # Should have required fields
                assert ''event_id'' in event_data
                assert ''event_type'' in event_data
                assert ''timestamp'' in event_data
                assert ''content'' in event_data
                assert ''content_hash'' in event_data
                assert ''chain_hash'' in event_data


class TestEventStoreError:
    """Test event store error handling."""
    
    def test_invalid_storage_path(self):
        """Test error handling for invalid storage paths."""
        with pytest.raises(Exception):
            # Try to create event store in non-existent parent directory
            create_event_store("/nonexistent/path/store")
    
    def test_corrupted_database(self, temp_storage):
        """Test handling of corrupted database."""
        # Create valid event store first
        store = create_event_store(temp_storage)
        store.append_event("test", {"data": "value"})
        
        # Corrupt the database file
        db_file = temp_storage / "event_index.db"
        with open(db_file, ''w'') as f:
            f.write("corrupted data")
        
        # Should handle corruption gracefully
        with pytest.raises(EventStoreError):
            corrupted_store = create_event_store(temp_storage)
            corrupted_store.append_event("test", {"data": "new"})


class TestEventStoreIntegration:
    """Integration tests for event store with other components."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "integration_test"
    
    def test_content_addressable_integration(self, temp_storage):
        """Test integration with content-addressable ID system."""
        store = create_event_store(temp_storage)
        
        # Same content should produce same IDs
        content1 = {"message": "Hello World", "priority": 1}
        content2 = {"message": "Hello World", "priority": 1}  # Identical
        content3 = {"message": "Hello World", "priority": 2}  # Different
        
        event1 = store.append_event("test", content1)
        event2 = store.append_event("test", content2)
        event3 = store.append_event("test", content3)
        
        # Same content = same content hash and event ID
        assert event1.content_hash == event2.content_hash
        assert event1.event_id == event2.event_id  # Idempotent
        
        # Different content = different hashes
        assert event1.content_hash != event3.content_hash
        assert event1.event_id != event3.event_id
    
    def test_hash_chain_integration(self, temp_storage):
        """Test integration with hash chain verification."""
        store = create_event_store(temp_storage)
        
        # Add events to build chain
        events = []
        for i in range(10):
            event = store.append_event("chain_integration", {"step": i})
            events.append(event)
        
        # Verify complete hash chain
        assert store.verify_integrity()
        
        # Chain hashes should all be different
        chain_hashes = [event.chain_hash for event in events]
        assert len(set(chain_hashes)) == len(chain_hashes)
        
        # Genesis entry should exist
        stats = store.get_statistics()
        assert stats[''hash_chain_length''] > 0
    
    def test_snapshot_with_real_memory_records(self, temp_storage):
        """Test snapshot creation with realistic memory records."""
        store = create_event_store(temp_storage)
        
        # Create realistic memory records
        memory_records = []
        active_set = {}
        
        for i in range(5):
            content = f"This is memory content number {i} with some detailed text."
            content_id = memory_content_id(content)
            
            record = MemoryRecord(
                memory_id=f"mem_{content_id[:8]}",
                content=content,
                embedding=np.random.rand(384).astype(np.float32),
                metadata={
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "type": "test_memory",
                    "index": i
                }
            )
            
            memory_records.append(record)
            active_set[content_id] = record.memory_id
        
        # Create snapshot
        snapshot = store.create_snapshot(memory_records, active_set)
        
        # Verify snapshot integrity
        assert len(snapshot.memory_records) == 5
        assert len(snapshot.active_set_state) == 5
        
        # Verify memory record serialization
        for original, serialized in zip(memory_records, snapshot.memory_records):
            assert serialized[''memory_id''] == original.memory_id
            assert serialized[''content''] == original.content
            assert serialized[''metadata''] == original.metadata
            
            # Embedding should be serialized as list
            if original.embedding is not None:
                assert isinstance(serialized[''embedding''], list)
                assert len(serialized[''embedding'']) == len(original.embedding)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])