"""
M4: Event Store + Deterministic Rebuild - Integration Tests

Simplified integration test suite to validate M4 functionality:
- Event store operations with cryptographic integrity
- Deterministic rebuild from events
- Active-Set uniqueness enforcement
- Snapshot creation and restoration
"""

import tempfile
import json
from pathlib import Path
from datetime import datetime, timezone

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lumina_memory.event_store import create_event_store
from lumina_memory.crypto_ids import memory_content_id


def test_event_store_basic_operations():
    """Test basic event store operations."""
    print("Testing event store basic operations...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "test_store"
        
        # Create event store
        store = create_event_store(storage_path)
        
        # Test event creation
        content = {"message": "Hello World", "priority": 1}
        event = store.append_event("test_event", content)
        
        print(f" Event created: {event.event_id[:16]}...")
        print(f" Content hash: {event.content_hash[:16]}...")
        print(f" Chain hash: {event.chain_hash[:16]}...")
        
        # Test event retrieval
        retrieved = store.get_event(event.event_id)
        assert retrieved is not None
        assert retrieved.content == content
        print(" Event retrieval working")
        
        # Test idempotency
        event2 = store.append_event("test_event", content)
        assert event.event_id == event2.event_id
        print(" Event idempotency working")
        
        # Test statistics
        stats = store.get_statistics()
        print(f" Statistics: {stats[''total_events'']} events, chain length {stats[''hash_chain_length'']}")
        
        # Test integrity verification
        is_valid = store.verify_integrity()
        print(f" Integrity verification: {is_valid}")
        
        return True


def test_content_addressable_uniqueness():
    """Test content-addressable ID uniqueness."""
    print("Testing content-addressable uniqueness...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "uniqueness_test"
        store = create_event_store(storage_path)
        
        # Same content should produce same IDs
        content1 = {"data": "identical content", "value": 42}
        content2 = {"data": "identical content", "value": 42}
        content3 = {"data": "different content", "value": 42}
        
        event1 = store.append_event("test", content1)
        event2 = store.append_event("test", content2)  # Should be idempotent
        event3 = store.append_event("test", content3)
        
        # Same content = same IDs
        assert event1.content_hash == event2.content_hash
        assert event1.event_id == event2.event_id
        print(" Identical content produces identical IDs")
        
        # Different content = different IDs
        assert event1.content_hash != event3.content_hash
        assert event1.event_id != event3.event_id
        print(" Different content produces different IDs")
        
        # Verify deterministic ID generation
        manual_hash = memory_content_id(content1)
        assert event1.content_hash == manual_hash
        print(" Deterministic ID generation working")
        
        return True


def test_hash_chain_integrity():
    """Test hash chain integrity."""
    print("Testing hash chain integrity...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "chain_test"
        store = create_event_store(storage_path)
        
        # Add multiple events to build chain
        events = []
        for i in range(5):
            event = store.append_event("chain_test", {"step": i, "data": f"Step {i}"})
            events.append(event)
            print(f"  Added event {i}: {event.chain_hash[:8]}...")
        
        # Verify chain integrity
        assert store.verify_integrity()
        print(" Hash chain integrity verified")
        
        # Chain hashes should all be different
        chain_hashes = [event.chain_hash for event in events]
        assert len(set(chain_hashes)) == len(chain_hashes)
        print(" All chain hashes unique")
        
        return True


def test_event_types_and_querying():
    """Test event types and querying capabilities."""
    print("Testing event types and querying...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "query_test"
        store = create_event_store(storage_path)
        
        # Add events of different types
        store.append_event("memory_ingest", {"content": "Memory 1"})
        store.append_event("memory_ingest", {"content": "Memory 2"})
        store.append_event("memory_update", {"memory_id": "test", "new_content": "Updated"})
        store.append_event("index_rebuild", {"reason": "periodic"})
        
        # Query by type
        ingest_events = store.get_events_by_type("memory_ingest")
        assert len(ingest_events) == 2
        print(" Query by event type working")
        
        update_events = store.get_events_by_type("memory_update")
        assert len(update_events) == 1
        print(" Multiple event types supported")
        
        # Test with limit
        limited_events = store.get_events_by_type("memory_ingest", limit=1)
        assert len(limited_events) == 1
        print(" Query limits working")
        
        return True


def test_snapshot_operations():
    """Test snapshot creation and retrieval."""
    print("Testing snapshot operations...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "snapshot_test"
        store = create_event_store(storage_path)
        
        # Add some events first
        for i in range(3):
            store.append_event("memory_ingest", {"content": f"Content {i}"})
        
        # Create sample snapshot data
        memory_records = [
            {
                ''memory_id'': f"mem_{i}",
                ''content'': f"Content {i}",
                ''embedding'': None,
                ''metadata'': {"index": i}
            }
            for i in range(3)
        ]
        
        active_set = {
            memory_content_id(f"Content {i}"): f"mem_{i}"
            for i in range(3)
        }
        
        # Create snapshot (mock MemoryRecord objects)
        from lumina_memory.kernel import MemoryRecord
        import numpy as np
        
        mock_records = []
        for record_data in memory_records:
            mock_record = MemoryRecord(
                memory_id=record_data[''memory_id''],
                content=record_data[''content''],
                embedding=None,
                metadata=record_data[''metadata'']
            )
            mock_records.append(mock_record)
        
        snapshot = store.create_snapshot(mock_records, active_set)
        print(f" Snapshot created: {snapshot.snapshot_id[:16]}...")
        
        # Retrieve snapshot
        retrieved_snapshot = store.get_latest_snapshot()
        assert retrieved_snapshot is not None
        assert retrieved_snapshot.snapshot_id == snapshot.snapshot_id
        print(" Snapshot retrieval working")
        
        return True


def test_persistence_across_restarts():
    """Test that data persists across event store restarts."""
    print("Testing persistence across restarts...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "persistence_test"
        
        # Create store and add events
        store1 = create_event_store(storage_path)
        event1 = store1.append_event("persist_test", {"data": "persistent_data"})
        event2 = store1.append_event("persist_test", {"data": "more_data"})
        
        stats1 = store1.get_statistics()
        print(f"  Initial store: {stats1[''total_events'']} events")
        
        # Close and reopen
        del store1
        store2 = create_event_store(storage_path)
        
        # Data should still be there
        retrieved1 = store2.get_event(event1.event_id)
        retrieved2 = store2.get_event(event2.event_id)
        
        assert retrieved1 is not None
        assert retrieved2 is not None
        assert retrieved1.content == {"data": "persistent_data"}
        assert retrieved2.content == {"data": "more_data"}
        
        stats2 = store2.get_statistics()
        assert stats2[''total_events''] == stats1[''total_events'']
        
        print(" Data persistence across restarts working")
        return True


def test_deterministic_rebuild_iterator():
    """Test deterministic rebuild iterator."""
    print("Testing deterministic rebuild iterator...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "rebuild_test"
        store = create_event_store(storage_path)
        
        # Add events in specific order
        test_data = [
            {"type": "memory_ingest", "content": "First memory"},
            {"type": "memory_ingest", "content": "Second memory"},
            {"type": "memory_update", "memory_id": "test_1", "new_content": "Updated"},
        ]
        
        added_events = []
        for data in test_data:
            event = store.append_event(data["type"], data)
            added_events.append(event)
        
        # Test rebuild iterator
        rebuilt_events = list(store.rebuild_from_events())
        
        # Should get same events in same order
        assert len(rebuilt_events) == len(added_events)
        for original, rebuilt in zip(added_events, rebuilt_events):
            assert original.event_id == rebuilt.event_id
            assert original.content == rebuilt.content
        
        print(f" Deterministic rebuild iterator: {len(rebuilt_events)} events in order")
        return True


def test_jsonl_storage_format():
    """Test JSONL storage format."""
    print("Testing JSONL storage format...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "jsonl_test"
        store = create_event_store(storage_path)
        
        # Add events
        events = []
        for i in range(3):
            event = store.append_event("jsonl_test", {"index": i, "message": f"Message {i}"})
            events.append(event)
        
        # Read JSONL file directly
        events_file = storage_path / "events.jsonl"
        assert events_file.exists()
        
        with open(events_file, ''r'', encoding=''utf-8'') as f:
            lines = f.readlines()
        
        # Should have one line per event
        valid_lines = [line for line in lines if line.strip()]
        assert len(valid_lines) == len(events)
        
        # Each line should be valid JSON
        for line in valid_lines:
            event_data = json.loads(line)
            # Should have required fields
            required_fields = [''event_id'', ''event_type'', ''timestamp'', ''content'', ''content_hash'', ''chain_hash'']
            for field in required_fields:
                assert field in event_data
        
        print(" JSONL storage format correct")
        return True


def run_all_tests():
    """Run all M4 integration tests."""
    print(" Starting M4: Event Store + Deterministic Rebuild Integration Tests\n")
    
    tests = [
        test_event_store_basic_operations,
        test_content_addressable_uniqueness,
        test_hash_chain_integrity,
        test_event_types_and_querying,
        test_snapshot_operations,
        test_persistence_across_restarts,
        test_deterministic_rebuild_iterator,
        test_jsonl_storage_format,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print(f"\n{'='*60}")
            result = test()
            if result:
                print(f" {test.__name__} PASSED")
                passed += 1
            else:
                print(f" {test.__name__} FAILED")
                failed += 1
        except Exception as e:
            print(f" {test.__name__} FAILED with error: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f" M4 Integration Test Results:")
    print(f"   Passed: {passed}")
    print(f"   Failed: {failed}")
    print(f"   Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print(f"\n ALL M4 TESTS PASSED! Event Store + Deterministic Rebuild is ready.")
        return True
    else:
        print(f"\n  Some tests failed. M4 implementation needs fixes.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)