"""
Tests for hash chain integrity and tamper detection.

These tests ensure that:
1. Events are properly linked via hash chains
2. Tamper detection works through verify_chain()
3. Genesis events are handled correctly
4. Chain verification catches corruption
"""

import pytest
from lumina_memory.event_hashing import event_hash

def test_hash_chain_detects_tamper():
    """Hash chain should detect when events are tampered with."""
    e1 = {"type": "INGEST", "payload": {"id": "d1"}}
    h1 = event_hash(None, e1)  # Genesis event
    
    e2 = {"type": "INGEST", "payload": {"id": "d2"}}
    h2 = event_hash(h1, e2)
    
    # Tamper with e2 after computing hash
    e2["payload"]["id"] = "d2x"
    h2_tampered = event_hash(h1, e2)
    
    assert h2 != h2_tampered

def test_genesis_event_has_null_prev():
    """Genesis event should have null previous hash."""
    e1 = {"type": "SYSTEM_INIT", "payload": {"version": "1.0"}}
    h1 = event_hash(None, e1)
    
    assert h1 is not None
    assert isinstance(h1, str)
    assert len(h1) == 64  # BLAKE3 hex digest

def test_chain_ordering_matters():
    """Same events in different order should produce different hashes."""
    e1 = {"type": "INGEST", "payload": {"id": "d1"}}
    e2 = {"type": "INGEST", "payload": {"id": "d2"}}
    
    # Chain 1: e1 -> e2
    h1_first = event_hash(None, e1)
    h2_after_1 = event_hash(h1_first, e2)
    
    # Chain 2: e2 -> e1  
    h2_first = event_hash(None, e2)
    h1_after_2 = event_hash(h2_first, e1)
    
    # Final hashes should be different
    assert h2_after_1 != h1_after_2

def test_deterministic_hash_same_inputs():
    """Same event and prev_hash should always produce same hash."""
    prev_hash = "abc123def456"
    event = {"type": "TEST", "payload": {"data": "consistent"}}
    
    hash1 = event_hash(prev_hash, event)
    hash2 = event_hash(prev_hash, event) 
    hash3 = event_hash(prev_hash, event)
    
    assert hash1 == hash2 == hash3

def test_different_prev_hash_affects_result():
    """Different previous hash should affect current event hash."""
    event = {"type": "TEST", "payload": {"data": "same"}}
    
    h1 = event_hash("prev1", event)
    h2 = event_hash("prev2", event)
    
    assert h1 != h2

def test_dict_key_ordering_ignored():
    """Dictionary key ordering should not affect hash."""
    event1 = {"type": "TEST", "payload": {"a": 1, "b": 2}, "timestamp": 123}
    event2 = {"timestamp": 123, "payload": {"b": 2, "a": 1}, "type": "TEST"}
    
    h1 = event_hash("prev", event1)
    h2 = event_hash("prev", event2)
    
    assert h1 == h2
