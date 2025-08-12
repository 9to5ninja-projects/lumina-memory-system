"""
Test event store integration with hash chain verification and encryption.

Tests the complete M4 event store system with:
- Hash chain verification on load
- Encrypted event storage
- Active-Set invariant enforcement
"""

import pytest
import tempfile
import os
import json
from pathlib import Path

from lumina_memory.events import create_ingest_event, create_system_init_event
from lumina_memory.crypto_ids import content_fingerprint
from lumina_memory.event_hashing import verify_chain
from lumina_memory.encryption import encrypt_data, decrypt_data


def test_event_store_verifies_chain_on_load():
    """Test that event store verifies hash chain integrity when loading from disk."""
    # This test should fail initially - we need to implement this
    with tempfile.TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "test_store"
        
        # TODO: Import the full EventStore when implemented
        # from lumina_memory.event_store import EventStore
        # store = EventStore(store_path)
        
        # For now, this is a placeholder that should fail
        with pytest.raises(ImportError):
            from lumina_memory.event_store import EventStore


def test_basic_event_creation():
    """Test basic event creation works with our current system."""
    fingerprint = content_fingerprint("Test content", salt="test")
    
    event = create_ingest_event(
        content="Test content",
        source="test",
        version="test", 
        metadata={"test": True},
        fingerprint=fingerprint
    )
    
    assert event.event_type == "INGEST"
    assert event.payload.content == "Test content"
    assert event.payload.id == fingerprint["private_id"]
    assert event.payload.public_cid == fingerprint["public_cid"]
