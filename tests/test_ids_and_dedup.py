"""
Tests for content-addressed IDs and deduplication behavior.

These tests ensure that:
1. Identical content produces identical IDs (idempotent ingest)
2. Different content produces different IDs
3. Conflict resolution works when same ID maps to different content
4. Memory.id and public_cid are content-addressed
"""

import pytest
from lumina_memory.crypto_ids import content_fingerprint

def test_idempotent_ingest_same_content():
    """Same content should produce same fingerprint regardless of whitespace."""
    salt = b"x" * 32  # 32-byte salt for BLAKE3
    fp1 = content_fingerprint("Hello  world", "s1", "v1", "m@1", secret_salt=salt)
    fp2 = content_fingerprint(" hello world ", "s1", "v1", "m@1", secret_salt=salt)
    assert fp1["public_cid"] == fp2["public_cid"]

def test_different_content_different_cid():
    """Different content should produce different fingerprints."""
    salt = b"x" * 32
    fp1 = content_fingerprint("Hello world", "s1", "v1", "m@1", secret_salt=salt)
    fp2 = content_fingerprint("Hello mars", "s1", "v1", "m@1", secret_salt=salt)
    assert fp1["public_cid"] != fp2["public_cid"]

def test_different_versions_different_cid():
    """Different embedding versions should produce different fingerprints."""
    salt = b"x" * 32
    fp1 = content_fingerprint("Hello world", "s1", "v1", "m@1", secret_salt=salt)
    fp2 = content_fingerprint("Hello world", "s1", "v2", "m@1", secret_salt=salt)
    assert fp1["public_cid"] != fp2["public_cid"]

def test_different_salt_different_private_id():
    """Different salts should produce different private IDs."""
    salt1 = b"a" * 32
    salt2 = b"b" * 32
    fp1 = content_fingerprint("Hello world", "s1", "v1", "m@1", secret_salt=salt1)
    fp2 = content_fingerprint("Hello world", "s1", "v1", "m@1", secret_salt=salt2)
    # Public CID should be same (no salt used)
    assert fp1["public_cid"] == fp2["public_cid"]
    # Private ID should be different (salt affects HMAC)
    assert fp1["private_id"] != fp2["private_id"]

def test_conflict_when_same_id_different_content(event_store_factory):
    """Create two INGEST events with same id but different normalized content.
    Expect a CONFLICT event or forked id on rebuild."""
    store = event_store_factory()
    salt = b"test_salt_32_bytes_long_exactly!"
    
    # Create first memory
    fp1 = content_fingerprint("Hello world", "s1", "v1", "m@1", secret_salt=salt)
    store.append_event({
        "type": "INGEST",
        "payload": {
            "id": fp1["private_id"],
            "public_cid": fp1["public_cid"],
            "content": "Hello world",
            "source": "s1",
            "version": "v1",
            "metadata": "m@1"
        }
    })
    
    # Force different content with same ID (corruption scenario)
    store.append_event({
        "type": "INGEST", 
        "payload": {
            "id": fp1["private_id"],  # Same ID
            "public_cid": "different_cid",  # But different content hash
            "content": "Hello mars",  # Different content
            "source": "s1",
            "version": "v1", 
            "metadata": "m@1"
        }
    })
    
    # Rebuild should detect conflict
    state = store.rebuild_deterministic()
    
    # Should have generated a CONFLICT event or forked the ID
    events = store.get_events()
    conflict_events = [e for e in events if e.get("type") == "CONFLICT"]
    assert len(conflict_events) > 0 or len(state["memories"]) == 2  # Either conflict event or forked

def test_content_canonicalization():
    """Test that content is properly canonicalized before fingerprinting."""
    salt = b"canon_test_salt_32_bytes_long_!!"
    # Different whitespace arrangements should normalize to same fingerprint
    variations = [
        "Hello   world",
        " Hello world ",
        "\n Hello  world \t",
        "hello world",  # Case normalization
    ]
    
    fingerprints = [
        content_fingerprint(text, "s1", "v1", "m@1", secret_salt=salt) 
        for text in variations
    ]
    
    # All should have same public_cid after canonicalization
    base_cid = fingerprints[0]["public_cid"]
    for fp in fingerprints[1:]:
        assert fp["public_cid"] == base_cid
