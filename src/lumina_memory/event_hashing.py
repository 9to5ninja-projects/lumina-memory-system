"""
Event hashing for hash chain integrity and tamper detection.

This module provides event_hash() for building cryptographic hash chains
and verify_chain() for detecting tampering.
"""

import json
from typing import Dict, Any, Optional, List, Union

from .crypto_ids import _canonicalize_dict, _blake3_hash

class ChainVerificationError(Exception):
    """Raised when hash chain verification fails."""
    pass

class TamperDetectionError(Exception):
    """Raised when event tampering is detected."""
    pass

def event_hash(prev_hash: Optional[str], event: Dict[str, Any]) -> str:
    """
    Generate hash for an event, linking it to previous event in chain.
    
    Args:
        prev_hash: Hash of previous event (None for genesis events)
        event: Event dictionary to hash
        
    Returns:
        Hex string hash of the event
    """
    # Create hashable event data
    hash_data = {
        "prev_hash": prev_hash,
        "event": event
    }
    
    # Canonicalize to ensure deterministic ordering
    canonical_json = _canonicalize_dict(hash_data)
    event_bytes = canonical_json.encode('utf-8')
    
    return _blake3_hash(event_bytes)

def verify_event_hash(event_dict: Dict[str, Any]) -> bool:
    """
    Verify that an event's stored hash matches its computed hash.
    
    Args:
        event_dict: Event dictionary with event_hash and prev_hash fields
        
    Returns:
        True if hash is valid, False otherwise
    """
    if "event_hash" not in event_dict:
        return False
    
    stored_hash = event_dict["event_hash"]
    prev_hash = event_dict.get("prev_hash")
    
    # Create event copy without the hash fields for verification
    event_copy = event_dict.copy()
    event_copy.pop("event_hash", None)
    
    computed_hash = event_hash(prev_hash, event_copy)
    return computed_hash == stored_hash

def verify_chain(events: List[Dict[str, Any]]) -> bool:
    """
    Verify entire hash chain for integrity.
    
    Args:
        events: List of event dictionaries in chronological order
        
    Returns:
        True if chain is valid
        
    Raises:
        ChainVerificationError: If chain structure is invalid
        TamperDetectionError: If tampering is detected
    """
    if not events:
        return True  # Empty chain is valid
    
    prev_hash = None
    
    for i, event in enumerate(events):
        # Check required fields
        if "event_hash" not in event:
            raise ChainVerificationError(f"Event {i} missing event_hash field")
        
        # For genesis event, prev_hash should be None
        if i == 0:
            if event.get("prev_hash") is not None:
                raise ChainVerificationError(f"Genesis event has non-null prev_hash: {event.get('prev_hash')}")
        else:
            # For non-genesis events, prev_hash should match previous event's hash
            expected_prev_hash = events[i-1]["event_hash"]
            if event.get("prev_hash") != expected_prev_hash:
                raise ChainVerificationError(
                    f"Event {i} prev_hash mismatch. Expected: {expected_prev_hash}, "
                    f"Got: {event.get('prev_hash')}"
                )
        
        # Verify event hash integrity
        if not verify_event_hash(event):
            raise TamperDetectionError(f"Event {i} hash verification failed - content has been tampered")
        
        prev_hash = event["event_hash"]
    
    return True

def create_genesis_event(event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a genesis event (first event in chain).
    
    Args:
        event_type: Type of genesis event
        payload: Event payload data
        
    Returns:
        Complete genesis event with hash
    """
    import time
    
    genesis_event = {
        "event_type": event_type,
        "payload": payload,
        "timestamp": time.time(),
        "prev_hash": None,
        "sequence_number": 0
    }
    
    # Generate hash for genesis event
    genesis_event["event_hash"] = event_hash(None, genesis_event)
    
    return genesis_event

def create_chained_event(
    prev_event: Dict[str, Any],
    event_type: str, 
    payload: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a new event chained to previous event.
    
    Args:
        prev_event: Previous event in chain
        event_type: Type of new event
        payload: Event payload data
        
    Returns:
        Complete chained event with hash
    """
    import time
    
    new_event = {
        "event_type": event_type,
        "payload": payload,
        "timestamp": time.time(),
        "prev_hash": prev_event["event_hash"],
        "sequence_number": prev_event.get("sequence_number", 0) + 1
    }
    
    # Generate hash for new event
    new_event["event_hash"] = event_hash(prev_event["event_hash"], new_event)
    
    return new_event

def get_chain_head_hash(events: List[Dict[str, Any]]) -> Optional[str]:
    """Get hash of the last event in chain."""
    if not events:
        return None
    return events[-1]["event_hash"]

def validate_chain_continuity(events: List[Dict[str, Any]]) -> List[int]:
    """
    Check for gaps or discontinuities in event chain.
    
    Returns:
        List of event indices where discontinuities are found
    """
    discontinuities = []
    
    for i in range(1, len(events)):
        expected_prev = events[i-1]["event_hash"]
        actual_prev = events[i].get("prev_hash")
        
        if expected_prev != actual_prev:
            discontinuities.append(i)
    
    return discontinuities
