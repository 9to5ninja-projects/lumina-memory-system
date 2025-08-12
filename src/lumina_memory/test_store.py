"""
Simple event store for testing M4 identity and integrity rails.

This is a minimal implementation to support the test suite while we build 
the full M4 event store system.
"""

import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from .event_hashing import event_hash, verify_chain, create_genesis_event
from .crypto_ids import content_fingerprint


class SimpleEventStore:
    """Simple in-memory event store for testing."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.events: List[Dict[str, Any]] = []
        self.storage_path = storage_path
        self._last_hash: Optional[str] = None
        
        # Initialize with genesis event
        genesis = create_genesis_event("SYSTEM_INIT", {
            "version": "0.4.0",
            "timestamp": time.time()
        })
        self.events.append(genesis)
        self._last_hash = genesis["event_hash"]
    
    def append_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Append a new event to the store."""
        # Create chained event
        event = {
            **event_data,
            "timestamp": time.time(),
            "prev_hash": self._last_hash,
            "sequence_number": len(self.events)
        }
        
        # Generate event hash
        event["event_hash"] = event_hash(self._last_hash, event)
        
        # Append to store
        self.events.append(event)
        self._last_hash = event["event_hash"]
        
        return event
    
    def get_events(self) -> List[Dict[str, Any]]:
        """Get all events."""
        return self.events.copy()
    
    def verify_chain(self) -> bool:
        """Verify the integrity of the event chain."""
        return verify_chain(self.events)
    
    def rebuild_deterministic(self) -> Dict[str, Any]:
        """Rebuild state from events with Active-Set uniqueness enforcement."""
        state = {
            "memories": {},
            "active_ids": set(),
            "conflicts": []
        }
        
        for event in self.events:
            if event.get("type") == "INGEST":
                payload = event.get("payload", {})
                memory_id = payload.get("id")
                public_cid = payload.get("public_cid")
                
                if memory_id and public_cid:
                    # Check for conflicts (same ID, different content)
                    if memory_id in state["active_ids"]:
                        existing_cid = None
                        for existing_id, memory in state["memories"].items():
                            if existing_id == memory_id:
                                existing_cid = memory.get("public_cid")
                                break
                        
                        if existing_cid != public_cid:
                            # Conflict detected - create conflict event
                            conflict_event = {
                                "type": "CONFLICT",
                                "payload": {
                                    "original_id": memory_id,
                                    "conflicting_id": f"{memory_id}_fork",
                                    "winning_cid": existing_cid,
                                    "losing_cid": public_cid,
                                    "resolution": "auto_fork"
                                }
                            }
                            
                            # Append conflict event
                            self.append_event(conflict_event)
                            state["conflicts"].append(conflict_event)
                            
                            # Fork the conflicting memory
                            memory_id = f"{memory_id}_fork"
                    
                    # Add/update memory
                    state["memories"][memory_id] = payload
                    state["active_ids"].add(memory_id)
        
        return state


def event_store_factory():
    """Factory function to create event store instances for testing."""
    return SimpleEventStore()
