"""
Event definitions and schemas for the Lumina Memory System.

This module defines the event types and structures used in the append-only event store.
All events are immutable once created and form a cryptographic hash chain.
"""

from typing import Dict, Any, Optional, Literal, Union
from dataclasses import dataclass, field
from datetime import datetime
import time

# Event type definitions
EventType = Literal[
    "INGEST",           # New memory ingested
    "RECALL",           # Memory recalled/accessed  
    "REINFORCE",        # Memory reinforcement/strengthening
    "CONSOLIDATE",      # Memory consolidation event
    "FORGET",           # Memory forgetting/deletion
    "POLICY_CHANGE",    # System policy or parameter change
    "MIGRATION",        # Data migration event
    "SYSTEM_INIT",      # System initialization (genesis event)
    "SNAPSHOT",         # Snapshot creation event
    "CONFLICT"          # Conflict resolution event (Active-Set enforcement)
]

@dataclass
class EventPayload:
    """Base class for event payloads."""
    pass

@dataclass 
class IngestPayload(EventPayload):
    """Payload for INGEST events - new memory creation."""
    id: str                    # Content-addressed ID (private_id from crypto_ids)
    public_cid: str           # Public content ID for deduplication
    content: str              # Normalized content text
    source: str               # Source identifier
    version: str              # Embedding model version
    metadata: Dict[str, Any]  # Additional metadata
    hrr_vector: Optional[list] = None  # HRR reference vector (serialized)
    embedding: Optional[list] = None   # Semantic embedding (if stored)

@dataclass
class RecallPayload(EventPayload):
    """Payload for RECALL events - memory access."""
    memory_id: str            # ID of recalled memory
    query: str                # Original query that triggered recall
    relevance_score: float    # Relevance/similarity score
    recall_context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConflictPayload(EventPayload):
    """Payload for CONFLICT events - Active-Set uniqueness enforcement."""
    conflict_type: str        # Type of conflict detected
    original_id: str         # Original memory ID
    conflicting_id: str      # Conflicting memory ID
    resolution: str          # How conflict was resolved
    winning_cid: str         # Content ID of winning memory
    losing_cid: str          # Content ID of losing memory

@dataclass
class SystemInitPayload(EventPayload):
    """Payload for SYSTEM_INIT events - genesis events."""
    system_version: str       # Version of the memory system
    initialization_params: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

# Payload type mapping
PAYLOAD_TYPES = {
    "INGEST": IngestPayload,
    "RECALL": RecallPayload,
    "CONFLICT": ConflictPayload,
    "SYSTEM_INIT": SystemInitPayload,
}

@dataclass
class Event:
    """
    Immutable event in the memory system event log.
    
    Events form a cryptographic hash chain and are the source of truth
    for all state changes in the memory system.
    """
    event_type: EventType
    payload: EventPayload
    timestamp: float = field(default_factory=time.time)
    event_id: Optional[str] = None      # Content-addressed event ID
    prev_hash: Optional[str] = None     # Previous event hash (for chaining)
    event_hash: Optional[str] = None    # This event's hash
    sequence_number: Optional[int] = None  # Sequential event number
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_type": self.event_type,
            "payload": self._payload_to_dict(),
            "timestamp": self.timestamp,
            "event_id": self.event_id,
            "prev_hash": self.prev_hash,
            "event_hash": self.event_hash,
            "sequence_number": self.sequence_number
        }
    
    def _payload_to_dict(self) -> Dict[str, Any]:
        """Convert payload to dictionary."""
        if hasattr(self.payload, '__dict__'):
            return self.payload.__dict__
        return self.payload
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary."""
        event_type = data["event_type"]
        payload_class = PAYLOAD_TYPES.get(event_type, EventPayload)
        
        payload_data = data["payload"]
        if payload_class != EventPayload:
            payload = payload_class(**payload_data)
        else:
            payload = payload_data
        
        return cls(
            event_type=event_type,
            payload=payload,
            timestamp=data.get("timestamp", time.time()),
            event_id=data.get("event_id"),
            prev_hash=data.get("prev_hash"),
            event_hash=data.get("event_hash"),
            sequence_number=data.get("sequence_number")
        )

def create_ingest_event(
    content: str,
    source: str,
    version: str,
    metadata: Dict[str, Any],
    fingerprint: Dict[str, str],
    hrr_vector: Optional[list] = None,
    embedding: Optional[list] = None
) -> Event:
    """Create an INGEST event with proper content addressing."""
    payload = IngestPayload(
        id=fingerprint["private_id"],
        public_cid=fingerprint["public_cid"],
        content=content,
        source=source,
        version=version,
        metadata=metadata,
        hrr_vector=hrr_vector,
        embedding=embedding
    )
    
    return Event(event_type="INGEST", payload=payload)

def create_conflict_event(
    original_id: str,
    conflicting_id: str,
    winning_cid: str,
    losing_cid: str,
    resolution: str = "auto_fork"
) -> Event:
    """Create a CONFLICT event for Active-Set uniqueness enforcement."""
    payload = ConflictPayload(
        conflict_type="content_hash_mismatch",
        original_id=original_id,
        conflicting_id=conflicting_id,
        resolution=resolution,
        winning_cid=winning_cid,
        losing_cid=losing_cid
    )
    
    return Event(event_type="CONFLICT", payload=payload)

def create_system_init_event(
    system_version: str,
    initialization_params: Optional[Dict[str, Any]] = None
) -> Event:
    """Create a SYSTEM_INIT genesis event."""
    payload = SystemInitPayload(
        system_version=system_version,
        initialization_params=initialization_params or {}
    )
    
    return Event(event_type="SYSTEM_INIT", payload=payload)
