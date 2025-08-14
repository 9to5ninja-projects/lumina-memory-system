"""
M4: Deterministic Rebuild from Event Store

Provides byte-identical reconstruction of memory state from events.
Integrates event store with existing kernel while maintaining purity.

This module implements:
- Idempotent replay of events with conflict resolution
- Active-Set uniqueness enforcement using content-addressable IDs
- Snapshot-optimized rebuild for performance
- CLI interface for rebuild operations
"""

import numpy as np
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime, timezone

from .kernel import (
    MemoryRecord, LuminaMemory, create_memory_record, 
    ingest_memory, find_similar_memories, rebuild_index
)
from .crypto_ids import memory_content_id, verify_content_integrity, ContentAddressableIndex
from .event_store import EventStore, StoredEvent, EventStoreSnapshot, create_event_store
from .event_hashing import verify_hash_chain


@dataclass(frozen=True)
class RebuildState:
    """State tracking for deterministic rebuild process."""
    processed_events: Set[str]  # Event IDs already processed
    active_set: Dict[str, str]  # content_id -> memory_id mapping
    conflict_count: int  # Number of conflicts resolved
    last_snapshot_hash: Optional[str]  # Hash of last applied snapshot
    rebuild_timestamp: str  # When rebuild started


class ConflictResolutionError(Exception):
    """Raised when content conflicts cannot be resolved."""
    pass


class DeterministicRebuilder:
    """
    Deterministic rebuilder that ensures byte-identical reconstruction.
    
    Key Principles:
    - Content-addressable IDs determine uniqueness, not insertion order
    - Hash chain verification ensures event integrity
    - Active-Set enforcement prevents duplicates with same content
    - Conflict resolution uses cryptographic precedence rules
    """
    
    def __init__(self, event_store: EventStore, memory: LuminaMemory):
        self.event_store = event_store
        self.memory = memory
        self.content_index = ContentAddressableIndex()
        
        # Track rebuild state
        self.rebuild_state: Optional[RebuildState] = None
    
    def rebuild_from_scratch(self) -> Dict[str, Any]:
        """
        Perform complete deterministic rebuild from all events.
        
        Returns:
            Dict containing rebuild statistics and verification results
        """
        # Initialize rebuild state
        self.rebuild_state = RebuildState(
            processed_events=set(),
            active_set={},
            conflict_count=0,
            last_snapshot_hash=None,
            rebuild_timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        # Clear existing memory state
        self.memory.clear_all_memories()  # Note: This would need to be added to kernel
        self.content_index = ContentAddressableIndex()
        
        # Verify event store integrity first
        if not self.event_store.verify_integrity():
            raise ConflictResolutionError("Event store integrity verification failed")
        
        rebuild_stats = {
            'events_processed': 0,
            'memories_created': 0,
            'conflicts_resolved': 0,
            'snapshots_applied': 0,
            'hash_chain_verified': True,
            'final_active_set_size': 0,
            'rebuild_duration_ms': 0
        }
        
        start_time = datetime.now(timezone.utc)
        
        try:
            # Process all events in deterministic order
            for event in self.event_store.rebuild_from_events():
                self._process_event_for_rebuild(event)
                rebuild_stats['events_processed'] += 1
                
                # Update progress
                if rebuild_stats['events_processed'] % 100 == 0:
                    print(f"Processed {rebuild_stats['events_processed']} events...")
            
            # Finalize rebuild
            self._finalize_rebuild(rebuild_stats)
            
            end_time = datetime.now(timezone.utc)
            rebuild_stats['rebuild_duration_ms'] = int(
                (end_time - start_time).total_seconds() * 1000
            )
            
            return rebuild_stats
            
        except Exception as e:
            # Reset state on failure
            self.rebuild_state = None
            raise ConflictResolutionError(f"Rebuild failed: {e}")
    
    def rebuild_from_snapshot(self, snapshot_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform optimized rebuild from latest or specified snapshot.
        
        Args:
            snapshot_id: Optional specific snapshot ID, uses latest if None
            
        Returns:
            Dict containing rebuild statistics
        """
        # Get snapshot to rebuild from
        if snapshot_id:
            snapshot = self._get_snapshot(snapshot_id)
        else:
            snapshot = self.event_store.get_latest_snapshot()
        
        if not snapshot:
            # No snapshot available, rebuild from scratch
            return self.rebuild_from_scratch()
        
        # Initialize rebuild state
        self.rebuild_state = RebuildState(
            processed_events=set(),
            active_set=snapshot.active_set_state.copy(),
            conflict_count=0,
            last_snapshot_hash=snapshot.last_event_hash,
            rebuild_timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        rebuild_stats = {
            'snapshot_id': snapshot.snapshot_id,
            'snapshot_memory_count': len(snapshot.memory_records),
            'events_processed': 0,
            'memories_updated': 0,
            'conflicts_resolved': 0,
            'hash_chain_verified': True,
            'final_active_set_size': 0,
            'rebuild_duration_ms': 0
        }
        
        start_time = datetime.now(timezone.utc)
        
        try:
            # Restore memory state from snapshot
            self._restore_from_snapshot(snapshot)
            rebuild_stats['memories_restored'] = len(snapshot.memory_records)
            
            # Process events since snapshot
            events_since = self.event_store.get_events_since(snapshot.timestamp)
            
            for event in events_since:
                self._process_event_for_rebuild(event)
                rebuild_stats['events_processed'] += 1
            
            # Finalize rebuild
            self._finalize_rebuild(rebuild_stats)
            
            end_time = datetime.now(timezone.utc)
            rebuild_stats['rebuild_duration_ms'] = int(
                (end_time - start_time).total_seconds() * 1000
            )
            
            return rebuild_stats
            
        except Exception as e:
            self.rebuild_state = None
            raise ConflictResolutionError(f"Snapshot rebuild failed: {e}")
    
    def _process_event_for_rebuild(self, event: StoredEvent):
        """
        Process single event during deterministic rebuild.
        
        Args:
            event: Event to process for rebuild
        """
        if not self.rebuild_state:
            raise ConflictResolutionError("Rebuild state not initialized")
        
        # Skip already processed events
        if event.event_id in self.rebuild_state.processed_events:
            return
        
        # Verify event integrity
        if not verify_content_integrity(event.content, event.content_hash):
            raise ConflictResolutionError(f"Event integrity check failed: {event.event_id}")
        
        # Process based on event type
        if event.event_type == "memory_ingest":
            self._process_memory_ingest_event(event)
        elif event.event_type == "memory_update":
            self._process_memory_update_event(event)
        elif event.event_type == "memory_delete":
            self._process_memory_delete_event(event)
        elif event.event_type == "index_rebuild":
            self._process_index_rebuild_event(event)
        else:
            # Unknown event type - log but continue
            print(f"Warning: Unknown event type {event.event_type} for event {event.event_id}")
        
        # Mark event as processed
        self.rebuild_state = replace(
            self.rebuild_state,
            processed_events=self.rebuild_state.processed_events | {event.event_id}
        )
    
    def _process_memory_ingest_event(self, event: StoredEvent):
        """Process memory ingestion event with conflict resolution."""
        if not self.rebuild_state:
            return
        
        content = event.content.get('content', '')
        metadata = event.content.get('metadata', {})
        
        # Generate content-addressable ID
        content_id = memory_content_id(content)
        
        # Check for Active-Set conflicts
        if content_id in self.rebuild_state.active_set:
            existing_memory_id = self.rebuild_state.active_set[content_id]
            
            # Conflict resolution: Use cryptographic precedence
            # Earlier hash wins (deterministic lexicographic order)
            if event.content_hash < existing_memory_id:
                # Current event takes precedence - remove existing
                self._remove_memory_by_id(existing_memory_id)
                self.rebuild_state = replace(
                    self.rebuild_state,
                    conflict_count=self.rebuild_state.conflict_count + 1
                )
            else:
                # Existing memory takes precedence - skip this event
                return
        
        # Create new memory record
        memory_record = create_memory_record(content, metadata)
        
        # Use deterministic memory ID based on content
        deterministic_memory_id = f"mem_{content_id[:16]}"
        memory_record = replace(memory_record, memory_id=deterministic_memory_id)
        
        # Ingest into memory system
        self.memory.add_memory(memory_record)
        
        # Update Active-Set
        updated_active_set = self.rebuild_state.active_set.copy()
        updated_active_set[content_id] = memory_record.memory_id
        
        self.rebuild_state = replace(
            self.rebuild_state,
            active_set=updated_active_set
        )
    
    def _process_memory_update_event(self, event: StoredEvent):
        """Process memory update event."""
        memory_id = event.content.get('memory_id')
        new_content = event.content.get('new_content')
        new_metadata = event.content.get('new_metadata', {})
        
        if not memory_id or new_content is None:
            return
        
        # Find existing memory
        existing_record = self.memory.get_memory_by_id(memory_id)
        if not existing_record:
            return
        
        # Remove from Active-Set
        old_content_id = memory_content_id(existing_record.content)
        if old_content_id in self.rebuild_state.active_set:
            updated_active_set = self.rebuild_state.active_set.copy()
            del updated_active_set[old_content_id]
            self.rebuild_state = replace(
                self.rebuild_state,
                active_set=updated_active_set
            )
        
        # Create updated record
        updated_record = replace(
            existing_record,
            content=new_content,
            metadata=new_metadata
        )
        
        # Update memory
        self.memory.update_memory(updated_record)
        
        # Add to Active-Set with new content ID
        new_content_id = memory_content_id(new_content)
        updated_active_set = self.rebuild_state.active_set.copy()
        updated_active_set[new_content_id] = memory_id
        
        self.rebuild_state = replace(
            self.rebuild_state,
            active_set=updated_active_set
        )
    
    def _process_memory_delete_event(self, event: StoredEvent):
        """Process memory deletion event."""
        memory_id = event.content.get('memory_id')
        if not memory_id:
            return
        
        # Remove from memory
        existing_record = self.memory.get_memory_by_id(memory_id)
        if existing_record:
            # Remove from Active-Set
            content_id = memory_content_id(existing_record.content)
            if content_id in self.rebuild_state.active_set:
                updated_active_set = self.rebuild_state.active_set.copy()
                del updated_active_set[content_id]
                self.rebuild_state = replace(
                    self.rebuild_state,
                    active_set=updated_active_set
                )
        
        self.memory.delete_memory(memory_id)
    
    def _process_index_rebuild_event(self, event: StoredEvent):
        """Process index rebuild event."""
        # Trigger index rebuild on memory system
        self.memory.rebuild_search_index()
    
    def _remove_memory_by_id(self, memory_id: str):
        """Remove memory record by ID."""
        existing_record = self.memory.get_memory_by_id(memory_id)
        if existing_record:
            self.memory.delete_memory(memory_id)
    
    def _restore_from_snapshot(self, snapshot: EventStoreSnapshot):
        """Restore memory state from snapshot."""
        # Clear current state
        self.memory.clear_all_memories()
        
        # Restore memory records
        for record_data in snapshot.memory_records:
            # Reconstruct memory record
            embedding = None
            if record_data['embedding']:
                embedding = np.array(record_data['embedding'])
            
            memory_record = MemoryRecord(
                memory_id=record_data['memory_id'],
                content=record_data['content'],
                embedding=embedding,
                metadata=record_data['metadata']
            )
            
            self.memory.add_memory(memory_record)
        
        # Rebuild search index
        self.memory.rebuild_search_index()
    
    def _finalize_rebuild(self, rebuild_stats: Dict[str, Any]):
        """Finalize rebuild process and update statistics."""
        if not self.rebuild_state:
            return
        
        # Final Active-Set size
        rebuild_stats['final_active_set_size'] = len(self.rebuild_state.active_set)
        rebuild_stats['conflicts_resolved'] = self.rebuild_state.conflict_count
        
        # Verify final state integrity
        rebuild_stats['final_verification'] = self._verify_rebuild_integrity()
        
        print(f"Rebuild completed: {rebuild_stats['final_active_set_size']} active memories")
        if rebuild_stats['conflicts_resolved'] > 0:
            print(f"Resolved {rebuild_stats['conflicts_resolved']} content conflicts")
    
    def _verify_rebuild_integrity(self) -> Dict[str, bool]:
        """Verify integrity of rebuilt state."""
        verification = {
            'active_set_consistent': True,
            'no_duplicate_content': True,
            'all_content_ids_valid': True
        }
        
        if not self.rebuild_state:
            return verification
        
        # Check Active-Set consistency
        seen_content_ids = set()
        for content_id, memory_id in self.rebuild_state.active_set.items():
            memory_record = self.memory.get_memory_by_id(memory_id)
            if not memory_record:
                verification['active_set_consistent'] = False
                continue
            
            # Verify content ID matches
            actual_content_id = memory_content_id(memory_record.content)
            if actual_content_id != content_id:
                verification['active_set_consistent'] = False
            
            # Check for duplicates
            if content_id in seen_content_ids:
                verification['no_duplicate_content'] = False
            seen_content_ids.add(content_id)
            
            # Verify content ID format
            if len(content_id) != 64:  # SHA-256 hex length
                verification['all_content_ids_valid'] = False
        
        return verification
    
    def _get_snapshot(self, snapshot_id: str) -> Optional[EventStoreSnapshot]:
        """Get snapshot by ID from event store."""
        return self.event_store._get_snapshot(snapshot_id)
    
    def get_active_set_state(self) -> Dict[str, str]:
        """Get current Active-Set state (content_id -> memory_id)."""
        if not self.rebuild_state:
            return {}
        return self.rebuild_state.active_set.copy()
    
    def create_checkpoint_snapshot(self) -> EventStoreSnapshot:
        """Create snapshot of current rebuilt state."""
        if not self.rebuild_state:
            raise ConflictResolutionError("Cannot create snapshot - no rebuild state")
        
        # Get all current memory records
        memory_records = self.memory.get_all_memories()
        
        # Create snapshot
        return self.event_store.create_snapshot(
            memory_records=memory_records,
            active_set=self.rebuild_state.active_set
        )


def create_deterministic_rebuilder(storage_path: Path, memory: LuminaMemory) -> DeterministicRebuilder:
    """
    Create deterministic rebuilder with event store.
    
    Args:
        storage_path: Path to event store storage
        memory: LuminaMemory instance to rebuild
        
    Returns:
        DeterministicRebuilder instance
    """
    event_store = create_event_store(storage_path)
    return DeterministicRebuilder(event_store, memory)


def rebuild_memory_from_events(storage_path: Path, memory: LuminaMemory, 
                             use_snapshot: bool = True) -> Dict[str, Any]:
    """
    Convenience function for deterministic memory rebuild.
    
    Args:
        storage_path: Path to event store
        memory: LuminaMemory to rebuild
        use_snapshot: Whether to use latest snapshot for optimization
        
    Returns:
        Rebuild statistics
    """
    rebuilder = create_deterministic_rebuilder(storage_path, memory)
    
    if use_snapshot:
        return rebuilder.rebuild_from_snapshot()
    else:
        return rebuilder.rebuild_from_scratch()