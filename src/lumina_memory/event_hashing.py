"""
Event Hashing and Hash Chain Verification for Lumina Memory

This module provides cryptographic integrity for the event log through:
- Event content hashing with SHA-256
- Hash chain linking for tamper detection  
- Event sequence verification
- Merkle tree construction for efficient verification

Design Principles:
- Immutable hash chain: Each event links to previous
- Tamper detection: Any modification breaks the chain
- Efficient verification: Merkle trees for batch validation
- Forward security: Future events cannot be forged
"""

import hashlib
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time


@dataclass(frozen=True)
class HashChainEntry:
    """Entry in the hash chain with cryptographic linking."""
    event_id: str              # Unique event identifier
    event_hash: str            # SHA-256 of event content
    previous_hash: str         # Hash of previous event in chain
    chain_hash: str            # Combined hash linking to chain
    sequence_number: int       # Position in chain
    timestamp: float           # When hash was computed
    
    def verify_chain_link(self, previous_entry: Optional['HashChainEntry']) -> bool:
        """Verify this entry properly links to previous in chain."""
        if previous_entry is None:
            # Genesis entry
            return self.previous_hash == "0" * 64 and self.sequence_number == 0
        
        # Verify previous hash matches
        if self.previous_hash != previous_entry.chain_hash:
            return False
            
        # Verify sequence number
        if self.sequence_number != previous_entry.sequence_number + 1:
            return False
            
        return True
    
    def compute_chain_hash(self) -> str:
        """Compute the chain hash for this entry."""
        chain_data = f"{self.event_hash}:{self.previous_hash}:{self.sequence_number}"
        return hashlib.sha256(chain_data.encode('utf-8')).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify the integrity of this hash chain entry."""
        expected_chain_hash = self.compute_chain_hash()
        return self.chain_hash == expected_chain_hash


def compute_event_hash(event_data: Dict[str, Any]) -> str:
    """
    Compute SHA-256 hash of event content for integrity verification.
    
    Uses canonical JSON representation for deterministic hashing.
    """
    # Create deterministic event representation
    hash_content = {
        'event_type': event_data.get('event_type'),
        'timestamp': event_data.get('timestamp'),
        'sequence': event_data.get('sequence'),
        'payload': event_data.get('payload', {}),
        'metadata': event_data.get('metadata', {}),
        'schema_version': event_data.get('schema_version', 'v1.0')
    }
    
    # Convert to canonical JSON
    json_str = json.dumps(hash_content, sort_keys=True, separators=(',', ':'))
    
    # Compute SHA-256
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()


def create_genesis_entry() -> HashChainEntry:
    """Create the genesis (first) entry in the hash chain."""
    genesis_event = {
        'event_type': 'genesis',
        'timestamp': time.time(),
        'sequence': 0,
        'payload': {'message': 'Genesis block for Lumina Memory event log'},
        'metadata': {'system': 'lumina_memory'},
        'schema_version': 'v1.0'
    }
    
    event_hash = compute_event_hash(genesis_event)
    previous_hash = "0" * 64  # No previous hash for genesis
    
    entry = HashChainEntry(
        event_id="genesis",
        event_hash=event_hash,
        previous_hash=previous_hash,
        chain_hash="",  # Will be computed
        sequence_number=0,
        timestamp=genesis_event['timestamp']
    )
    
    # Compute and update chain hash
    chain_hash = entry.compute_chain_hash()
    return HashChainEntry(
        event_id=entry.event_id,
        event_hash=entry.event_hash,
        previous_hash=entry.previous_hash,
        chain_hash=chain_hash,
        sequence_number=entry.sequence_number,
        timestamp=entry.timestamp
    )


def add_event_to_chain(
    event_data: Dict[str, Any],
    previous_entry: HashChainEntry
) -> HashChainEntry:
    """
    Add new event to hash chain with proper linking.
    
    Args:
        event_data: Event to add to chain
        previous_entry: Previous entry in the chain
        
    Returns:
        New hash chain entry
    """
    event_hash = compute_event_hash(event_data)
    
    entry = HashChainEntry(
        event_id=event_data.get('event_id', 'unknown'),
        event_hash=event_hash,
        previous_hash=previous_entry.chain_hash,
        chain_hash="",  # Will be computed
        sequence_number=previous_entry.sequence_number + 1,
        timestamp=event_data.get('timestamp', time.time())
    )
    
    # Compute chain hash
    chain_hash = entry.compute_chain_hash()
    return HashChainEntry(
        event_id=entry.event_id,
        event_hash=entry.event_hash,
        previous_hash=entry.previous_hash,
        chain_hash=chain_hash,
        sequence_number=entry.sequence_number,
        timestamp=entry.timestamp
    )


def verify_hash_chain(chain: List[HashChainEntry]) -> Tuple[bool, Optional[str]]:
    """
    Verify integrity of entire hash chain.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not chain:
        return False, "Empty chain"
    
    # Verify genesis entry
    if chain[0].sequence_number != 0:
        return False, "First entry is not genesis (sequence != 0)"
    
    if not chain[0].verify_integrity():
        return False, f"Genesis entry integrity verification failed"
    
    # Verify each subsequent entry
    for i in range(1, len(chain)):
        current = chain[i]
        previous = chain[i-1]
        
        # Verify integrity of current entry
        if not current.verify_integrity():
            return False, f"Entry {i} integrity verification failed"
        
        # Verify chain linking
        if not current.verify_chain_link(previous):
            return False, f"Entry {i} chain link verification failed"
    
    return True, None


class MerkleTree:
    """
    Merkle tree for efficient batch verification of events.
    
    Enables proving that specific events are included in a set
    without revealing the entire set.
    """
    
    def __init__(self, event_hashes: List[str]):
        """Build Merkle tree from list of event hashes."""
        if not event_hashes:
            raise ValueError("Cannot build Merkle tree from empty list")
        
        self.leaves = event_hashes[:]
        self.tree = self._build_tree(event_hashes)
        self.root_hash = self.tree[0] if self.tree else None
    
    def _build_tree(self, hashes: List[str]) -> List[str]:
        """Build Merkle tree bottom-up."""
        if len(hashes) == 1:
            return hashes
        
        # Ensure even number of hashes (duplicate last if odd)
        if len(hashes) % 2 == 1:
            hashes = hashes + [hashes[-1]]
        
        # Build next level
        next_level = []
        for i in range(0, len(hashes), 2):
            combined = hashes[i] + hashes[i+1]
            parent_hash = hashlib.sha256(combined.encode('utf-8')).hexdigest()
            next_level.append(parent_hash)
        
        # Recursively build tree
        upper_tree = self._build_tree(next_level)
        return upper_tree + next_level + hashes
    
    def get_proof(self, leaf_hash: str) -> Optional[List[str]]:
        """
        Get Merkle proof for a specific leaf.
        
        Returns list of sibling hashes needed to verify inclusion.
        """
        if leaf_hash not in self.leaves:
            return None
        
        # Find leaf index
        leaf_index = self.leaves.index(leaf_hash)
        
        # Generate proof path
        proof = []
        current_index = leaf_index
        current_level_size = len(self.leaves)
        
        # Work up the tree
        tree_offset = len(self.tree) - len(self.leaves)
        
        while current_level_size > 1:
            # Ensure even number of nodes
            if current_level_size % 2 == 1:
                current_level_size += 1
            
            # Find sibling
            if current_index % 2 == 0:
                sibling_index = current_index + 1
            else:
                sibling_index = current_index - 1
            
            # Get sibling hash from tree
            if sibling_index < current_level_size:
                sibling_hash = self.tree[tree_offset + sibling_index]
                proof.append(sibling_hash)
            
            # Move to parent level
            current_index = current_index // 2
            current_level_size = current_level_size // 2
            tree_offset -= current_level_size
        
        return proof
    
    def verify_proof(self, leaf_hash: str, proof: List[str]) -> bool:
        """
        Verify Merkle proof for leaf inclusion.
        
        Args:
            leaf_hash: Hash of the leaf to verify
            proof: List of sibling hashes for proof path
            
        Returns:
            True if proof is valid, False otherwise
        """
        current_hash = leaf_hash
        
        for sibling_hash in proof:
            # Combine hashes (order matters for deterministic results)
            if current_hash < sibling_hash:
                combined = current_hash + sibling_hash
            else:
                combined = sibling_hash + current_hash
            
            current_hash = hashlib.sha256(combined.encode('utf-8')).hexdigest()
        
        return current_hash == self.root_hash


class HashChainManager:
    """
    Manager for maintaining and verifying event hash chains.
    
    Provides high-level interface for hash chain operations.
    """
    
    def __init__(self):
        self.chain: List[HashChainEntry] = []
        self._initialize_genesis()
    
    def _initialize_genesis(self):
        """Initialize chain with genesis entry."""
        genesis = create_genesis_entry()
        self.chain.append(genesis)
    
    def add_event(self, event_data: Dict[str, Any]) -> HashChainEntry:
        """Add event to chain and return new entry."""
        if not self.chain:
            raise ValueError("Chain not initialized")
        
        previous_entry = self.chain[-1]
        new_entry = add_event_to_chain(event_data, previous_entry)
        self.chain.append(new_entry)
        
        return new_entry
    
    def verify_integrity(self) -> Tuple[bool, Optional[str]]:
        """Verify integrity of entire chain."""
        return verify_hash_chain(self.chain)
    
    def get_chain_head(self) -> Optional[HashChainEntry]:
        """Get the latest entry in the chain."""
        return self.chain[-1] if self.chain else None
    
    def get_merkle_tree(self) -> MerkleTree:
        """Build Merkle tree from all events in chain."""
        event_hashes = [entry.event_hash for entry in self.chain]
        return MerkleTree(event_hashes)
    
    def verify_event_inclusion(self, event_hash: str) -> bool:
        """Verify that an event is included in the chain."""
        return any(entry.event_hash == event_hash for entry in self.chain)
    
    def get_chain_stats(self) -> Dict[str, Any]:
        """Get statistics about the hash chain."""
        return {
            'total_events': len(self.chain),
            'genesis_hash': self.chain[0].chain_hash if self.chain else None,
            'head_hash': self.chain[-1].chain_hash if self.chain else None,
            'chain_length': len(self.chain) - 1,  # Exclude genesis
            'integrity_valid': self.verify_integrity()[0]
        }


# Global hash chain manager
global_hash_chain = HashChainManager()
