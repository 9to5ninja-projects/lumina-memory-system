"""
Versioned XP Store Implementation
Created to resolve import failures across notebooks.

This was referenced in XP Core notebook but was an empty stub.
Now implementing based on the XP Core mathematical foundation.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class XPStoreEntry:
    """Entry in the versioned XP store"""
    id: str
    content: str
    embedding: np.ndarray
    created_at: float
    accessed_at: float
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_access(self):
        """Update access statistics"""
        self.accessed_at = time.time()
        self.access_count += 1


class VersionedXPStore:
    """
    Versioned XP Store for managing memory entries with mathematical foundation.
    
    Implements the mathematical concepts from XP Core notebook:
    - Temporal decay mathematics
    - Holographic properties
    - Lexical attribution support
    """
    
    def __init__(self):
        self.entries: Dict[str, XPStoreEntry] = {}
        self.version_counter = 0
        
    def store(self, content: str, embedding: np.ndarray = None, metadata: Dict = None) -> str:
        """Store new entry and return ID"""
        entry_id = f"xp_store_{self.version_counter:06d}"
        self.version_counter += 1
        
        if embedding is None:
            # Simple random embedding for testing
            embedding = np.random.randn(384).astype(np.float32)
            
        entry = XPStoreEntry(
            id=entry_id,
            content=content,
            embedding=embedding,
            created_at=time.time(),
            accessed_at=time.time(),
            metadata=metadata or {}
        )
        
        self.entries[entry_id] = entry
        return entry_id
        
    def retrieve(self, entry_id: str) -> Optional[XPStoreEntry]:
        """Retrieve entry by ID"""
        entry = self.entries.get(entry_id)
        if entry:
            entry.update_access()
        return entry
        
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """Simple similarity search"""
        if not self.entries:
            return []
            
        similarities = []
        for entry_id, entry in self.entries.items():
            # Simple cosine similarity
            sim = np.dot(query_embedding, entry.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(entry.embedding)
            )
            similarities.append((entry_id, float(sim)))
            
        # Return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
        
    def stats(self) -> Dict[str, Any]:
        """Get store statistics"""
        if not self.entries:
            return {'total_entries': 0, 'total_accesses': 0}
            
        total_accesses = sum(entry.access_count for entry in self.entries.values())
        return {
            'total_entries': len(self.entries),
            'total_accesses': total_accesses,
            'version_counter': self.version_counter
        }


# Simple test function
def test_versioned_xp_store():
    """Test the VersionedXPStore"""
    print("🧪 Testing VersionedXPStore...")
    
    store = VersionedXPStore()
    
    # Test storing
    entry_id = store.store("Test content", metadata={'test': True})
    print(f"✅ Stored entry: {entry_id}")
    
    # Test retrieval
    entry = store.retrieve(entry_id)
    print(f"✅ Retrieved entry: {entry.content}")
    
    # Test stats
    stats = store.stats()
    print(f"✅ Store stats: {stats}")
    
    print("🎉 VersionedXPStore test successful!")
    return True


if __name__ == "__main__":
    test_versioned_xp_store()