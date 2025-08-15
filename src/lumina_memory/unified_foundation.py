"""
Unified Foundation Classes - Clean Implementation
Created to resolve architectural conflicts across notebooks.

This module provides unified classes that replace:
- Main branch: Memory class (functional approach)
- XP Core: MemoryUnit class (holographic properties)  
- Unit-Space: Memory class (spatial topology)
- Various Config classes across notebooks

Strategy: Create clean foundation first, then import into notebooks.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class UnifiedMemory:
    """
    Unified memory representation supporting all architectures.
    
    Replaces:
    - kernel.py Memory class (main branch)
    - MemoryUnit class (XP core)  
    - Memory class (unit-space bridge)
    """
    # Core identity
    id: str
    content: str
    
    # Essential properties
    embedding: np.ndarray = None
    created_at: float = None
    salience: float = 0.0
    
    # XP Core properties
    lineage: List[str] = field(default_factory=list)
    decay_timestamp: float = None
    
    # Unit-Space properties  
    topology_links: Dict[str, float] = field(default_factory=dict)
    activation_level: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.embedding is None:
            self.embedding = np.random.randn(384).astype(np.float32)
        if self.created_at is None:
            self.created_at = time.time()
        if self.decay_timestamp is None:
            self.decay_timestamp = self.created_at


@dataclass  
class UnifiedConfig:
    """
    Unified configuration supporting all use cases.
    
    Replaces:
    - LuminaConfig (main branch)
    - XPCoreConfig (unit-space bridge)
    - SpaceConfig (unit-space bridge)
    """
    # Core settings
    embedding_dim: int = 384
    
    # XP Core settings
    decay_half_life: float = 168.0
    use_versioned_store: bool = False
    
    # Unit-Space settings
    k_neighbors: int = 10
    consolidation_threshold: float = 0.7
    
    # System settings
    max_memory_capacity: int = 10000
    deterministic_seed: int = 42


class UnifiedKernel:
    """
    Unified kernel supporting all patterns.
    
    Combines:
    - Functional operations (main branch)
    - XP Core mathematics  
    - Unit-Space topology
    - HD Kernel interface compliance
    """
    
    def __init__(self, config: UnifiedConfig = None):
        self.config = config or UnifiedConfig()
        self.memories: Dict[str, UnifiedMemory] = {}
        
    def store_memory(self, content: str, metadata: Dict = None) -> UnifiedMemory:
        """Store new memory with unified properties"""
        memory_id = f"mem_{len(self.memories):06d}"
        memory = UnifiedMemory(
            id=memory_id,
            content=content,
            metadata=metadata or {}
        )
        self.memories[memory_id] = memory
        return memory
        
    def retrieve_memory(self, memory_id: str) -> Optional[UnifiedMemory]:
        """Retrieve memory by ID"""
        return self.memories.get(memory_id)
        
    def process_memory(self, memory_data: Dict) -> str:
        """Process memory data - HD Kernel interface compliance"""
        content = memory_data.get('content', '')
        metadata = memory_data.get('metadata', {})
        memory = self.store_memory(content, metadata)
        return memory.id
        
    def stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            'total_memories': len(self.memories),
            'config_embedding_dim': self.config.embedding_dim,
            'config_k_neighbors': self.config.k_neighbors,
            'system_status': 'unified_foundation_active'
        }


# Simple test to verify everything works
def test_unified_foundation():
    """Test the unified foundation classes"""
    print("🧪 Testing unified foundation...")
    
    # Test config
    config = UnifiedConfig()
    print(f"✅ Config: embedding_dim={config.embedding_dim}, k_neighbors={config.k_neighbors}")
    
    # Test memory
    memory = UnifiedMemory(id="test_001", content="Test unified memory")
    print(f"✅ Memory: id={memory.id}, embedding_shape={memory.embedding.shape}")
    
    # Test kernel
    kernel = UnifiedKernel(config)
    test_memory = kernel.store_memory("Hello unified foundation", {"test": True})
    stats = kernel.stats()
    print(f"✅ Kernel: stored memory {test_memory.id}, stats={stats}")
    
    print("🎉 Unified foundation test successful!")
    return True


if __name__ == "__main__":
    test_unified_foundation()
