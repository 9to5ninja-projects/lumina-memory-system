"""
Lumina Memory: Holographic Memory System for AI Applications

A production-ready memory system with pure functional kernel, event sourcing,
and mathematical guarantees for deterministic behavior.
"""

__version__ = "0.2.0-alpha"  # M2: Kernel module complete
__author__ = "Lumina Team"
__email__ = "dev@lumina.ai"

try:
    # Core kernel (pure functional)
    from .kernel import Memory, superpose, reinforce, decay, forget
    from .kernel import SALIENCE_REINFORCE_CAP, DEFAULT_HALF_LIFE
    
    # Legacy system components (will be refactored to use kernel)
    from .config import LuminaConfig
    from .memory_system import MemorySystem
    from .core import MemoryEntry, QueryResult
    from .embeddings import EmbeddingProvider
    from .vector_store import VectorStore
    from .utils import setup_logging, normalize_similarity

    # Main API exports
    __all__ = [
        # Pure functional kernel
        "Memory",
        "superpose", 
        "reinforce",
        "decay",
        "forget",
        "SALIENCE_REINFORCE_CAP",
        "DEFAULT_HALF_LIFE",
        
        # Legacy system (to be refactored)
        "MemorySystem",
        "LuminaConfig", 
        "MemoryEntry",
        "QueryResult",
        "EmbeddingProvider",
        "VectorStore",
        "setup_logging",
        "normalize_similarity",
    ]

    # Configure default logging (legacy)
    setup_logging()

except ImportError as e:
    # Graceful degradation for CI
    print(f"Warning: Some imports failed in lumina_memory.__init__: {e}")
    __all__ = []
