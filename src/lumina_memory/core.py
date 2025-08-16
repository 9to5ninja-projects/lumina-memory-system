"""Core data structures for Lumina Memory System."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class QueryType(Enum):
    """Types of memory queries."""
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    METADATA = "metadata"
    HYBRID = "hybrid"


@dataclass
class MemoryEntry:
    """Core memory entry with metadata."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    importance_score: float = 0.0
    
    def __post_init__(self):
        """Validate entry after initialization."""
        if not self.content.strip():
            raise ValueError("Memory content cannot be empty")
        if not (0.0 <= self.importance_score <= 1.0):
            raise ValueError("Importance score must be between 0.0 and 1.0")


@dataclass  
class QueryResult:
    """Result from a memory query."""
    
    entry: MemoryEntry
    similarity_score: float
    retrieval_context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate query result."""
        if not (0.0 <= self.similarity_score <= 1.0):
            raise ValueError("Similarity score must be between 0.0 and 1.0")


class LuminaError(Exception):
    """Base exception for Lumina Memory System."""
    pass


class ConfigurationError(LuminaError):
    """Configuration-related errors."""
    pass


class EmbeddingError(LuminaError):
    """Embedding generation errors.""" 
    pass


class StorageError(LuminaError):
    """Vector storage errors."""
    pass


class QueryError(LuminaError):
    """Query execution errors."""
    pass


class EmbeddingVersionError(LuminaError):`n    """Embedding version compatibility errors."""`n    pass`n`n`nclass MemoryError(LuminaError):
    """Memory management errors."""
    pass
