"""Vector storage implementations for Lumina Memory System."""

import logging
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .core import MemoryEntry, StorageError

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    """Abstract base class for vector storage."""
    
    @abstractmethod
    def add(self, entries: List[MemoryEntry]) -> None:
        """Add memory entries to the store."""
        pass
    
    @abstractmethod 
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar entries."""
        pass
    
    @abstractmethod
    def remove(self, entry_ids: List[str]) -> None:
        """Remove entries by ID."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all entries."""
        pass
    
    @property
    @abstractmethod
    def size(self) -> int:
        """Get number of entries."""
        pass


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store implementation."""
    
    def __init__(self, dimension: int, metric: str = "cosine"):
        """Initialize FAISS vector store."""
        self.dimension = dimension
        self.metric = metric
        self.index = None
        self.entry_map: Dict[int, str] = {}  # index_id -> entry_id
        self.reverse_map: Dict[str, int] = {}  # entry_id -> index_id
        self.next_id = 0
        self._lock = threading.Lock()
        
        self._create_index()
        logger.info(f"FAISS store initialized: dim={dimension}, metric={metric}")
    
    def _create_index(self):
        """Create FAISS index based on metric."""
        try:
            import faiss
            
            if self.metric == "cosine":
                self.index = faiss.IndexFlatIP(self.dimension)
            elif self.metric == "euclidean": 
                self.index = faiss.IndexFlatL2(self.dimension)
            elif self.metric == "inner_product":
                self.index = faiss.IndexFlatIP(self.dimension)
            else:
                raise ValueError(f"Unsupported metric: {self.metric}")
                
        except ImportError as e:
            raise StorageError(f"FAISS not available: {e}")
        except Exception as e:
            raise StorageError(f"Failed to create FAISS index: {e}")
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity."""
        if self.metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            return embeddings / norms
        return embeddings
    
    def add(self, entries: List[MemoryEntry]) -> None:
        """Add memory entries to the store."""
        if not entries:
            return
            
        embeddings = []
        entry_ids = []
        
        for entry in entries:
            if entry.embedding is None:
                raise StorageError(f"Entry {entry.id} has no embedding")
            if entry.embedding.shape[0] != self.dimension:
                raise StorageError(f"Entry {entry.id} embedding dimension mismatch")
            embeddings.append(entry.embedding)
            entry_ids.append(entry.id)
        
        embeddings = np.array(embeddings).astype("float32")
        embeddings = self._normalize_embeddings(embeddings)
        
        with self._lock:
            try:
                self.index.add(embeddings)
                
                for i, entry_id in enumerate(entry_ids):
                    index_id = self.next_id + i
                    self.entry_map[index_id] = entry_id
                    self.reverse_map[entry_id] = index_id
                
                self.next_id += len(entries)
                logger.info(f"Added {len(entries)} entries to FAISS store")
                
            except Exception as e:
                raise StorageError(f"Failed to add entries to FAISS: {e}")
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar entries."""
        if query_embedding.shape[0] != self.dimension:
            raise StorageError("Query embedding dimension mismatch")
            
        if self.index.ntotal == 0:
            return []
        
        query = query_embedding.reshape(1, -1).astype("float32")
        query = self._normalize_embeddings(query)
        
        k = min(k, self.index.ntotal)
        
        try:
            with self._lock:
                scores, indices = self.index.search(query, k)
                
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx != -1 and idx in self.entry_map:
                        entry_id = self.entry_map[idx]
                        similarity = float(score)
                        if self.metric == "euclidean":
                            similarity = 1.0 / (1.0 + score)
                        results.append((entry_id, similarity))
                
                return results
                
        except Exception as e:
            raise StorageError(f"FAISS search failed: {e}")
    
    def remove(self, entry_ids: List[str]) -> None:
        """Remove entries by ID."""
        with self._lock:
            for entry_id in entry_ids:
                if entry_id in self.reverse_map:
                    index_id = self.reverse_map[entry_id]
                    del self.entry_map[index_id]
                    del self.reverse_map[entry_id]
    
    def clear(self) -> None:
        """Clear all entries.""" 
        with self._lock:
            self._create_index()
            self.entry_map.clear()
            self.reverse_map.clear()
            self.next_id = 0
    
    @property
    def size(self) -> int:
        """Get number of entries."""
        return len(self.entry_map)


class InMemoryVectorStore(VectorStore):
    """Simple in-memory vector store for testing."""
    
    def __init__(self, metric: str = "cosine"):
        """Initialize in-memory store."""
        self.metric = metric
        self.entries: Dict[str, MemoryEntry] = {}
        self._lock = threading.Lock()
    
    def add(self, entries: List[MemoryEntry]) -> None:
        """Add memory entries to the store."""
        with self._lock:
            for entry in entries:
                self.entries[entry.id] = entry
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar entries."""
        if not self.entries:
            return []
        
        similarities = []
        with self._lock:
            for entry_id, entry in self.entries.items():
                if entry.embedding is not None:
                    similarity = self._calculate_similarity(query_embedding, entry.embedding)
                    similarities.append((entry_id, similarity))
        
        # Sort by similarity (descending) and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def _calculate_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate similarity between two vectors."""
        if self.metric == "cosine":
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        elif self.metric == "euclidean":
            distance = np.linalg.norm(a - b)
            return 1.0 / (1.0 + distance)
        else:
            return float(np.dot(a, b))  # inner product
    
    def remove(self, entry_ids: List[str]) -> None:
        """Remove entries by ID."""
        with self._lock:
            for entry_id in entry_ids:
                self.entries.pop(entry_id, None)
    
    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self.entries.clear()
    
    @property
    def size(self) -> int:
        """Get number of entries."""
        return len(self.entries)
