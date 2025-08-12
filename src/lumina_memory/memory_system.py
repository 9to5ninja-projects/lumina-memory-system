"""Main Memory System for Lumina."""

import logging
import time
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from .config import LuminaConfig
from .core import MemoryEntry, QueryResult, QueryType, MemoryError
from .embeddings import EmbeddingProvider
from .vector_store import VectorStore
from .utils import normalize_similarity

logger = logging.getLogger(__name__)


class MemorySystem:
    """Main Lumina Memory System with clean API."""
    
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        config: Optional[LuminaConfig] = None,
    ):
        """Initialize memory system."""
        self.config = config or LuminaConfig()
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        
        # Memory stores
        self.stm: deque = deque(maxlen=self.config.stm_capacity)
        self.ltm: Dict[str, MemoryEntry] = {}
        
        # Statistics
        self.stats = {
            "total_memories": 0,
            "total_queries": 0,
            "total_ingestions": 0,
            "avg_query_time": 0.0,
            "memory_hits": 0,
            "memory_misses": 0,
        }
        
        logger.info("Memory system initialized")
    
    def ingest(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Ingest new content into memory.
        
        Args:
            content: Text content to remember
            metadata: Optional metadata dict
            
        Returns:
            Memory entry ID
        """
        try:
            # Generate embedding
            embedding = self.embedding_provider.embed_single(content)
            
            # Create memory entry
            entry = MemoryEntry(
                content=content,
                embedding=embedding,
                metadata=metadata or {},
                timestamp=datetime.now(),
            )
            
            # Add to short-term memory
            self.stm.append(entry)
            
            # Add to vector store
            self.vector_store.add([entry])
            
            # Update statistics
            self.stats["total_memories"] += 1
            self.stats["total_ingestions"] += 1
            
            logger.info(f"Ingested memory: {entry.id[:8]}...")
            return entry.id
            
        except Exception as e:
            logger.error(f"Failed to ingest memory: {e}")
            raise MemoryError(f"Ingestion failed: {e}")
    
    def recall(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        query_type: QueryType = QueryType.SEMANTIC,
    ) -> List[Dict[str, Any]]:
        """
        Recall relevant memories.
        
        Args:
            query: Query string
            k: Number of results to return
            filters: Optional metadata filters
            query_type: Type of query to perform
            
        Returns:
            List of memory results with content, similarity, and metadata
        """
        start_time = time.time()
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_provider.embed_single(query)
            
            # Search vector store
            search_results = self.vector_store.search(query_embedding, k=k * 2)
            
            # Get full memory entries and apply filters
            results = []
            for entry_id, raw_similarity in search_results:
                # Find entry in STM or LTM
                entry = self._find_entry(entry_id)
                if entry is None:
                    continue
                
                # Apply filters
                if filters and not self._matches_filters(entry, filters):
                    continue
                
                # Normalize similarity score
                similarity = normalize_similarity(raw_similarity, self.vector_store.metric)
                
                # Update access count
                entry.access_count += 1
                
                results.append({
                    "id": entry.id,
                    "content": entry.content,
                    "similarity": similarity,
                    "metadata": entry.metadata,
                    "timestamp": entry.timestamp.isoformat(),
                    "access_count": entry.access_count,
                })
                
                if len(results) >= k:
                    break
            
            # Update statistics
            query_time = time.time() - start_time
            self.stats["total_queries"] += 1
            self.stats["avg_query_time"] = (
                (self.stats["avg_query_time"] * (self.stats["total_queries"] - 1) + query_time)
                / self.stats["total_queries"]
            )
            
            if results:
                self.stats["memory_hits"] += 1
            else:
                self.stats["memory_misses"] += 1
            
            logger.info(f"Recall completed: {len(results)} results in {query_time:.3f}s")
            return results
            
        except Exception as e:
            logger.error(f"Recall failed: {e}")
            raise MemoryError(f"Recall failed: {e}")
    
    def consolidate(self) -> int:
        """
        Consolidate memories from STM to LTM.
        
        Returns:
            Number of memories consolidated
        """
        try:
            consolidated = 0
            
            # Move important memories from STM to LTM
            for entry in list(self.stm):
                if entry.importance_score >= self.config.consolidation_threshold:
                    self.ltm[entry.id] = entry
                    consolidated += 1
            
            logger.info(f"Consolidated {consolidated} memories to LTM")
            return consolidated
            
        except Exception as e:
            logger.error(f"Consolidation failed: {e}")
            raise MemoryError(f"Consolidation failed: {e}")
    
    def forget(self, entry_ids: List[str]) -> int:
        """
        Forget specific memories.
        
        Args:
            entry_ids: List of memory IDs to forget
            
        Returns:
            Number of memories forgotten
        """
        try:
            forgotten = 0
            
            # Remove from vector store
            self.vector_store.remove(entry_ids)
            
            # Remove from STM and LTM
            for entry_id in entry_ids:
                # Remove from STM
                self.stm = deque(
                    (entry for entry in self.stm if entry.id != entry_id),
                    maxlen=self.stm.maxlen
                )
                
                # Remove from LTM
                if entry_id in self.ltm:
                    del self.ltm[entry_id]
                    forgotten += 1
            
            # Update statistics
            self.stats["total_memories"] -= forgotten
            
            logger.info(f"Forgot {forgotten} memories")
            return forgotten
            
        except Exception as e:
            logger.error(f"Forget operation failed: {e}")
            raise MemoryError(f"Forget failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        current_stats = self.stats.copy()
        current_stats.update({
            "stm_size": len(self.stm),
            "ltm_size": len(self.ltm),
            "vector_store_size": self.vector_store.size,
            "embedding_dimension": self.embedding_provider.embedding_dimension,
        })
        return current_stats
    
    def _find_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """Find memory entry by ID."""
        # Check STM first
        for entry in self.stm:
            if entry.id == entry_id:
                return entry
        
        # Check LTM
        return self.ltm.get(entry_id)
    
    def _matches_filters(self, entry: MemoryEntry, filters: Dict[str, Any]) -> bool:
        """Check if entry matches metadata filters."""
        for key, value in filters.items():
            if key not in entry.metadata:
                return False
            if entry.metadata[key] != value:
                return False
        return True
