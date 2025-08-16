"""
Spatial Memory System Integration
================================

This module integrates the spatial environment with the existing Lumina Memory System,
providing a unified interface that combines traditional vector storage with spatial
attribute-based positioning.

The system maintains both:
1. Traditional vector store for fast similarity search
2. Spatial environment for attribute-based relationships and environmental behavior

This dual approach provides:
- Fast retrieval through vector indices
- Rich spatial relationships through attribute space
- Environmental evolution and adaptation
- Multi-modal querying capabilities
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Union
import logging

from .core import MemoryEntry, QueryResult, QueryType
from .vector_store import VectorStore
from .embeddings import EmbeddingProvider
from .spatial_environment import SpatialEnvironment, SpatialUnit, spatial_memory_search
from .memory_system import MemorySystem
from .config import LuminaConfig

logger = logging.getLogger(__name__)


class SpatialMemorySystem(MemorySystem):
    """
    Enhanced memory system with spatial environment capabilities.
    
    Extends the base MemorySystem to include spatial positioning and
    environmental behavior while maintaining compatibility with existing APIs.
    """
    
    def __init__(self, 
                 embedding_provider: EmbeddingProvider,
                 vector_store: VectorStore,
                 config: Optional[LuminaConfig] = None,
                 spatial_dimension: int = 256,
                 enable_spatial_evolution: bool = True,
                 spatial_decay_rate: float = 0.1,
                 evolution_interval: int = 300):
        """
        Initialize spatial memory system.
        
        Args:
            embedding_provider: Provider for generating embeddings
            vector_store: Traditional vector storage for fast retrieval
            config: System configuration (LuminaConfig object)
            spatial_dimension: Dimension of spatial attribute space
            enable_spatial_evolution: Whether to enable automatic environment evolution
            spatial_decay_rate: Rate at which spatial relationships decay
            evolution_interval: Time between environment evolutions (seconds)
        """
        # Initialize base memory system
        super().__init__(embedding_provider, vector_store, config)
        
        # Initialize spatial environment
        self.spatial_env = SpatialEnvironment(
            dimension=spatial_dimension,
            decay_rate=spatial_decay_rate
        )
        
        self.spatial_dimension = spatial_dimension
        self.enable_spatial_evolution = enable_spatial_evolution
        self.last_evolution_check = time.time()
        self.evolution_interval = evolution_interval
        
        logger.info(f"Spatial memory system initialized with {spatial_dimension}D spatial environment")
    
    def ingest(self, content: str, metadata: Dict = None, importance: float = 0.5) -> str:
        """
        Ingest content into both traditional and spatial memory systems.
        
        This method:
        1. Creates a memory entry with embedding
        2. Stores in traditional vector store
        3. Adds to spatial environment
        4. Updates spatial relationships
        
        Returns:
            Memory entry ID
        """
        # Create memory entry
        entry = MemoryEntry(
            content=content,
            metadata=metadata or {},
            importance_score=importance
        )
        
        # Generate embedding
        try:
            entry.embedding = self.embedding_provider.embed_single(content)
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
        
        # Store in traditional vector store
        try:
            self.vector_store.add([entry])
        except Exception as e:
            logger.error(f"Failed to add to vector store: {e}")
            raise
        
        # Add to spatial environment
        try:
            spatial_unit = self.spatial_env.add_unit(entry)
            logger.debug(f"Added memory {entry.id} to spatial environment")
        except Exception as e:
            logger.error(f"Failed to add to spatial environment: {e}")
            # Continue without spatial - system should still work
        
        # Check if environment evolution is needed
        self._check_evolution()
        
        return entry.id
    
    def recall(self, query: str, k: int = 10, query_type: str = "hybrid") -> List[QueryResult]:
        """
        Recall memories using both traditional and spatial approaches.
        
        Args:
            query: Query string
            k: Number of results to return
            query_type: Type of query ("traditional", "spatial", "hybrid")
            
        Returns:
            List of QueryResult objects with spatial context
        """
        if query_type == "traditional":
            return self._traditional_recall(query, k)
        elif query_type == "spatial":
            return self._spatial_recall(query, k)
        elif query_type == "hybrid":
            return self._hybrid_recall(query, k)
        else:
            raise ValueError(f"Unknown query type: {query_type}")
    
    def _traditional_recall(self, query: str, k: int) -> List[QueryResult]:
        """Traditional vector-based recall."""
        return super().recall(query, k)
    
    def _spatial_recall(self, query: str, k: int) -> List[QueryResult]:
        """Spatial environment-based recall."""
        try:
            # Use spatial search
            results = spatial_memory_search(self.spatial_env, query, k=k)
            
            # Convert to QueryResult objects
            query_results = []
            for memory_entry, similarity_score in results:
                # Get spatial context
                spatial_unit = self.spatial_env.units.get(memory_entry.id)
                spatial_context = {}
                
                if spatial_unit:
                    spatial_context = {
                        "spatial_energy": spatial_unit.spatial_energy,
                        "cluster_id": spatial_unit.cluster_id,
                        "neighbor_count": len(spatial_unit.neighbors),
                        "activation_level": spatial_unit.activation_level,
                        "query_type": "spatial"
                    }
                
                query_result = QueryResult(
                    entry=memory_entry,
                    similarity_score=similarity_score,
                    retrieval_context=spatial_context
                )
                query_results.append(query_result)
            
            return query_results
            
        except Exception as e:
            logger.error(f"Spatial recall failed: {e}")
            # Fallback to traditional recall
            return self._traditional_recall(query, k)
    
    def _hybrid_recall(self, query: str, k: int) -> List[QueryResult]:
        """
        Hybrid recall combining traditional and spatial approaches.
        
        This method:
        1. Gets results from both traditional and spatial systems
        2. Combines and re-ranks results
        3. Provides rich context from both systems
        """
        # Get traditional results
        traditional_results = self._traditional_recall(query, k * 2)  # Get more for merging
        
        # Get spatial results
        spatial_results = self._spatial_recall(query, k * 2)
        
        # Combine results with hybrid scoring
        combined_results = self._combine_results(traditional_results, spatial_results, k)
        
        return combined_results
    
    def _combine_results(self, traditional: List[QueryResult], 
                        spatial: List[QueryResult], k: int) -> List[QueryResult]:
        """Combine and re-rank traditional and spatial results."""
        # Create lookup for spatial results
        spatial_lookup = {result.entry.id: result for result in spatial}
        
        # Combine results with hybrid scoring
        hybrid_results = []
        seen_ids = set()
        
        for trad_result in traditional:
            if trad_result.entry.id in seen_ids:
                continue
            
            seen_ids.add(trad_result.entry.id)
            
            # Get spatial result if available
            spatial_result = spatial_lookup.get(trad_result.entry.id)
            
            if spatial_result:
                # Hybrid scoring: combine traditional and spatial scores
                hybrid_score = (0.6 * trad_result.similarity_score + 
                               0.4 * spatial_result.similarity_score)
                
                # Combine contexts
                hybrid_context = {
                    **trad_result.retrieval_context,
                    **spatial_result.retrieval_context,
                    "query_type": "hybrid",
                    "traditional_score": trad_result.similarity_score,
                    "spatial_score": spatial_result.similarity_score
                }
            else:
                # Only traditional result available
                hybrid_score = trad_result.similarity_score * 0.8  # Slight penalty
                hybrid_context = {
                    **trad_result.retrieval_context,
                    "query_type": "hybrid",
                    "traditional_score": trad_result.similarity_score,
                    "spatial_score": None
                }
            
            hybrid_result = QueryResult(
                entry=trad_result.entry,
                similarity_score=hybrid_score,
                retrieval_context=hybrid_context
            )
            hybrid_results.append(hybrid_result)
        
        # Add spatial-only results
        for spatial_result in spatial:
            if spatial_result.entry.id not in seen_ids:
                seen_ids.add(spatial_result.entry.id)
                
                hybrid_score = spatial_result.similarity_score * 0.7  # Penalty for no traditional match
                hybrid_context = {
                    **spatial_result.retrieval_context,
                    "query_type": "hybrid",
                    "traditional_score": None,
                    "spatial_score": spatial_result.similarity_score
                }
                
                hybrid_result = QueryResult(
                    entry=spatial_result.entry,
                    similarity_score=hybrid_score,
                    retrieval_context=hybrid_context
                )
                hybrid_results.append(hybrid_result)
        
        # Sort by hybrid score and return top k
        hybrid_results.sort(key=lambda x: x.similarity_score, reverse=True)
        return hybrid_results[:k]
    
    def _check_evolution(self):
        """Check if spatial environment evolution is needed."""
        if not self.enable_spatial_evolution:
            return
        
        current_time = time.time()
        if current_time - self.last_evolution_check > self.evolution_interval:
            try:
                self.spatial_env.evolve_environment()
                self.last_evolution_check = current_time
                logger.debug("Spatial environment evolved")
            except Exception as e:
                logger.error(f"Environment evolution failed: {e}")
    
    def force_evolution(self):
        """Force immediate evolution of the spatial environment."""
        try:
            self.spatial_env.evolve_environment(force_evolution=True)
            self.last_evolution_check = time.time()
            logger.info("Forced spatial environment evolution")
        except Exception as e:
            logger.error(f"Forced evolution failed: {e}")
    
    def get_spatial_neighbors(self, memory_id: str, radius: float = 0.5) -> List[str]:
        """Get spatial neighbors of a memory within a similarity radius."""
        return self.spatial_env.get_unit_neighborhood(memory_id, radius)
    
    def get_memory_spatial_info(self, memory_id: str) -> Dict:
        """Get detailed spatial information about a memory."""
        if memory_id not in self.spatial_env.units:
            return {"error": "Memory not found in spatial environment"}
        
        unit = self.spatial_env.units[memory_id]
        
        return {
            "memory_id": memory_id,
            "spatial_energy": unit.spatial_energy,
            "activation_level": unit.activation_level,
            "cluster_id": unit.cluster_id,
            "neighbor_count": len(unit.neighbors),
            "neighbors": list(unit.neighbors.keys()),
            "last_update": unit.last_update,
            "attribute_vector_norm": np.linalg.norm(unit.attribute_vector),
            "drift_velocity_norm": np.linalg.norm(unit.drift_velocity)
        }
    
    def get_spatial_stats(self) -> Dict:
        """Get comprehensive spatial environment statistics."""
        return self.spatial_env.get_environment_stats()
    
    def visualize_spatial_environment(self, method: str = "tsne") -> Dict:
        """Create visualization data for the spatial environment."""
        return self.spatial_env.visualize_environment(method)
    
    def spatial_query(self, query_vector: np.ndarray, k: int = 10, 
                     query_type: str = "similarity") -> List[Tuple[str, float]]:
        """Direct spatial query using a vector."""
        return self.spatial_env.spatial_query(query_vector, k, query_type)
    
    def consolidate_spatial_memories(self, similarity_threshold: float = 0.9):
        """
        Consolidate very similar memories in the spatial environment.
        
        This method identifies memories that are extremely similar in attribute space
        and can be consolidated to reduce redundancy while preserving information.
        """
        consolidation_candidates = []
        
        # Find pairs of very similar memories
        unit_ids = list(self.spatial_env.units.keys())
        for i, id1 in enumerate(unit_ids):
            for id2 in unit_ids[i+1:]:
                unit1 = self.spatial_env.units[id1]
                unit2 = self.spatial_env.units[id2]
                
                similarity_score = np.dot(unit1.attribute_vector, unit2.attribute_vector)
                if similarity_score > similarity_threshold:
                    consolidation_candidates.append((id1, id2, similarity_score))
        
        # Sort by similarity (highest first)
        consolidation_candidates.sort(key=lambda x: x[2], reverse=True)
        
        consolidated_count = 0
        for id1, id2, similarity_score in consolidation_candidates:
            # Check if both memories still exist (might have been consolidated already)
            if id1 in self.spatial_env.units and id2 in self.spatial_env.units:
                # Consolidate by merging content and removing one
                unit1 = self.spatial_env.units[id1]
                unit2 = self.spatial_env.units[id2]
                
                # Merge content (keep the more important one as primary)
                if unit1.memory_entry.importance_score >= unit2.memory_entry.importance_score:
                    primary, secondary = unit1, unit2
                    primary_id, secondary_id = id1, id2
                else:
                    primary, secondary = unit2, unit1
                    primary_id, secondary_id = id2, id1
                
                # Update primary memory with consolidated information
                primary.memory_entry.content += f"\n[Consolidated: {secondary.memory_entry.content}]"
                primary.memory_entry.importance_score = max(
                    primary.memory_entry.importance_score,
                    secondary.memory_entry.importance_score
                )
                primary.memory_entry.access_count += secondary.memory_entry.access_count
                
                # Remove secondary memory from both systems
                try:
                    self.vector_store.remove([secondary_id])
                    del self.spatial_env.units[secondary_id]
                    consolidated_count += 1
                    logger.debug(f"Consolidated memories {primary_id} and {secondary_id}")
                except Exception as e:
                    logger.error(f"Failed to consolidate memories: {e}")
        
        if consolidated_count > 0:
            logger.info(f"Consolidated {consolidated_count} memory pairs")
            # Force evolution to update spatial structure
            self.force_evolution()
        
        return consolidated_count
    
    def forget_spatial_memories(self, criteria: Dict) -> int:
        """
        Forget memories based on spatial criteria.
        
        Args:
            criteria: Dictionary with forgetting criteria:
                - min_energy: Minimum spatial energy to keep
                - max_age_hours: Maximum age in hours to keep
                - min_importance: Minimum importance score to keep
                - cluster_ids: List of cluster IDs to forget
                
        Returns:
            Number of memories forgotten
        """
        to_forget = []
        current_time = time.time()
        
        for unit_id, unit in self.spatial_env.units.items():
            should_forget = False
            
            # Check energy threshold
            if 'min_energy' in criteria and unit.spatial_energy < criteria['min_energy']:
                should_forget = True
            
            # Check age threshold
            if 'max_age_hours' in criteria:
                age_hours = (current_time - unit.memory_entry.timestamp.timestamp()) / 3600
                if age_hours > criteria['max_age_hours']:
                    should_forget = True
            
            # Check importance threshold
            if 'min_importance' in criteria and unit.memory_entry.importance_score < criteria['min_importance']:
                should_forget = True
            
            # Check cluster membership
            if 'cluster_ids' in criteria and unit.cluster_id in criteria['cluster_ids']:
                should_forget = True
            
            if should_forget:
                to_forget.append(unit_id)
        
        # Remove forgotten memories
        forgotten_count = 0
        for memory_id in to_forget:
            try:
                # Remove from vector store
                self.vector_store.remove([memory_id])
                
                # Remove from spatial environment
                del self.spatial_env.units[memory_id]
                
                forgotten_count += 1
                logger.debug(f"Forgot memory {memory_id}")
                
            except Exception as e:
                logger.error(f"Failed to forget memory {memory_id}: {e}")
        
        if forgotten_count > 0:
            logger.info(f"Forgot {forgotten_count} memories based on spatial criteria")
            # Force evolution to update spatial structure
            self.force_evolution()
        
        return forgotten_count


# Factory function for easy creation
def create_spatial_memory_system(embedding_provider: EmbeddingProvider,
                                vector_store: VectorStore,
                                config: Optional[LuminaConfig] = None,
                                spatial_config: Dict = None) -> SpatialMemorySystem:
    """Create a spatial memory system with default configuration."""
    # Default spatial configuration
    default_spatial_config = {
        'spatial_decay_rate': 0.1,
        'evolution_interval': 300,  # 5 minutes
        'spatial_dimension': 256,
        'enable_spatial_evolution': True
    }
    
    if spatial_config:
        default_spatial_config.update(spatial_config)
    
    return SpatialMemorySystem(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        config=config,
        spatial_dimension=default_spatial_config['spatial_dimension'],
        enable_spatial_evolution=default_spatial_config['enable_spatial_evolution'],
        spatial_decay_rate=default_spatial_config['spatial_decay_rate'],
        evolution_interval=default_spatial_config['evolution_interval']
    )