"""
Spatial Environment System for Lumina Memory
===========================================

This module implements a non-locational spatial environment where memory units
exist in a multi-dimensional attribute space. Units are positioned based on their
intrinsic properties (semantic, temporal, emotional, structural) rather than 
physical coordinates.

The environment uses HRR (Holographic Reduced Representations) as the foundation
for managing theoretical "locations" and relationships between units.

Key Concepts:
- Units exist in attribute space, not coordinate space
- Location is determined by semantic, temporal, emotional, and structural properties
- Environment evolves dynamically while maintaining constant adaptive nature
- Scalable through HRR operations and efficient indexing
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import logging

from .hrr import (
    reference_vector, bind_vectors, unbind_vectors, superpose_vectors,
    similarity, normalize_vector
)
from .math_foundation import (
    circular_convolution, circular_correlation, memory_unit_score,
    mathematical_coherence, cosine_similarity
)
from .core import MemoryEntry

logger = logging.getLogger(__name__)


@dataclass
class SpatialUnit:
    """
    A memory unit with spatial properties in attribute space.
    
    The unit's 'location' is determined by its attribute vector,
    which is a superposition of semantic, temporal, emotional,
    and structural components.
    """
    memory_entry: MemoryEntry
    
    # Attribute vectors that determine spatial position
    semantic_vector: np.ndarray = None
    temporal_vector: np.ndarray = None
    emotional_vector: np.ndarray = None
    structural_vector: np.ndarray = None
    
    # Composite attribute vector (unit's "location" in space)
    attribute_vector: np.ndarray = None
    
    # Spatial relationships
    neighbors: Dict[str, float] = field(default_factory=dict)  # unit_id -> similarity
    cluster_id: Optional[str] = None
    activation_level: float = 0.0
    
    # Environmental properties
    spatial_energy: float = 0.0  # Energy level in the environment
    drift_velocity: np.ndarray = None  # Direction of spatial drift
    last_update: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Initialize spatial properties."""
        # drift_velocity will be initialized when the unit is added to environment
        pass


class SpatialEnvironment:
    """
    The spatial environment that manages memory units in attribute space.
    
    This environment:
    1. Positions units based on their intrinsic attributes
    2. Maintains dynamic relationships between units
    3. Evolves over time while preserving structure
    4. Provides efficient spatial queries and operations
    """
    
    def __init__(self, dimension: int = 256, decay_rate: float = 0.1):
        """
        Initialize the spatial environment.
        
        Args:
            dimension: Dimensionality of the attribute space
            decay_rate: Rate at which spatial relationships decay over time
        """
        self.dimension = dimension
        self.decay_rate = decay_rate
        
        # Core storage
        self.units: Dict[str, SpatialUnit] = {}
        self.clusters: Dict[str, Set[str]] = defaultdict(set)
        
        # Spatial indices for efficient queries
        self.semantic_index: Dict[str, np.ndarray] = {}
        self.temporal_index: Dict[float, Set[str]] = defaultdict(set)
        self.energy_index: Dict[str, float] = {}
        
        # Environment state
        self.global_energy: float = 0.0
        self.last_evolution: float = time.time()
        self.evolution_count: int = 0
        
        # HRR basis vectors for different attribute types
        self._init_basis_vectors()
        
        logger.info(f"Spatial environment initialized: dim={dimension}, decay_rate={decay_rate}")
    
    def _init_basis_vectors(self):
        """Initialize HRR basis vectors for different attribute types."""
        # Create deterministic basis vectors for each attribute type
        self.semantic_basis = reference_vector("SEMANTIC", {"type": "basis"}, self.dimension)
        self.temporal_basis = reference_vector("TEMPORAL", {"type": "basis"}, self.dimension)
        self.emotional_basis = reference_vector("EMOTIONAL", {"type": "basis"}, self.dimension)
        self.structural_basis = reference_vector("STRUCTURAL", {"type": "basis"}, self.dimension)
        
        # Relation vectors for binding operations
        self.position_relation = reference_vector("POSITION", {"type": "relation"}, self.dimension)
        self.neighbor_relation = reference_vector("NEIGHBOR", {"type": "relation"}, self.dimension)
    
    def add_unit(self, memory_entry: MemoryEntry) -> SpatialUnit:
        """
        Add a memory unit to the spatial environment.
        
        The unit's position is determined by computing its attribute vectors
        and combining them into a composite spatial representation.
        """
        # Create spatial unit
        unit = SpatialUnit(memory_entry=memory_entry)
        
        # Compute attribute vectors
        unit.semantic_vector = self._compute_semantic_vector(memory_entry)
        unit.temporal_vector = self._compute_temporal_vector(memory_entry)
        unit.emotional_vector = self._compute_emotional_vector(memory_entry)
        unit.structural_vector = self._compute_structural_vector(memory_entry)
        
        # Compute composite attribute vector (unit's "location")
        unit.attribute_vector = self._compute_attribute_vector(unit)
        
        # Initialize spatial properties
        unit.spatial_energy = self._compute_spatial_energy(unit)
        unit.activation_level = memory_entry.importance_score
        
        # Initialize drift velocity with correct dimension
        if unit.drift_velocity is None:
            unit.drift_velocity = np.zeros(self.dimension)
        
        # Add to environment
        self.units[memory_entry.id] = unit
        self._update_indices(unit)
        self._find_neighbors(unit)
        
        # Update global environment state
        self.global_energy += unit.spatial_energy
        
        logger.debug(f"Added unit {memory_entry.id} to spatial environment")
        return unit
    
    def _compute_semantic_vector(self, memory_entry: MemoryEntry) -> np.ndarray:
        """Compute semantic attribute vector using HRR."""
        if memory_entry.embedding is not None:
            # Bind semantic content with semantic basis
            semantic_content = normalize_vector(memory_entry.embedding[:self.dimension])
            return bind_vectors(self.semantic_basis, semantic_content)
        else:
            # Use content-based HRR vector
            return reference_vector(memory_entry.content, {"type": "semantic"}, self.dimension)
    
    def _compute_temporal_vector(self, memory_entry: MemoryEntry) -> np.ndarray:
        """Compute temporal attribute vector using HRR."""
        # Create temporal representation based on timestamp
        timestamp_str = str(int(memory_entry.timestamp.timestamp()))
        temporal_content = reference_vector(timestamp_str, {"type": "temporal"}, self.dimension)
        return bind_vectors(self.temporal_basis, temporal_content)
    
    def _compute_emotional_vector(self, memory_entry: MemoryEntry) -> np.ndarray:
        """Compute emotional attribute vector using HRR."""
        # Use importance score and access count as emotional indicators
        emotional_data = {
            "importance": memory_entry.importance_score,
            "access_count": memory_entry.access_count
        }
        emotional_content = reference_vector(str(emotional_data), {"type": "emotional"}, self.dimension)
        return bind_vectors(self.emotional_basis, emotional_content)
    
    def _compute_structural_vector(self, memory_entry: MemoryEntry) -> np.ndarray:
        """Compute structural attribute vector using HRR."""
        # Use metadata to determine structural properties
        structural_data = {
            "metadata_keys": sorted(memory_entry.metadata.keys()),
            "content_length": len(memory_entry.content)
        }
        structural_content = reference_vector(str(structural_data), {"type": "structural"}, self.dimension)
        return bind_vectors(self.structural_basis, structural_content)
    
    def _compute_attribute_vector(self, unit: SpatialUnit) -> np.ndarray:
        """
        Compute composite attribute vector that represents the unit's location.
        
        This is the core of the spatial positioning system - the unit's "location"
        in attribute space is a weighted superposition of all its attribute vectors.
        """
        vectors = [
            unit.semantic_vector,
            unit.temporal_vector,
            unit.emotional_vector,
            unit.structural_vector
        ]
        
        # Weights based on the unit's properties
        importance = unit.memory_entry.importance_score
        recency = 1.0 / (1.0 + (time.time() - unit.memory_entry.timestamp.timestamp()) / 3600)
        
        weights = [
            0.4,  # Semantic weight
            0.2 * recency,  # Temporal weight (decays with time)
            0.2 * importance,  # Emotional weight (based on importance)
            0.2  # Structural weight
        ]
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Superpose vectors with weights
        attribute_vector = superpose_vectors(vectors, weights)
        return normalize_vector(attribute_vector)
    
    def _compute_spatial_energy(self, unit: SpatialUnit) -> float:
        """Compute the spatial energy of a unit in the environment."""
        # Energy is based on the unit's position and relationships
        base_energy = unit.memory_entry.importance_score
        
        # Add energy from spatial coherence
        coherence_energy = 0.0
        if len(self.units) > 1:
            for other_id, other_unit in self.units.items():
                if other_id != unit.memory_entry.id:
                    similarity_score = similarity(unit.attribute_vector, other_unit.attribute_vector)
                    coherence_energy += similarity_score * other_unit.memory_entry.importance_score
        
        return base_energy + 0.1 * coherence_energy
    
    def _update_indices(self, unit: SpatialUnit):
        """Update spatial indices for efficient queries."""
        unit_id = unit.memory_entry.id
        
        # Update semantic index
        self.semantic_index[unit_id] = unit.semantic_vector
        
        # Update temporal index
        timestamp = unit.memory_entry.timestamp.timestamp()
        self.temporal_index[timestamp].add(unit_id)
        
        # Update energy index
        self.energy_index[unit_id] = unit.spatial_energy
    
    def _find_neighbors(self, unit: SpatialUnit, k: int = 10):
        """Find spatial neighbors for a unit based on attribute similarity."""
        unit_id = unit.memory_entry.id
        similarities = []
        
        for other_id, other_unit in self.units.items():
            if other_id != unit_id:
                sim = similarity(unit.attribute_vector, other_unit.attribute_vector)
                similarities.append((other_id, sim))
        
        # Sort by similarity and take top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        unit.neighbors = dict(similarities[:k])
        
        # Update neighbors' neighbor lists (bidirectional)
        for neighbor_id, sim in unit.neighbors.items():
            if neighbor_id in self.units:
                self.units[neighbor_id].neighbors[unit_id] = sim
    
    def evolve_environment(self, force_evolution: bool = False):
        """
        Evolve the spatial environment over time.
        
        This implements the dynamic behavior of the environment:
        1. Units drift based on their relationships
        2. Clusters form and dissolve
        3. Energy redistributes across the space
        4. Spatial structure adapts while maintaining coherence
        """
        current_time = time.time()
        time_delta = current_time - self.last_evolution
        
        # Only evolve if enough time has passed or forced
        if not force_evolution and time_delta < 60:  # Evolve at most once per minute
            return
        
        logger.debug(f"Evolving spatial environment (evolution #{self.evolution_count + 1})")
        
        # 1. Compute drift forces for each unit
        self._compute_drift_forces(time_delta)
        
        # 2. Update unit positions based on drift
        self._apply_spatial_drift(time_delta)
        
        # 3. Update clusters based on new positions
        self._update_clusters()
        
        # 4. Redistribute energy across the environment
        self._redistribute_energy()
        
        # 5. Update neighbor relationships
        self._update_all_neighbors()
        
        # Update evolution state
        self.last_evolution = current_time
        self.evolution_count += 1
        
        logger.info(f"Environment evolution complete: {len(self.units)} units, {len(self.clusters)} clusters")
    
    def _compute_drift_forces(self, time_delta: float):
        """Compute drift forces for each unit based on spatial relationships."""
        for unit_id, unit in self.units.items():
            drift_force = np.zeros(self.dimension)
            
            # Attraction to similar units
            for neighbor_id, similarity_score in unit.neighbors.items():
                if neighbor_id in self.units:
                    neighbor = self.units[neighbor_id]
                    direction = neighbor.attribute_vector - unit.attribute_vector
                    force_magnitude = similarity_score * 0.1  # Attraction strength
                    drift_force += force_magnitude * direction
            
            # Repulsion from dissimilar units (maintain diversity)
            for other_id, other_unit in self.units.items():
                if other_id != unit_id and other_id not in unit.neighbors:
                    similarity_score = similarity(unit.attribute_vector, other_unit.attribute_vector)
                    if similarity_score < 0.1:  # Very dissimilar
                        direction = unit.attribute_vector - other_unit.attribute_vector
                        force_magnitude = (0.1 - similarity_score) * 0.05  # Repulsion strength
                        drift_force += force_magnitude * normalize_vector(direction)
            
            # Temporal decay (older memories drift toward periphery)
            age_hours = (time.time() - unit.memory_entry.timestamp.timestamp()) / 3600
            decay_force = -unit.attribute_vector * self.decay_rate * age_hours * 0.01
            drift_force += decay_force
            
            # Update drift velocity with momentum
            momentum = 0.8
            unit.drift_velocity = momentum * unit.drift_velocity + (1 - momentum) * drift_force
    
    def _apply_spatial_drift(self, time_delta: float):
        """Apply drift forces to update unit positions in attribute space."""
        for unit in self.units.values():
            # Apply drift to attribute vector
            drift_amount = unit.drift_velocity * time_delta * 0.1  # Scale drift speed
            new_position = unit.attribute_vector + drift_amount
            
            # Normalize to maintain unit vector constraint
            unit.attribute_vector = normalize_vector(new_position)
            
            # Update spatial energy based on new position
            unit.spatial_energy = self._compute_spatial_energy(unit)
            unit.last_update = time.time()
    
    def _update_clusters(self):
        """Update cluster assignments based on current spatial positions."""
        # Clear existing clusters
        self.clusters.clear()
        
        # Simple clustering based on similarity threshold
        cluster_threshold = 0.7
        cluster_id = 0
        
        for unit_id, unit in self.units.items():
            if unit.cluster_id is None:
                # Start new cluster
                current_cluster = f"cluster_{cluster_id}"
                cluster_id += 1
                
                # Add unit to cluster
                unit.cluster_id = current_cluster
                self.clusters[current_cluster].add(unit_id)
                
                # Find similar units to add to same cluster
                for other_id, other_unit in self.units.items():
                    if other_id != unit_id and other_unit.cluster_id is None:
                        sim = similarity(unit.attribute_vector, other_unit.attribute_vector)
                        if sim > cluster_threshold:
                            other_unit.cluster_id = current_cluster
                            self.clusters[current_cluster].add(other_id)
    
    def _redistribute_energy(self):
        """Redistribute energy across the spatial environment."""
        total_energy = sum(unit.spatial_energy for unit in self.units.values())
        
        if total_energy > 0:
            # Normalize energy distribution
            for unit in self.units.values():
                unit.spatial_energy = unit.spatial_energy / total_energy * len(self.units)
                
            # Update global energy
            self.global_energy = total_energy
    
    def _update_all_neighbors(self):
        """Update neighbor relationships for all units."""
        for unit in self.units.values():
            self._find_neighbors(unit)
    
    def spatial_query(self, query_vector: np.ndarray, k: int = 10, 
                     query_type: str = "similarity") -> List[Tuple[str, float]]:
        """
        Perform spatial query in the attribute environment.
        
        Args:
            query_vector: Query vector in attribute space
            k: Number of results to return
            query_type: Type of query ("similarity", "energy", "cluster")
            
        Returns:
            List of (unit_id, score) tuples
        """
        if query_type == "similarity":
            return self._similarity_query(query_vector, k)
        elif query_type == "energy":
            return self._energy_query(k)
        elif query_type == "cluster":
            return self._cluster_query(query_vector, k)
        else:
            raise ValueError(f"Unknown query type: {query_type}")
    
    def _similarity_query(self, query_vector: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Query based on attribute similarity."""
        similarities = []
        
        for unit_id, unit in self.units.items():
            sim = similarity(query_vector, unit.attribute_vector)
            similarities.append((unit_id, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def _energy_query(self, k: int) -> List[Tuple[str, float]]:
        """Query based on spatial energy levels."""
        energy_scores = [(unit_id, unit.spatial_energy) for unit_id, unit in self.units.items()]
        energy_scores.sort(key=lambda x: x[1], reverse=True)
        return energy_scores[:k]
    
    def _cluster_query(self, query_vector: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Query based on cluster membership and similarity."""
        # Find most similar cluster
        cluster_similarities = {}
        for cluster_id, unit_ids in self.clusters.items():
            cluster_center = np.mean([self.units[uid].attribute_vector for uid in unit_ids], axis=0)
            cluster_similarities[cluster_id] = similarity(query_vector, cluster_center)
        
        # Get best cluster
        best_cluster = max(cluster_similarities.keys(), key=lambda x: cluster_similarities[x])
        
        # Return units from best cluster, sorted by similarity
        results = []
        for unit_id in self.clusters[best_cluster]:
            unit = self.units[unit_id]
            sim = similarity(query_vector, unit.attribute_vector)
            results.append((unit_id, sim))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def get_unit_neighborhood(self, unit_id: str, radius: float = 0.5) -> List[str]:
        """Get all units within a certain similarity radius of the given unit."""
        if unit_id not in self.units:
            return []
        
        target_unit = self.units[unit_id]
        neighbors = []
        
        for other_id, other_unit in self.units.items():
            if other_id != unit_id:
                sim = similarity(target_unit.attribute_vector, other_unit.attribute_vector)
                if sim >= radius:
                    neighbors.append(other_id)
        
        return neighbors
    
    def get_environment_stats(self) -> Dict[str, any]:
        """Get comprehensive statistics about the spatial environment."""
        if not self.units:
            return {"status": "empty"}
        
        # Basic stats
        stats = {
            "total_units": len(self.units),
            "total_clusters": len(self.clusters),
            "global_energy": self.global_energy,
            "evolution_count": self.evolution_count,
            "last_evolution": self.last_evolution
        }
        
        # Energy distribution
        energies = [unit.spatial_energy for unit in self.units.values()]
        stats.update({
            "energy_mean": np.mean(energies),
            "energy_std": np.std(energies),
            "energy_min": np.min(energies),
            "energy_max": np.max(energies)
        })
        
        # Cluster distribution
        cluster_sizes = [len(unit_ids) for unit_ids in self.clusters.values()]
        if cluster_sizes:
            stats.update({
                "cluster_size_mean": np.mean(cluster_sizes),
                "cluster_size_std": np.std(cluster_sizes),
                "largest_cluster": max(cluster_sizes),
                "smallest_cluster": min(cluster_sizes)
            })
        
        # Connectivity stats
        neighbor_counts = [len(unit.neighbors) for unit in self.units.values()]
        stats.update({
            "connectivity_mean": np.mean(neighbor_counts),
            "connectivity_std": np.std(neighbor_counts),
            "most_connected": max(neighbor_counts) if neighbor_counts else 0,
            "least_connected": min(neighbor_counts) if neighbor_counts else 0
        })
        
        return stats
    
    def visualize_environment(self, method: str = "tsne") -> Dict[str, any]:
        """
        Create a 2D visualization of the spatial environment.
        
        Args:
            method: Dimensionality reduction method ("tsne", "pca", "umap")
            
        Returns:
            Dictionary with visualization data
        """
        if len(self.units) < 2:
            return {"error": "Need at least 2 units for visualization"}
        
        # Collect attribute vectors
        unit_ids = list(self.units.keys())
        vectors = np.array([self.units[uid].attribute_vector for uid in unit_ids])
        
        # Apply dimensionality reduction
        if method == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            coords_2d = reducer.fit_transform(vectors)
        elif method == "tsne":
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
            coords_2d = reducer.fit_transform(vectors)
        elif method == "umap":
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42)
                coords_2d = reducer.fit_transform(vectors)
            except ImportError:
                # Fallback to PCA if UMAP not available
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=2)
                coords_2d = reducer.fit_transform(vectors)
        else:
            raise ValueError(f"Unknown visualization method: {method}")
        
        # Prepare visualization data
        viz_data = {
            "method": method,
            "coordinates": coords_2d.tolist(),
            "unit_ids": unit_ids,
            "clusters": {},
            "energies": [self.units[uid].spatial_energy for uid in unit_ids],
            "importances": [self.units[uid].memory_entry.importance_score for uid in unit_ids]
        }
        
        # Add cluster information
        for cluster_id, unit_ids_in_cluster in self.clusters.items():
            cluster_coords = [coords_2d[unit_ids.index(uid)] for uid in unit_ids_in_cluster if uid in unit_ids]
            viz_data["clusters"][cluster_id] = {
                "unit_ids": list(unit_ids_in_cluster),
                "coordinates": cluster_coords
            }
        
        return viz_data


# Utility functions for working with the spatial environment

def create_spatial_environment(memory_entries: List[MemoryEntry], 
                             dimension: int = 256) -> SpatialEnvironment:
    """Create and populate a spatial environment with memory entries."""
    env = SpatialEnvironment(dimension=dimension)
    
    for entry in memory_entries:
        env.add_unit(entry)
    
    # Initial evolution to establish spatial structure
    env.evolve_environment(force_evolution=True)
    
    return env


def spatial_memory_search(env: SpatialEnvironment, 
                         query_content: str,
                         query_metadata: Dict = None,
                         k: int = 10) -> List[Tuple[MemoryEntry, float]]:
    """
    Search for memories in the spatial environment using content and metadata.
    
    This creates a query vector in attribute space and finds the most similar units.
    """
    # Create query vector using same process as unit creation
    query_semantic = reference_vector(query_content, query_metadata or {}, env.dimension)
    query_temporal = reference_vector(str(time.time()), {"type": "temporal"}, env.dimension)
    query_emotional = reference_vector("0.5", {"type": "emotional"}, env.dimension)  # Neutral
    query_structural = reference_vector(str(len(query_content)), {"type": "structural"}, env.dimension)
    
    # Combine into attribute vector
    query_vectors = [query_semantic, query_temporal, query_emotional, query_structural]
    query_weights = [0.6, 0.1, 0.1, 0.2]  # Emphasize semantic similarity
    query_attribute = normalize_vector(superpose_vectors(query_vectors, query_weights))
    
    # Perform spatial query
    results = env.spatial_query(query_attribute, k=k, query_type="similarity")
    
    # Convert to memory entries with scores
    memory_results = []
    for unit_id, score in results:
        if unit_id in env.units:
            memory_entry = env.units[unit_id].memory_entry
            memory_results.append((memory_entry, score))
    
    return memory_results