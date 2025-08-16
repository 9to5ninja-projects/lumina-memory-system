"""
Storage Integration Layer for XP Core Compatibility
==================================================

This module provides a compatibility layer that allows existing xp_core logic
to work seamlessly with the new optimized storage system while gradually
migrating away from JSON bloat.

Key Features:
- Backward compatibility with existing MemoryEntry and SpatialUnit classes
- Transparent migration from JSON to binary storage
- Performance monitoring and gradual optimization
- CPI metrics integration for consciousness battery

Integration Strategy:
1. Wrap existing storage calls with optimized backend
2. Maintain API compatibility for xp_core components
3. Provide migration utilities for existing data
4. Add performance monitoring and metrics

Author: Lumina Memory Team
License: MIT
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import numpy as np

# Import our components
from .core import MemoryEntry
from .spatial_environment import SpatialUnit, SpatialEnvironment
from .xp_core_unified import XPUnit, UnifiedXPConfig
from .optimized_storage import (
    OptimizedStorageBackend, HRRUnitData, StorageMetrics,
    SpatialEnvironmentStorage
)
from .hrr import normalize_vector, bind_vectors, unbind_vectors

logger = logging.getLogger(__name__)


class StorageIntegrationConfig:
    """Configuration for storage integration and migration."""
    
    def __init__(self):
        # Migration settings
        self.enable_optimized_storage = True
        self.migrate_on_access = True  # Migrate JSON data when accessed
        self.keep_json_backup = True   # Keep JSON files as backup during migration
        
        # Performance settings
        self.cache_size = 1000
        self.batch_migration_size = 100
        self.performance_monitoring = True
        
        # Storage paths
        self.json_storage_path = "./json_storage"
        self.optimized_storage_path = "./optimized_storage"
        self.backup_path = "./storage_backup"


class CompatibilityStorageAdapter:
    """
    Adapter that provides backward compatibility for existing xp_core components
    while using optimized storage underneath.
    """
    
    def __init__(self, config: Optional[StorageIntegrationConfig] = None):
        """Initialize the compatibility adapter."""
        self.config = config or StorageIntegrationConfig()
        
        # Initialize optimized storage if enabled
        if self.config.enable_optimized_storage:
            self.optimized_storage = OptimizedStorageBackend(
                self.config.optimized_storage_path
            )
            self.use_optimized = True
            logger.info("Optimized storage enabled")
        else:
            self.optimized_storage = None
            self.use_optimized = False
            logger.info("Using legacy JSON storage")
        
        # Legacy JSON storage paths
        self.json_path = Path(self.config.json_storage_path)
        self.json_path.mkdir(parents=True, exist_ok=True)
        
        # Migration tracking
        self.migration_stats = {
            'units_migrated': 0,
            'migration_errors': 0,
            'json_files_processed': 0,
            'total_size_saved': 0
        }
        
        # Performance monitoring
        self.performance_stats = {
            'read_operations': 0,
            'write_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_read_time': 0.0,
            'avg_write_time': 0.0
        }
    
    def store_memory_entry(self, entry: MemoryEntry) -> bool:
        """
        Store a MemoryEntry with optimized storage.
        
        Maintains compatibility with existing xp_core logic.
        """
        start_time = time.time()
        
        try:
            if self.use_optimized:
                # Convert MemoryEntry to HRRUnitData
                unit_data = self._memory_entry_to_hrr_unit(entry)
                success = self.optimized_storage.store_unit(entry.id, unit_data)
            else:
                # Legacy JSON storage
                success = self._store_memory_entry_json(entry)
            
            # Update performance stats
            write_time = time.time() - start_time
            self.performance_stats['write_operations'] += 1
            self.performance_stats['avg_write_time'] = (
                (self.performance_stats['avg_write_time'] * 
                 (self.performance_stats['write_operations'] - 1) + write_time) /
                self.performance_stats['write_operations']
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to store memory entry {entry.id}: {e}")
            return False
    
    def load_memory_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """
        Load a MemoryEntry with automatic migration from JSON if needed.
        """
        start_time = time.time()
        
        try:
            memory_entry = None
            
            if self.use_optimized:
                # Try optimized storage first
                unit_data = self.optimized_storage.load_unit(entry_id)
                if unit_data:
                    memory_entry = self._hrr_unit_to_memory_entry(unit_data)
                    self.performance_stats['cache_hits'] += 1
                else:
                    self.performance_stats['cache_misses'] += 1
                    
                    # Try JSON storage and migrate if found
                    if self.config.migrate_on_access:
                        json_entry = self._load_memory_entry_json(entry_id)
                        if json_entry:
                            # Migrate to optimized storage
                            if self.store_memory_entry(json_entry):
                                self.migration_stats['units_migrated'] += 1
                                logger.info(f"Migrated memory entry {entry_id} to optimized storage")
                            memory_entry = json_entry
            else:
                # Legacy JSON storage only
                memory_entry = self._load_memory_entry_json(entry_id)
            
            # Update performance stats
            read_time = time.time() - start_time
            self.performance_stats['read_operations'] += 1
            self.performance_stats['avg_read_time'] = (
                (self.performance_stats['avg_read_time'] * 
                 (self.performance_stats['read_operations'] - 1) + read_time) /
                self.performance_stats['read_operations']
            )
            
            return memory_entry
            
        except Exception as e:
            logger.error(f"Failed to load memory entry {entry_id}: {e}")
            return None
    
    def store_spatial_unit(self, unit_id: str, spatial_unit: SpatialUnit) -> bool:
        """Store a SpatialUnit with optimized storage."""
        start_time = time.time()
        
        try:
            if self.use_optimized:
                # Convert SpatialUnit to HRRUnitData
                unit_data = self._spatial_unit_to_hrr_unit(spatial_unit)
                success = self.optimized_storage.store_unit(unit_id, unit_data)
            else:
                # Legacy JSON storage
                success = self._store_spatial_unit_json(unit_id, spatial_unit)
            
            self.performance_stats['write_operations'] += 1
            return success
            
        except Exception as e:
            logger.error(f"Failed to store spatial unit {unit_id}: {e}")
            return False
    
    def load_spatial_unit(self, unit_id: str) -> Optional[SpatialUnit]:
        """Load a SpatialUnit with automatic migration."""
        start_time = time.time()
        
        try:
            spatial_unit = None
            
            if self.use_optimized:
                # Try optimized storage first
                unit_data = self.optimized_storage.load_unit(unit_id)
                if unit_data:
                    spatial_unit = self._hrr_unit_to_spatial_unit(unit_data)
                elif self.config.migrate_on_access:
                    # Try JSON and migrate
                    json_unit = self._load_spatial_unit_json(unit_id)
                    if json_unit:
                        if self.store_spatial_unit(unit_id, json_unit):
                            self.migration_stats['units_migrated'] += 1
                        spatial_unit = json_unit
            else:
                spatial_unit = self._load_spatial_unit_json(unit_id)
            
            self.performance_stats['read_operations'] += 1
            return spatial_unit
            
        except Exception as e:
            logger.error(f"Failed to load spatial unit {unit_id}: {e}")
            return None
    
    def _memory_entry_to_hrr_unit(self, entry: MemoryEntry) -> HRRUnitData:
        """Convert MemoryEntry to HRRUnitData."""
        unit_data = HRRUnitData(
            unit_id=entry.id,
            content_hash="",  # Will be calculated in store_unit
            created_timestamp=entry.timestamp.timestamp(),
            last_accessed=time.time(),
            dimension=len(entry.embedding) if entry.embedding is not None else 384
        )
        
        # Use embedding as bound vector
        if entry.embedding is not None:
            unit_data.bound_vector = np.array(entry.embedding, dtype=np.float32)
        
        # Store metadata
        unit_data.metadata = entry.metadata.copy()
        unit_data.source = entry.metadata.get('source', 'memory_system')
        
        # Extract tags from metadata
        if 'tags' in entry.metadata:
            unit_data.tags = entry.metadata['tags']
        
        return unit_data
    
    def _hrr_unit_to_memory_entry(self, unit_data: HRRUnitData) -> MemoryEntry:
        """Convert HRRUnitData back to MemoryEntry."""
        from datetime import datetime
        
        # Create MemoryEntry
        entry = MemoryEntry(
            content=unit_data.metadata.get('content', ''),
            embedding=unit_data.bound_vector.tolist() if unit_data.bound_vector is not None else None,
            metadata=unit_data.metadata.copy(),
            timestamp=datetime.fromtimestamp(unit_data.created_timestamp)
        )
        
        # Override ID to maintain consistency
        entry.id = unit_data.unit_id
        
        return entry
    
    def _spatial_unit_to_hrr_unit(self, spatial_unit: SpatialUnit) -> HRRUnitData:
        """Convert SpatialUnit to HRRUnitData."""
        unit_data = HRRUnitData(
            unit_id="",  # Will be set by caller
            content_hash="",
            created_timestamp=getattr(spatial_unit, 'last_update', time.time()),
            last_accessed=time.time(),
            dimension=len(spatial_unit.attribute_vector) if spatial_unit.attribute_vector is not None else 256
        )
        
        # Copy vectors
        if spatial_unit.semantic_vector is not None:
            unit_data.role_vector = spatial_unit.semantic_vector.astype(np.float32)
        if spatial_unit.temporal_vector is not None:
            unit_data.filler_vector = spatial_unit.temporal_vector.astype(np.float32)
        if spatial_unit.attribute_vector is not None:
            unit_data.bound_vector = spatial_unit.attribute_vector.astype(np.float32)
        if spatial_unit.emotional_vector is not None:
            unit_data.context_vector = spatial_unit.emotional_vector.astype(np.float32)
        
        # Copy spatial properties
        unit_data.semantic_strength = getattr(spatial_unit, 'semantic_strength', 0.0)
        unit_data.temporal_strength = getattr(spatial_unit, 'temporal_strength', 0.0)
        unit_data.emotional_strength = getattr(spatial_unit, 'emotional_strength', 0.0)
        unit_data.structural_strength = getattr(spatial_unit, 'structural_strength', 0.0)
        
        # Copy metadata from memory entry if available
        if spatial_unit.memory_entry:
            unit_data.metadata = spatial_unit.memory_entry.metadata.copy()
            unit_data.source = unit_data.metadata.get('source', 'spatial_environment')
        
        return unit_data
    
    def _hrr_unit_to_spatial_unit(self, unit_data: HRRUnitData) -> SpatialUnit:
        """Convert HRRUnitData back to SpatialUnit."""
        # Create a minimal MemoryEntry for compatibility
        memory_entry = MemoryEntry(
            content=unit_data.metadata.get('content', ''),
            embedding=unit_data.bound_vector.tolist() if unit_data.bound_vector is not None else None,
            metadata=unit_data.metadata.copy()
        )
        memory_entry.id = unit_data.unit_id
        
        # Create SpatialUnit
        spatial_unit = SpatialUnit(
            memory_entry=memory_entry,
            semantic_vector=unit_data.role_vector,
            temporal_vector=unit_data.filler_vector,
            emotional_vector=unit_data.context_vector,
            attribute_vector=unit_data.bound_vector
        )
        
        # Restore spatial properties
        spatial_unit.semantic_strength = unit_data.semantic_strength
        spatial_unit.temporal_strength = unit_data.temporal_strength
        spatial_unit.emotional_strength = unit_data.emotional_strength
        spatial_unit.structural_strength = unit_data.structural_strength
        spatial_unit.last_update = unit_data.last_accessed
        
        return spatial_unit
    
    def _store_memory_entry_json(self, entry: MemoryEntry) -> bool:
        """Legacy JSON storage for MemoryEntry."""
        try:
            json_file = self.json_path / f"memory_{entry.id}.json"
            
            data = {
                'id': entry.id,
                'content': entry.content,
                'embedding': entry.embedding,
                'metadata': entry.metadata,
                'timestamp': entry.timestamp.isoformat(),
                'access_count': entry.access_count
            }
            
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store memory entry JSON {entry.id}: {e}")
            return False
    
    def _load_memory_entry_json(self, entry_id: str) -> Optional[MemoryEntry]:
        """Legacy JSON loading for MemoryEntry."""
        try:
            json_file = self.json_path / f"memory_{entry_id}.json"
            
            if not json_file.exists():
                return None
            
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            from datetime import datetime
            
            entry = MemoryEntry(
                content=data['content'],
                embedding=data['embedding'],
                metadata=data['metadata'],
                timestamp=datetime.fromisoformat(data['timestamp'])
            )
            entry.id = data['id']
            entry.access_count = data.get('access_count', 0)
            
            return entry
            
        except Exception as e:
            logger.error(f"Failed to load memory entry JSON {entry_id}: {e}")
            return None
    
    def _store_spatial_unit_json(self, unit_id: str, spatial_unit: SpatialUnit) -> bool:
        """Legacy JSON storage for SpatialUnit."""
        try:
            json_file = self.json_path / f"spatial_{unit_id}.json"
            
            data = {
                'unit_id': unit_id,
                'vectors': {},
                'properties': {
                    'activation_level': spatial_unit.activation_level,
                    'spatial_energy': spatial_unit.spatial_energy,
                    'last_update': getattr(spatial_unit, 'last_update', time.time()),
                    'cluster_id': spatial_unit.cluster_id
                },
                'neighbors': spatial_unit.neighbors,
                'metadata': {}
            }
            
            # Store vectors as lists (JSON-serializable)
            if spatial_unit.semantic_vector is not None:
                data['vectors']['semantic'] = spatial_unit.semantic_vector.tolist()
            if spatial_unit.temporal_vector is not None:
                data['vectors']['temporal'] = spatial_unit.temporal_vector.tolist()
            if spatial_unit.emotional_vector is not None:
                data['vectors']['emotional'] = spatial_unit.emotional_vector.tolist()
            if spatial_unit.attribute_vector is not None:
                data['vectors']['attribute'] = spatial_unit.attribute_vector.tolist()
            
            # Store memory entry metadata if available
            if spatial_unit.memory_entry:
                data['metadata'] = spatial_unit.memory_entry.metadata.copy()
            
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store spatial unit JSON {unit_id}: {e}")
            return False
    
    def _load_spatial_unit_json(self, unit_id: str) -> Optional[SpatialUnit]:
        """Legacy JSON loading for SpatialUnit."""
        try:
            json_file = self.json_path / f"spatial_{unit_id}.json"
            
            if not json_file.exists():
                return None
            
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Create minimal MemoryEntry
            memory_entry = MemoryEntry(
                content=data['metadata'].get('content', ''),
                embedding=data['vectors'].get('attribute', []),
                metadata=data['metadata']
            )
            memory_entry.id = unit_id
            
            # Create SpatialUnit
            spatial_unit = SpatialUnit(memory_entry=memory_entry)
            
            # Restore vectors
            vectors = data['vectors']
            if 'semantic' in vectors:
                spatial_unit.semantic_vector = np.array(vectors['semantic'], dtype=np.float32)
            if 'temporal' in vectors:
                spatial_unit.temporal_vector = np.array(vectors['temporal'], dtype=np.float32)
            if 'emotional' in vectors:
                spatial_unit.emotional_vector = np.array(vectors['emotional'], dtype=np.float32)
            if 'attribute' in vectors:
                spatial_unit.attribute_vector = np.array(vectors['attribute'], dtype=np.float32)
            
            # Restore properties
            props = data['properties']
            spatial_unit.activation_level = props.get('activation_level', 0.0)
            spatial_unit.spatial_energy = props.get('spatial_energy', 0.0)
            spatial_unit.last_update = props.get('last_update', time.time())
            spatial_unit.cluster_id = props.get('cluster_id')
            spatial_unit.neighbors = data.get('neighbors', {})
            
            return spatial_unit
            
        except Exception as e:
            logger.error(f"Failed to load spatial unit JSON {unit_id}: {e}")
            return None
    
    def get_migration_stats(self) -> Dict[str, Any]:
        """Get migration statistics."""
        return self.migration_stats.copy()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()
    
    def get_storage_metrics(self) -> Optional[StorageMetrics]:
        """Get storage metrics from optimized backend."""
        if self.optimized_storage:
            return self.optimized_storage.get_storage_metrics()
        return None
    
    def migrate_all_json_data(self) -> Dict[str, Any]:
        """
        Migrate all existing JSON data to optimized storage.
        
        Returns:
            Migration results and statistics
        """
        if not self.use_optimized:
            return {'error': 'Optimized storage not enabled'}
        
        results = {
            'memory_entries_migrated': 0,
            'spatial_units_migrated': 0,
            'errors': 0,
            'total_size_saved': 0,
            'migration_time': 0
        }
        
        start_time = time.time()
        
        try:
            # Migrate memory entries
            for json_file in self.json_path.glob("memory_*.json"):
                try:
                    entry_id = json_file.stem.replace('memory_', '')
                    entry = self._load_memory_entry_json(entry_id)
                    
                    if entry and self.store_memory_entry(entry):
                        results['memory_entries_migrated'] += 1
                        
                        # Calculate size savings
                        json_size = json_file.stat().st_size
                        results['total_size_saved'] += json_size * 0.7  # Estimated savings
                        
                        if self.config.keep_json_backup:
                            # Move to backup instead of deleting
                            backup_path = Path(self.config.backup_path)
                            backup_path.mkdir(parents=True, exist_ok=True)
                            json_file.rename(backup_path / json_file.name)
                        else:
                            json_file.unlink()
                            
                except Exception as e:
                    logger.error(f"Failed to migrate memory entry {json_file}: {e}")
                    results['errors'] += 1
            
            # Migrate spatial units
            for json_file in self.json_path.glob("spatial_*.json"):
                try:
                    unit_id = json_file.stem.replace('spatial_', '')
                    spatial_unit = self._load_spatial_unit_json(unit_id)
                    
                    if spatial_unit and self.store_spatial_unit(unit_id, spatial_unit):
                        results['spatial_units_migrated'] += 1
                        
                        json_size = json_file.stat().st_size
                        results['total_size_saved'] += json_size * 0.7
                        
                        if self.config.keep_json_backup:
                            backup_path = Path(self.config.backup_path)
                            backup_path.mkdir(parents=True, exist_ok=True)
                            json_file.rename(backup_path / json_file.name)
                        else:
                            json_file.unlink()
                            
                except Exception as e:
                    logger.error(f"Failed to migrate spatial unit {json_file}: {e}")
                    results['errors'] += 1
            
            results['migration_time'] = time.time() - start_time
            
            logger.info(f"Migration complete: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            results['error'] = str(e)
            return results
    
    def close(self):
        """Close storage backends and cleanup."""
        if self.optimized_storage:
            self.optimized_storage.close()


class EnhancedSpatialEnvironment(SpatialEnvironment):
    """
    Enhanced SpatialEnvironment that uses optimized storage while maintaining
    full compatibility with existing xp_core logic.
    """
    
    def __init__(self, dimension: int = 256, decay_rate: float = 0.1, 
                 storage_config: Optional[StorageIntegrationConfig] = None):
        """Initialize enhanced spatial environment with optimized storage."""
        super().__init__(dimension, decay_rate)
        
        # Initialize storage adapter
        self.storage_adapter = CompatibilityStorageAdapter(storage_config)
        
        # Override units storage to use adapter
        self._units_storage: Dict[str, str] = {}  # unit_id -> storage_key mapping
        
        logger.info("Enhanced SpatialEnvironment initialized with optimized storage")
    
    def add_unit(self, unit_id: str, attributes: Dict[str, float]) -> bool:
        """Add a unit to the spatial environment with optimized storage."""
        try:
            # Create SpatialUnit as before
            memory_entry = MemoryEntry(
                content=f"Spatial unit {unit_id}",
                embedding=None,
                metadata={'attributes': attributes, 'unit_id': unit_id}
            )
            
            spatial_unit = SpatialUnit(
                memory_entry=memory_entry,
                activation_level=0.0,
                spatial_energy=1.0
            )
            
            # Initialize vectors based on attributes
            if 'semantic' in attributes:
                spatial_unit.semantic_vector = np.random.random(self.dimension) * attributes['semantic']
            if 'temporal' in attributes:
                spatial_unit.temporal_vector = np.random.random(self.dimension) * attributes['temporal']
            if 'emotional' in attributes:
                spatial_unit.emotional_vector = np.random.random(self.dimension) * attributes['emotional']
            if 'structural' in attributes:
                spatial_unit.structural_vector = np.random.random(self.dimension) * attributes['structural']
            
            # Create composite attribute vector
            vectors = []
            if spatial_unit.semantic_vector is not None:
                vectors.append(spatial_unit.semantic_vector)
            if spatial_unit.temporal_vector is not None:
                vectors.append(spatial_unit.temporal_vector)
            if spatial_unit.emotional_vector is not None:
                vectors.append(spatial_unit.emotional_vector)
            if spatial_unit.structural_vector is not None:
                vectors.append(spatial_unit.structural_vector)
            
            if vectors:
                spatial_unit.attribute_vector = normalize_vector(np.mean(vectors, axis=0))
            
            # Store using optimized storage
            success = self.storage_adapter.store_spatial_unit(unit_id, spatial_unit)
            
            if success:
                # Keep reference in memory for compatibility
                self.units[unit_id] = spatial_unit
                self._units_storage[unit_id] = unit_id
                
                logger.info(f"Added spatial unit {unit_id} with optimized storage")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to add spatial unit {unit_id}: {e}")
            return False
    
    def get_unit(self, unit_id: str) -> Optional[SpatialUnit]:
        """Get a spatial unit with automatic loading from optimized storage."""
        # Check memory first
        if unit_id in self.units:
            return self.units[unit_id]
        
        # Load from storage
        spatial_unit = self.storage_adapter.load_spatial_unit(unit_id)
        
        if spatial_unit:
            # Cache in memory
            self.units[unit_id] = spatial_unit
            self._units_storage[unit_id] = unit_id
        
        return spatial_unit
    
    def get_relationships(self) -> List[Tuple[str, str, float]]:
        """Get unit relationships with optimized storage support."""
        relationships = []
        
        # Ensure all units are loaded
        for unit_id in list(self._units_storage.keys()):
            if unit_id not in self.units:
                self.get_unit(unit_id)
        
        # Calculate relationships as before
        unit_ids = list(self.units.keys())
        
        for i, unit_a_id in enumerate(unit_ids):
            unit_a = self.units[unit_a_id]
            if unit_a.attribute_vector is None:
                continue
                
            for unit_b_id in unit_ids[i+1:]:
                unit_b = self.units[unit_b_id]
                if unit_b.attribute_vector is None:
                    continue
                
                # Calculate similarity
                similarity = np.dot(unit_a.attribute_vector, unit_b.attribute_vector)
                
                if similarity > 0.1:  # Threshold for meaningful relationships
                    relationships.append((unit_a_id, unit_b_id, similarity))
        
        return relationships
    
    def get_storage_metrics(self) -> Optional[StorageMetrics]:
        """Get storage performance metrics."""
        return self.storage_adapter.get_storage_metrics()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.storage_adapter.get_performance_stats()
    
    def migrate_to_optimized_storage(self) -> Dict[str, Any]:
        """Migrate existing data to optimized storage."""
        return self.storage_adapter.migrate_all_json_data()
    
    def close(self):
        """Close storage backends."""
        self.storage_adapter.close()


# Example usage and testing
if __name__ == "__main__":
    # Test the integration
    config = StorageIntegrationConfig()
    config.enable_optimized_storage = True
    
    # Create enhanced spatial environment
    env = EnhancedSpatialEnvironment(dimension=256, storage_config=config)
    
    # Add test units
    for i in range(10):
        attributes = {
            'semantic': np.random.random(),
            'temporal': np.random.random(),
            'emotional': np.random.random(),
            'structural': np.random.random()
        }
        env.add_unit(f"test_unit_{i}", attributes)
    
    # Test relationships
    relationships = env.get_relationships()
    print(f"Found {len(relationships)} relationships")
    
    # Get performance metrics
    metrics = env.get_storage_metrics()
    if metrics:
        print(f"Storage metrics: {metrics}")
    
    stats = env.get_performance_stats()
    print(f"Performance stats: {stats}")
    
    # Close
    env.close()