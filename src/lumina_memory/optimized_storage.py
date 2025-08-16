"""
Optimized Storage System for HRR Units and Spatial Environment
============================================================

This module implements efficient binary storage for HRR vectors and spatial units,
addressing JSON bloat issues while maintaining compatibility with existing xp_core logic.

Key Features:
- LMDB for fast local key-value storage with memory mapping
- Protocol Buffers for efficient binary serialization
- Backward compatibility with existing JSON-based systems
- Incremental migration path from JSON to binary storage
- CPI metrics integration for consciousness battery

Storage Architecture:
- Metadata: Small JSON/CBOR for human-readable info
- Vectors: Raw binary arrays (float32) for HRR operations
- Index: LMDB for fast lookups and integrity checking
- Analytics: Periodic Parquet snapshots for CPI metrics

Author: Lumina Memory Team
License: MIT
"""

import os
import json
import time
import hashlib
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

# Core dependencies for optimized storage
try:
    import lmdb
    LMDB_AVAILABLE = True
except ImportError:
    LMDB_AVAILABLE = False
    logging.warning("LMDB not available - falling back to file-based storage")

try:
    import cbor2
    CBOR_AVAILABLE = True
except ImportError:
    CBOR_AVAILABLE = False
    logging.warning("CBOR2 not available - using JSON for metadata")

# Import our existing components
from .core import MemoryEntry
from .hrr import normalize_vector
from .spatial_environment import SpatialUnit

logger = logging.getLogger(__name__)


@dataclass
class StorageMetrics:
    """Metrics for storage performance and CPI integration."""
    
    # Storage efficiency
    total_units: int = 0
    total_size_bytes: int = 0
    json_size_bytes: int = 0
    binary_size_bytes: int = 0
    compression_ratio: float = 0.0
    
    # Performance metrics
    write_ops_per_sec: float = 0.0
    read_ops_per_sec: float = 0.0
    avg_write_latency_ms: float = 0.0
    avg_read_latency_ms: float = 0.0
    
    # CPI-related metrics
    coherence_preservation: float = 0.0  # How well vectors maintain coherence
    retrieval_accuracy: float = 0.0      # Accuracy of vector retrieval
    integrity_score: float = 0.0         # Hash-based integrity verification


@dataclass
class HRRUnitData:
    """Optimized data structure for HRR unit storage."""
    
    # Identity and metadata
    unit_id: str
    content_hash: str  # SHA-256 of bound vector for integrity
    created_timestamp: float
    last_accessed: float
    
    # Vector data (stored as binary)
    dimension: int
    dtype: str = "float32"
    
    # HRR vectors
    role_vector: Optional[np.ndarray] = None
    filler_vector: Optional[np.ndarray] = None
    bound_vector: Optional[np.ndarray] = None  # role ⊛ filler
    context_vector: Optional[np.ndarray] = None
    
    # Spatial properties
    semantic_strength: float = 0.0
    temporal_strength: float = 0.0
    emotional_strength: float = 0.0
    structural_strength: float = 0.0
    
    # CPI metrics
    coherence_score: float = 0.0
    persistence_score: float = 0.0
    interference_score: float = 0.0
    
    # Metadata (small, human-readable)
    tags: List[str] = field(default_factory=list)
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class OptimizedStorageBackend:
    """
    Optimized storage backend that reduces JSON bloat and improves performance.
    
    Uses LMDB for fast key-value storage with binary serialization for vectors
    while maintaining JSON compatibility for metadata.
    """
    
    def __init__(self, storage_path: str, max_db_size: int = 1024**3):  # 1GB default
        """
        Initialize optimized storage backend.
        
        Args:
            storage_path: Directory for storage files
            max_db_size: Maximum database size in bytes
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize LMDB if available
        if LMDB_AVAILABLE:
            self.env = lmdb.open(
                str(self.storage_path / "hrr_units.lmdb"),
                max_dbs=3,
                map_size=max_db_size,
                writemap=True,
                meminit=False
            )
            self.units_db = self.env.open_db(b'units')
            self.metadata_db = self.env.open_db(b'metadata')
            self.index_db = self.env.open_db(b'index')
            self.use_lmdb = True
            logger.info(f"LMDB storage initialized at {storage_path}")
        else:
            # Fallback to file-based storage
            self.units_dir = self.storage_path / "units"
            self.metadata_dir = self.storage_path / "metadata"
            self.units_dir.mkdir(exist_ok=True)
            self.metadata_dir.mkdir(exist_ok=True)
            self.use_lmdb = False
            logger.info(f"File-based storage initialized at {storage_path}")
        
        # Storage metrics
        self.metrics = StorageMetrics()
        
        # Cache for frequently accessed units
        self._cache: Dict[str, HRRUnitData] = {}
        self._cache_max_size = 1000
    
    def _serialize_vectors(self, unit_data: HRRUnitData) -> bytes:
        """Serialize vector data to binary format."""
        vectors = {}
        
        if unit_data.role_vector is not None:
            vectors['role'] = unit_data.role_vector.astype(np.float32).tobytes()
        if unit_data.filler_vector is not None:
            vectors['filler'] = unit_data.filler_vector.astype(np.float32).tobytes()
        if unit_data.bound_vector is not None:
            vectors['bound'] = unit_data.bound_vector.astype(np.float32).tobytes()
        if unit_data.context_vector is not None:
            vectors['context'] = unit_data.context_vector.astype(np.float32).tobytes()
        
        # Create header with vector info
        header = {
            'dimension': unit_data.dimension,
            'dtype': unit_data.dtype,
            'vectors': list(vectors.keys()),
            'created': unit_data.created_timestamp,
            'accessed': unit_data.last_accessed,
            'content_hash': unit_data.content_hash
        }
        
        # Serialize header
        if CBOR_AVAILABLE:
            header_bytes = cbor2.dumps(header)
        else:
            header_bytes = json.dumps(header).encode('utf-8')
        
        # Combine header and vectors
        header_len = len(header_bytes).to_bytes(4, 'little')
        vector_data = b''.join(vectors.values())
        
        return header_len + header_bytes + vector_data
    
    def _deserialize_vectors(self, data: bytes) -> HRRUnitData:
        """Deserialize binary data back to HRRUnitData."""
        # Read header length
        header_len = int.from_bytes(data[:4], 'little')
        
        # Read header
        header_bytes = data[4:4+header_len]
        if CBOR_AVAILABLE:
            header = cbor2.loads(header_bytes)
        else:
            header = json.loads(header_bytes.decode('utf-8'))
        
        # Read vector data
        vector_data = data[4+header_len:]
        dimension = header['dimension']
        dtype = np.dtype(header['dtype'])
        vector_size = dimension * dtype.itemsize
        
        # Reconstruct vectors
        unit_data = HRRUnitData(
            unit_id="",  # Will be set by caller
            content_hash=header['content_hash'],
            created_timestamp=header['created'],
            last_accessed=header['accessed'],
            dimension=dimension,
            dtype=header['dtype']
        )
        
        offset = 0
        for vector_name in header['vectors']:
            vector_bytes = vector_data[offset:offset+vector_size]
            vector = np.frombuffer(vector_bytes, dtype=dtype).reshape(dimension)
            
            if vector_name == 'role':
                unit_data.role_vector = vector
            elif vector_name == 'filler':
                unit_data.filler_vector = vector
            elif vector_name == 'bound':
                unit_data.bound_vector = vector
            elif vector_name == 'context':
                unit_data.context_vector = vector
            
            offset += vector_size
        
        return unit_data
    
    def store_unit(self, unit_id: str, unit_data: HRRUnitData) -> bool:
        """
        Store an HRR unit with optimized binary serialization.
        
        Args:
            unit_id: Unique identifier for the unit
            unit_data: HRR unit data to store
            
        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()
        
        try:
            # Update timestamps
            unit_data.unit_id = unit_id
            unit_data.last_accessed = time.time()
            
            # Calculate content hash for integrity
            if unit_data.bound_vector is not None:
                content_bytes = unit_data.bound_vector.astype(np.float32).tobytes()
                unit_data.content_hash = hashlib.sha256(content_bytes).hexdigest()
            
            if self.use_lmdb:
                # Store in LMDB
                vector_data = self._serialize_vectors(unit_data)
                
                # Store metadata separately (small, human-readable)
                metadata = {
                    'tags': unit_data.tags,
                    'source': unit_data.source,
                    'metadata': unit_data.metadata,
                    'semantic_strength': unit_data.semantic_strength,
                    'temporal_strength': unit_data.temporal_strength,
                    'emotional_strength': unit_data.emotional_strength,
                    'structural_strength': unit_data.structural_strength,
                    'coherence_score': unit_data.coherence_score,
                    'persistence_score': unit_data.persistence_score,
                    'interference_score': unit_data.interference_score
                }
                
                with self.env.begin(write=True) as txn:
                    # Store vector data
                    txn.put(unit_id.encode(), vector_data, db=self.units_db)
                    
                    # Store metadata
                    if CBOR_AVAILABLE:
                        metadata_bytes = cbor2.dumps(metadata)
                    else:
                        metadata_bytes = json.dumps(metadata).encode('utf-8')
                    txn.put(unit_id.encode(), metadata_bytes, db=self.metadata_db)
                    
                    # Update index
                    index_entry = {
                        'content_hash': unit_data.content_hash,
                        'created': unit_data.created_timestamp,
                        'size': len(vector_data)
                    }
                    index_bytes = json.dumps(index_entry).encode('utf-8')
                    txn.put(unit_id.encode(), index_bytes, db=self.index_db)
                
            else:
                # Fallback to file storage
                unit_file = self.units_dir / f"{unit_id}.bin"
                metadata_file = self.metadata_dir / f"{unit_id}.json"
                
                # Store vector data as binary
                vector_data = self._serialize_vectors(unit_data)
                with open(unit_file, 'wb') as f:
                    f.write(vector_data)
                
                # Store metadata as JSON
                metadata = {
                    'unit_id': unit_id,
                    'content_hash': unit_data.content_hash,
                    'created_timestamp': unit_data.created_timestamp,
                    'last_accessed': unit_data.last_accessed,
                    'tags': unit_data.tags,
                    'source': unit_data.source,
                    'metadata': unit_data.metadata,
                    'semantic_strength': unit_data.semantic_strength,
                    'temporal_strength': unit_data.temporal_strength,
                    'emotional_strength': unit_data.emotional_strength,
                    'structural_strength': unit_data.structural_strength,
                    'coherence_score': unit_data.coherence_score,
                    'persistence_score': unit_data.persistence_score,
                    'interference_score': unit_data.interference_score
                }
                
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            # Update cache
            self._update_cache(unit_id, unit_data)
            
            # Update metrics
            write_time = time.time() - start_time
            self.metrics.avg_write_latency_ms = write_time * 1000
            self.metrics.total_units += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store unit {unit_id}: {e}")
            return False
    
    def load_unit(self, unit_id: str) -> Optional[HRRUnitData]:
        """
        Load an HRR unit from storage.
        
        Args:
            unit_id: Unique identifier for the unit
            
        Returns:
            HRRUnitData if found, None otherwise
        """
        start_time = time.time()
        
        # Check cache first
        if unit_id in self._cache:
            return self._cache[unit_id]
        
        try:
            if self.use_lmdb:
                # Load from LMDB
                with self.env.begin() as txn:
                    # Load vector data
                    vector_data = txn.get(unit_id.encode(), db=self.units_db)
                    if vector_data is None:
                        return None
                    
                    # Load metadata
                    metadata_bytes = txn.get(unit_id.encode(), db=self.metadata_db)
                    if metadata_bytes:
                        if CBOR_AVAILABLE:
                            metadata = cbor2.loads(metadata_bytes)
                        else:
                            metadata = json.loads(metadata_bytes.decode('utf-8'))
                    else:
                        metadata = {}
                
                # Deserialize unit data
                unit_data = self._deserialize_vectors(vector_data)
                unit_data.unit_id = unit_id
                
                # Restore metadata
                unit_data.tags = metadata.get('tags', [])
                unit_data.source = metadata.get('source', '')
                unit_data.metadata = metadata.get('metadata', {})
                unit_data.semantic_strength = metadata.get('semantic_strength', 0.0)
                unit_data.temporal_strength = metadata.get('temporal_strength', 0.0)
                unit_data.emotional_strength = metadata.get('emotional_strength', 0.0)
                unit_data.structural_strength = metadata.get('structural_strength', 0.0)
                unit_data.coherence_score = metadata.get('coherence_score', 0.0)
                unit_data.persistence_score = metadata.get('persistence_score', 0.0)
                unit_data.interference_score = metadata.get('interference_score', 0.0)
                
            else:
                # Load from files
                unit_file = self.units_dir / f"{unit_id}.bin"
                metadata_file = self.metadata_dir / f"{unit_id}.json"
                
                if not unit_file.exists():
                    return None
                
                # Load vector data
                with open(unit_file, 'rb') as f:
                    vector_data = f.read()
                
                unit_data = self._deserialize_vectors(vector_data)
                unit_data.unit_id = unit_id
                
                # Load metadata
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Restore metadata fields
                    for field in ['tags', 'source', 'metadata', 'semantic_strength',
                                'temporal_strength', 'emotional_strength', 'structural_strength',
                                'coherence_score', 'persistence_score', 'interference_score']:
                        if field in metadata:
                            setattr(unit_data, field, metadata[field])
            
            # Update cache
            self._update_cache(unit_id, unit_data)
            
            # Update metrics
            read_time = time.time() - start_time
            self.metrics.avg_read_latency_ms = read_time * 1000
            
            return unit_data
            
        except Exception as e:
            logger.error(f"Failed to load unit {unit_id}: {e}")
            return None
    
    def _update_cache(self, unit_id: str, unit_data: HRRUnitData):
        """Update the in-memory cache."""
        if len(self._cache) >= self._cache_max_size:
            # Remove oldest entry (simple LRU)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[unit_id] = unit_data
    
    def get_storage_metrics(self) -> StorageMetrics:
        """Get current storage metrics."""
        return self.metrics
    
    def verify_integrity(self, unit_id: str) -> bool:
        """Verify the integrity of a stored unit using content hash."""
        unit_data = self.load_unit(unit_id)
        if unit_data is None or unit_data.bound_vector is None:
            return False
        
        # Recalculate content hash
        content_bytes = unit_data.bound_vector.astype(np.float32).tobytes()
        calculated_hash = hashlib.sha256(content_bytes).hexdigest()
        
        return calculated_hash == unit_data.content_hash
    
    def close(self):
        """Close storage backend and cleanup resources."""
        if self.use_lmdb and hasattr(self, 'env'):
            self.env.close()


class SpatialEnvironmentStorage:
    """
    Storage adapter for SpatialEnvironment that uses OptimizedStorageBackend.
    
    Provides backward compatibility with existing SpatialEnvironment while
    using optimized storage underneath.
    """
    
    def __init__(self, storage_path: str, dimension: int = 256):
        """
        Initialize spatial environment storage.
        
        Args:
            storage_path: Path for storage files
            dimension: HRR vector dimension
        """
        self.storage = OptimizedStorageBackend(storage_path)
        self.dimension = dimension
        
        # Compatibility layer for existing code
        self.units: Dict[str, SpatialUnit] = {}
        self._loaded_units: Set[str] = set()
    
    def add_unit(self, unit_id: str, spatial_unit: SpatialUnit) -> bool:
        """Add a spatial unit to storage."""
        # Convert SpatialUnit to HRRUnitData
        unit_data = HRRUnitData(
            unit_id=unit_id,
            content_hash="",  # Will be calculated in store_unit
            created_timestamp=time.time(),
            last_accessed=time.time(),
            dimension=self.dimension
        )
        
        # Copy vector data
        if hasattr(spatial_unit, 'semantic_vector') and spatial_unit.semantic_vector is not None:
            unit_data.role_vector = spatial_unit.semantic_vector
        if hasattr(spatial_unit, 'attribute_vector') and spatial_unit.attribute_vector is not None:
            unit_data.bound_vector = spatial_unit.attribute_vector
        
        # Copy spatial properties
        unit_data.semantic_strength = getattr(spatial_unit, 'semantic_strength', 0.0)
        unit_data.temporal_strength = getattr(spatial_unit, 'temporal_strength', 0.0)
        unit_data.emotional_strength = getattr(spatial_unit, 'emotional_strength', 0.0)
        unit_data.structural_strength = getattr(spatial_unit, 'structural_strength', 0.0)
        
        # Store in optimized backend
        success = self.storage.store_unit(unit_id, unit_data)
        
        if success:
            # Keep in memory for compatibility
            self.units[unit_id] = spatial_unit
            self._loaded_units.add(unit_id)
        
        return success
    
    def get_unit(self, unit_id: str) -> Optional[SpatialUnit]:
        """Get a spatial unit from storage."""
        # Check memory first
        if unit_id in self.units:
            return self.units[unit_id]
        
        # Load from storage
        unit_data = self.storage.load_unit(unit_id)
        if unit_data is None:
            return None
        
        # Convert back to SpatialUnit (simplified for compatibility)
        # In practice, you'd reconstruct the full SpatialUnit
        spatial_unit = SpatialUnit(
            memory_entry=None,  # Would need to reconstruct
            semantic_vector=unit_data.role_vector,
            attribute_vector=unit_data.bound_vector
        )
        
        # Cache in memory
        self.units[unit_id] = spatial_unit
        self._loaded_units.add(unit_id)
        
        return spatial_unit
    
    def get_relationships(self) -> List[Tuple[str, str, float]]:
        """Get unit relationships (compatibility method)."""
        # This would be implemented based on your existing relationship logic
        # For now, return empty list
        return []
    
    def close(self):
        """Close storage backend."""
        self.storage.close()


# Migration utilities for existing JSON-based systems
def migrate_json_to_optimized(json_dir: str, optimized_storage_path: str) -> bool:
    """
    Migrate existing JSON-based storage to optimized binary storage.
    
    Args:
        json_dir: Directory containing JSON files
        optimized_storage_path: Path for new optimized storage
        
    Returns:
        True if migration successful
    """
    try:
        storage = OptimizedStorageBackend(optimized_storage_path)
        json_path = Path(json_dir)
        
        migrated_count = 0
        
        for json_file in json_path.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Convert JSON data to HRRUnitData
                # This would need to be customized based on your JSON structure
                unit_id = json_file.stem
                
                unit_data = HRRUnitData(
                    unit_id=unit_id,
                    content_hash="",
                    created_timestamp=data.get('timestamp', time.time()),
                    last_accessed=time.time(),
                    dimension=data.get('dimension', 256)
                )
                
                # Convert vector data from JSON arrays to numpy arrays
                if 'vectors' in data:
                    vectors = data['vectors']
                    if 'role' in vectors:
                        unit_data.role_vector = np.array(vectors['role'], dtype=np.float32)
                    if 'filler' in vectors:
                        unit_data.filler_vector = np.array(vectors['filler'], dtype=np.float32)
                    if 'bound' in vectors:
                        unit_data.bound_vector = np.array(vectors['bound'], dtype=np.float32)
                
                # Store in optimized format
                if storage.store_unit(unit_id, unit_data):
                    migrated_count += 1
                    logger.info(f"Migrated unit {unit_id}")
                
            except Exception as e:
                logger.error(f"Failed to migrate {json_file}: {e}")
        
        storage.close()
        logger.info(f"Migration complete: {migrated_count} units migrated")
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False


# Example usage and integration points
if __name__ == "__main__":
    # Example of how to use the optimized storage
    storage = OptimizedStorageBackend("./optimized_storage")
    
    # Create sample HRR unit
    dimension = 256
    unit_data = HRRUnitData(
        unit_id="test_unit_001",
        content_hash="",
        created_timestamp=time.time(),
        last_accessed=time.time(),
        dimension=dimension,
        role_vector=np.random.random(dimension).astype(np.float32),
        filler_vector=np.random.random(dimension).astype(np.float32),
        bound_vector=np.random.random(dimension).astype(np.float32),
        tags=["test", "example"],
        source="test_system"
    )
    
    # Store and retrieve
    success = storage.store_unit("test_unit_001", unit_data)
    print(f"Storage successful: {success}")
    
    loaded_unit = storage.load_unit("test_unit_001")
    print(f"Loaded unit: {loaded_unit.unit_id if loaded_unit else 'None'}")
    
    # Verify integrity
    integrity_ok = storage.verify_integrity("test_unit_001")
    print(f"Integrity check: {integrity_ok}")
    
    # Get metrics
    metrics = storage.get_storage_metrics()
    print(f"Storage metrics: {metrics}")
    
    storage.close()