"""
Holographic Reduced Representations (HRR) for Lumina Memory

This module implements Vector Symbolic Architecture operations for creating
holographic memory representations. HRRs enable:
- Binding operations for associative memory
- Superposition for memory consolidation
- Circular convolution for structured representations
- Memory fingerprinting and similarity detection

Design Principles:
- Mathematical rigor: Proper VSA algebra implementation
- Efficient operations: NumPy-based vectorized computations
- Deterministic behavior: Reproducible results with same inputs
- Composable operations: Build complex representations from simple ones

References:
- Plate, T. A. (2003). Holographic reduced representations.
- Kanerva, P. (2009). Hyperdimensional computing.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import hashlib
from scipy.fft import fft, ifft
import warnings


@dataclass
class HRRVector:
    """
    Holographic Reduced Representation vector with operations.
    
    Encapsulates a high-dimensional vector with HRR-specific operations
    for binding, unbinding, and similarity computation.
    """
    vector: np.ndarray          # High-dimensional vector
    dimension: int              # Vector dimensionality
    name: Optional[str] = None  # Optional human-readable name
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata
    
    def __post_init__(self):
        """Validate HRR vector after creation."""
        if self.vector.ndim != 1:
            raise ValueError("HRR vector must be 1-dimensional")
        if len(self.vector) != self.dimension:
            raise ValueError(f"Vector length {len(self.vector)} != dimension {self.dimension}")
        if self.metadata is None:
            self.metadata = {}
    
    def normalize(self) -> 'HRRVector':
        """Normalize vector to unit length."""
        norm = np.linalg.norm(self.vector)
        if norm == 0:
            warnings.warn("Cannot normalize zero vector")
            return self
        
        normalized_vector = self.vector / norm
        return HRRVector(
            vector=normalized_vector,
            dimension=self.dimension,
            name=f"norm({self.name})" if self.name else None,
            metadata={**self.metadata, 'normalized': True}
        )
    
    def similarity(self, other: 'HRRVector') -> float:
        """Compute cosine similarity with another HRR vector."""
        if self.dimension != other.dimension:
            raise ValueError(f"Dimension mismatch: {self.dimension} != {other.dimension}")
        
        # Compute cosine similarity
        dot_product = np.dot(self.vector, other.vector)
        norm_self = np.linalg.norm(self.vector)
        norm_other = np.linalg.norm(other.vector)
        
        if norm_self == 0 or norm_other == 0:
            return 0.0
        
        return dot_product / (norm_self * norm_other)
    
    def bind(self, other: 'HRRVector') -> 'HRRVector':
        """
        Bind this vector with another using circular convolution.
        
        Binding creates associative connections between concepts.
        If A is bound to B to create C, then C can be unbound with A to retrieve B.
        """
        if self.dimension != other.dimension:
            raise ValueError(f"Dimension mismatch: {self.dimension} != {other.dimension}")
        
        # Circular convolution via FFT
        fft_self = fft(self.vector)
        fft_other = fft(other.vector)
        bound_fft = fft_self * fft_other
        bound_vector = np.real(ifft(bound_fft))
        
        # Create name for bound vector
        name_self = self.name or "unknown"
        name_other = other.name or "unknown"
        bound_name = f"({name_self}  {name_other})"
        
        return HRRVector(
            vector=bound_vector,
            dimension=self.dimension,
            name=bound_name,
            metadata={
                'operation': 'bind',
                'operands': [name_self, name_other],
                'source_metadata': [self.metadata, other.metadata]
            }
        )
    
    def unbind(self, other: 'HRRVector') -> 'HRRVector':
        """
        Unbind this vector with another using circular correlation.
        
        Unbinding retrieves associated information. If C = A  B,
        then C  A  B (approximately).
        """
        if self.dimension != other.dimension:
            raise ValueError(f"Dimension mismatch: {self.dimension} != {other.dimension}")
        
        # Circular correlation via FFT (conjugate of one operand)
        fft_self = fft(self.vector)
        fft_other_conj = np.conj(fft(other.vector))
        unbound_fft = fft_self * fft_other_conj
        unbound_vector = np.real(ifft(unbound_fft))
        
        # Create name for unbound vector
        name_self = self.name or "unknown"
        name_other = other.name or "unknown"
        unbound_name = f"({name_self}  {name_other})"
        
        return HRRVector(
            vector=unbound_vector,
            dimension=self.dimension,
            name=unbound_name,
            metadata={
                'operation': 'unbind',
                'operands': [name_self, name_other],
                'source_metadata': [self.metadata, other.metadata]
            }
        )
    
    def superpose(self, other: 'HRRVector', weight_self: float = 1.0, weight_other: float = 1.0) -> 'HRRVector':
        """
        Superpose (add) this vector with another.
        
        Superposition creates memory sets. Elements can be retrieved
        from the superposition using similarity queries.
        """
        if self.dimension != other.dimension:
            raise ValueError(f"Dimension mismatch: {self.dimension} != {other.dimension}")
        
        superposed_vector = weight_self * self.vector + weight_other * other.vector
        
        # Create name for superposed vector
        name_self = self.name or "unknown"
        name_other = other.name or "unknown"
        superposed_name = f"({name_self} + {name_other})"
        
        return HRRVector(
            vector=superposed_vector,
            dimension=self.dimension,
            name=superposed_name,
            metadata={
                'operation': 'superpose',
                'operands': [name_self, name_other],
                'weights': [weight_self, weight_other],
                'source_metadata': [self.metadata, other.metadata]
            }
        )
    
    def permute(self, permutation: Optional[np.ndarray] = None) -> 'HRRVector':
        """
        Permute vector elements to create role/filler distinction.
        
        Permutation can be used to represent different roles in structured
        representations (e.g., subject vs object in relations).
        """
        if permutation is None:
            # Create default circular shift permutation
            permutation = np.roll(np.arange(self.dimension), 1)
        
        if len(permutation) != self.dimension:
            raise ValueError(f"Permutation length {len(permutation)} != dimension {self.dimension}")
        
        permuted_vector = self.vector[permutation]
        
        return HRRVector(
            vector=permuted_vector,
            dimension=self.dimension,
            name=f"perm({self.name})" if self.name else None,
            metadata={**self.metadata, 'operation': 'permute'}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert HRR vector to dictionary for serialization."""
        return {
            'vector': self.vector.tolist(),
            'dimension': self.dimension,
            'name': self.name,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HRRVector':
        """Create HRR vector from dictionary."""
        return cls(
            vector=np.array(data['vector']),
            dimension=data['dimension'],
            name=data.get('name'),
            metadata=data.get('metadata', {})
        )


def generate_random_hrr(dimension: int, name: Optional[str] = None, seed: Optional[int] = None) -> HRRVector:
    """
    Generate random HRR vector with specified properties.
    
    Args:
        dimension: Vector dimensionality
        name: Optional name for the vector
        seed: Optional random seed for reproducibility
        
    Returns:
        Random HRR vector
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random vector from normal distribution
    vector = np.random.normal(0, 1/np.sqrt(dimension), dimension)
    
    return HRRVector(
        vector=vector,
        dimension=dimension,
        name=name,
        metadata={'type': 'random', 'seed': seed}
    )


def encode_sequence(items: List[HRRVector], position_vectors: Optional[List[HRRVector]] = None) -> HRRVector:
    """
    Encode a sequence of items using position binding.
    
    Each item is bound with its position vector and then all are superposed.
    This allows the sequence order to be preserved and queried.
    
    Args:
        items: List of HRR vectors to encode in sequence
        position_vectors: Optional position vectors (generated if None)
        
    Returns:
        HRR vector representing the sequence
    """
    if not items:
        raise ValueError("Cannot encode empty sequence")
    
    dimension = items[0].dimension
    
    # Generate position vectors if not provided
    if position_vectors is None:
        position_vectors = [
            generate_random_hrr(dimension, f"pos_{i}", seed=i * 1000)
            for i in range(len(items))
        ]
    
    if len(position_vectors) != len(items):
        raise ValueError("Position vectors length must match items length")
    
    # Bind each item with its position and accumulate
    sequence_vector = None
    
    for i, (item, pos) in enumerate(zip(items, position_vectors)):
        bound_item = item.bind(pos)
        
        if sequence_vector is None:
            sequence_vector = bound_item
        else:
            sequence_vector = sequence_vector.superpose(bound_item)
    
    sequence_vector.name = f"sequence_{len(items)}_items"
    sequence_vector.metadata = {
        'type': 'sequence',
        'length': len(items),
        'item_names': [item.name for item in items]
    }
    
    return sequence_vector


def create_memory_fingerprint(
    content: str,
    embedding: np.ndarray,
    dimension: int = 1024,
    metadata: Optional[Dict[str, Any]] = None
) -> HRRVector:
    """
    Create HRR-based fingerprint for memory content.
    
    Combines content hash with embedding information to create
    a holographic fingerprint that captures both semantic and
    structural properties of the memory.
    
    Args:
        content: Memory content string
        embedding: Embedding vector
        dimension: HRR vector dimension
        metadata: Optional metadata
        
    Returns:
        HRR fingerprint vector
    """
    # Create content-based seed from hash
    content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
    content_seed = int(content_hash[:8], 16) % (2**31 - 1)
    
    # Generate content HRR vector
    content_hrr = generate_random_hrr(dimension, "content", seed=content_seed)
    
    # Convert embedding to HRR dimension if needed
    if len(embedding) != dimension:
        # Pad or truncate embedding to match dimension
        if len(embedding) < dimension:
            padded_embedding = np.pad(embedding, (0, dimension - len(embedding)))
        else:
            padded_embedding = embedding[:dimension]
        
        embedding_hrr = HRRVector(
            vector=padded_embedding,
            dimension=dimension,
            name="embedding",
            metadata={'source': 'memory_embedding'}
        )
    else:
        embedding_hrr = HRRVector(
            vector=embedding,
            dimension=dimension,
            name="embedding",
            metadata={'source': 'memory_embedding'}
        )
    
    # Bind content HRR with embedding HRR to create fingerprint
    fingerprint = content_hrr.bind(embedding_hrr.normalize())
    fingerprint.name = "memory_fingerprint"
    fingerprint.metadata = {
        'type': 'memory_fingerprint',
        'content_hash': content_hash,
        'embedding_dimension': len(embedding),
        'memory_metadata': metadata or {}
    }
    
    return fingerprint


def compute_hrr_similarity_matrix(vectors: List[HRRVector]) -> np.ndarray:
    """
    Compute similarity matrix between all pairs of HRR vectors.
    
    Args:
        vectors: List of HRR vectors
        
    Returns:
        Symmetric similarity matrix
    """
    n = len(vectors)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):  # Only compute upper triangle
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                sim = vectors[i].similarity(vectors[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim  # Symmetric
    
    return similarity_matrix


class HRRMemoryBank:
    """
    Memory bank using HRR for associative storage and retrieval.
    
    Provides high-level interface for storing and querying memories
    using holographic reduced representations.
    """
    
    def __init__(self, dimension: int = 1024):
        """Initialize HRR memory bank with specified dimension."""
        self.dimension = dimension
        self.memories: Dict[str, HRRVector] = {}
        self.associations: Dict[str, List[str]] = {}  # Track associations
        
    def store_memory(
        self,
        memory_id: str,
        content: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> HRRVector:
        """Store memory with HRR fingerprint."""
        fingerprint = create_memory_fingerprint(content, embedding, self.dimension, metadata)
        fingerprint.name = f"memory_{memory_id}"
        fingerprint.metadata['memory_id'] = memory_id
        
        self.memories[memory_id] = fingerprint
        return fingerprint
    
    def query_similar(self, query_vector: HRRVector, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """
        Query for similar memories using HRR similarity.
        
        Returns:
            List of (memory_id, similarity_score) tuples above threshold
        """
        results = []
        
        for memory_id, memory_vector in self.memories.items():
            similarity = query_vector.similarity(memory_vector)
            if similarity >= threshold:
                results.append((memory_id, similarity))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def create_association(self, memory_id1: str, memory_id2: str) -> Optional[HRRVector]:
        """Create associative binding between two memories."""
        if memory_id1 not in self.memories or memory_id2 not in self.memories:
            return None
        
        # Bind the two memory vectors
        mem1 = self.memories[memory_id1]
        mem2 = self.memories[memory_id2]
        association = mem1.bind(mem2)
        
        association_id = f"{memory_id1}_{memory_id2}_assoc"
        association.name = association_id
        
        # Store association and update tracking
        self.memories[association_id] = association
        
        if memory_id1 not in self.associations:
            self.associations[memory_id1] = []
        if memory_id2 not in self.associations:
            self.associations[memory_id2] = []
        
        self.associations[memory_id1].append(association_id)
        self.associations[memory_id2].append(association_id)
        
        return association
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory bank statistics."""
        return {
            'dimension': self.dimension,
            'total_memories': len(self.memories),
            'total_associations': len(self.associations),
            'memory_ids': list(self.memories.keys())
        }
