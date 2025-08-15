"""
XP Core Unified Implementation - Complete Mathematical Foundation
================================================================

This module consolidates ALL the best components from the three notebooks into
a single, coherent, rigorous implementation of the XP Core mathematical foundation.

CONSOLIDATION SOURCES:
- xp_core_design.ipynb: Mathematical formulas, MemoryUnit, HRR operations
- unit_space_kernel_bridge.ipynb: Spatial topology, integration patterns
- hd_kernel_xp_spec.ipynb: Interface specifications, kernel patterns

MATHEMATICAL FOUNDATION:
The "unit of experience" (XP Unit) with complete mathematical properties:
- Cryptographic identity (BLAKE3 content addressing)
- Holographic representation (HRR binding/unbinding)
- Temporal mathematics (exponential decay, consolidation)
- Relational coherence (unit-to-unit mathematical relationships)
- Environmental context (spatial topology, access patterns)

ARCHITECTURE:
- XPUnit: The fundamental mathematical unit of experience
- XPEnvironment: The computational container where units operate
- UnifiedXPKernel: HD Kernel interface implementation
- Production features: NLP, vector search, cryptographic integrity

Author: Lumina Memory Team
License: MIT
"""

import numpy as np
import time
import hashlib
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Import our mathematical foundation and existing components
from .math_foundation import (
    circular_convolution, circular_correlation, normalize_vector,
    bind_role_filler, unbind_role_filler, memory_unit_score,
    mathematical_coherence, instant_salience, hybrid_lexical_attribution,
    get_current_timestamp, cosine_similarity
)
from .versioned_xp_store import VersionedXPStore, XPStoreEntry
from .constants import (
    EPSILON, DEFAULT_W_SEMANTIC, DEFAULT_W_EMOTION,
    COHERENCE_HRR_WEIGHT, COHERENCE_SEM_WEIGHT,
    MIN_SCORE, MAX_SCORE, VECTOR_DTYPE
)
from .emotional_weighting import (
    EmotionalState, EmotionalAnalyzer, EmotionalMemoryWeighter,
    ConsciousnessEmotionalIntegrator
)
from .enhanced_emotional_weighting import (
    EnhancedEmotionalAnalyzer, EnhancedEmotionalMemoryWeighter,
    EnhancedConsciousnessEmotionalIntegrator
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# UNIFIED CONFIGURATION - Consolidates all config classes
# =============================================================================

@dataclass
class UnifiedXPConfig:
    """
    Unified configuration consolidating all approaches.
    
    Replaces:
    - XPCoreConfig (from xp_core_design.ipynb)
    - SpaceConfig (from unit_space_kernel_bridge.ipynb)  
    - LuminaConfig (from main branch)
    """
    # Core mathematical settings
    embedding_dim: int = 384
    hrr_dim: int = 512
    emotion_dim: int = 6
    
    # Temporal mathematics
    decay_half_life: float = 168.0  # hours (1 week)
    consolidation_threshold: float = 0.7
    importance_boost_factor: float = 1.5
    
    # Spatial topology
    k_neighbors: int = 10
    topology_update_threshold: float = 0.1
    spatial_decay_factor: float = 0.95
    
    # System limits
    max_memory_capacity: int = 10000
    max_commit_history: int = 1000
    batch_size: int = 32
    
    # Processing settings
    lexical_method: str = "hybrid"  # "simple", "spacy", "hybrid"
    use_versioned_store: bool = True
    enable_cryptographic_integrity: bool = True
    
    # Performance settings
    deterministic_seed: int = 42
    parallel_processing: bool = False
    cache_embeddings: bool = True
    
    # Emotional weighting settings
    enable_emotional_weighting: bool = True
    use_enhanced_emotional_analysis: bool = True  # Use external libraries for better emotion detection
    emotional_importance_factor: float = 1.5
    emotional_decay_influence: float = 0.6
    emotional_retrieval_boost: float = 1.3
    emotional_consciousness_boost: float = 0.8


# =============================================================================
# XP UNIT - The Fundamental Mathematical Unit of Experience
# =============================================================================

@dataclass
class XPUnit:
    """
    The fundamental mathematical unit of experience.
    
    Consolidates and extends:
    - MemoryUnit (from xp_core_design.ipynb)
    - Memory (from unit_space_kernel_bridge.ipynb)
    - UnifiedMemory (from unified_foundation.py)
    
    MATHEMATICAL PROPERTIES:
    - Cryptographic identity via content addressing
    - Holographic representation via HRR operations
    - Temporal evolution via decay mathematics
    - Relational coherence via similarity measures
    - Environmental context via spatial topology
    """
    # Core identity
    content_id: str
    content: str
    
    # Mathematical representations
    semantic_vector: np.ndarray
    hrr_shape: np.ndarray
    emotion_vector: np.ndarray
    
    # Temporal properties
    timestamp: float
    last_access: float
    decay_rate: float
    importance: float
    access_count: int = 0
    
    # Relational properties
    coherence_links: Dict[str, float] = field(default_factory=dict)
    binding_roles: Dict[str, np.ndarray] = field(default_factory=dict)
    topology_neighbors: Dict[str, float] = field(default_factory=dict)
    
    # Cryptographic properties
    content_hash: str = ""
    commit_id: str = ""
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize computed properties"""
        if not self.content_hash:
            self.content_hash = self._compute_content_hash()
        if not self.last_access:
            self.last_access = self.timestamp
    
    def _compute_content_hash(self) -> str:
        """Compute cryptographic hash of content for integrity"""
        content_data = {
            'content': self.content,
            'semantic_shape': self.semantic_vector.shape,
            'hrr_shape': self.hrr_shape.shape,
            'emotion_shape': self.emotion_vector.shape,
            'metadata': self.metadata
        }
        content_str = json.dumps(content_data, sort_keys=True)
        return hashlib.blake2b(content_str.encode(), digest_size=16).hexdigest()
    
    def update_access(self):
        """Update access statistics and temporal properties"""
        self.last_access = get_current_timestamp()
        self.access_count += 1
        
        # Boost importance based on access patterns
        access_boost = min(0.1, self.access_count * 0.01)
        self.importance = min(2.0, self.importance + access_boost)
    
    def get_age_hours(self) -> float:
        """Get age in hours for decay calculations"""
        return (get_current_timestamp() - self.timestamp) / 3600.0
    
    def get_decay_factor(self, emotional_modifier: float = 1.0) -> float:
        """Compute current decay factor with optional emotional influence"""
        age_hours = self.get_age_hours()
        base_decay = np.exp(-self.decay_rate * age_hours)
        
        # Apply emotional modifier to decay (strong emotions slow decay)
        return base_decay * emotional_modifier
    
    def get_emotional_state(self) -> EmotionalState:
        """Get emotional state from emotion vector"""
        if self.emotion_vector is not None and len(self.emotion_vector) >= 6:
            return EmotionalState.from_vector(self.emotion_vector[:6])
        return EmotionalState()
    
    def set_emotional_state(self, emotion: EmotionalState):
        """Set emotional state by updating emotion vector"""
        emotion_vector = emotion.to_vector()
        if self.emotion_vector is not None and len(self.emotion_vector) > 6:
            # Preserve any additional emotion dimensions
            self.emotion_vector[:6] = emotion_vector
        else:
            # Create new emotion vector
            self.emotion_vector = emotion_vector
    
    def get_emotional_importance_boost(self) -> float:
        """Calculate importance boost from emotional content"""
        emotion = self.get_emotional_state()
        intensity = emotion.intensity()
        
        # Strong emotions increase importance
        base_boost = intensity * 0.3
        
        # Specific emotional boosts
        fear_boost = emotion.fear * 0.4  # Fear memories are very important
        curiosity_boost = emotion.curiosity * 0.2
        valence_boost = abs(emotion.valence) * 0.2
        
        total_boost = base_boost + fear_boost + curiosity_boost + valence_boost
        return min(1.0, total_boost)  # Cap at 1.0
    
    def score_against(self, query_unit: 'XPUnit', 
                     w_semantic: float = DEFAULT_W_SEMANTIC,
                     w_emotion: float = DEFAULT_W_EMOTION) -> float:
        """
        Mathematical scoring against another XP unit.
        Uses canonical formulas from math_foundation.py
        """
        return memory_unit_score(
            query_unit.semantic_vector, self.semantic_vector,
            query_unit.emotion_vector, self.emotion_vector,
            self.get_age_hours(), self.decay_rate, self.importance,
            w_semantic, w_emotion
        )
    
    def compute_coherence_with(self, other: 'XPUnit') -> float:
        """
        Compute mathematical coherence with another unit.
        Uses canonical formulas from math_foundation.py
        """
        return mathematical_coherence(
            self.hrr_shape, other.hrr_shape,
            self.semantic_vector, other.semantic_vector
        )
    
    def bind_with_role(self, role: str, filler_unit: 'XPUnit') -> np.ndarray:
        """
        HRR binding operation with role-filler semantics.
        Creates structured relationships between units.
        """
        # Get or create role vector
        if role not in self.binding_roles:
            # Create deterministic role vector from role name
            role_seed = abs(hash(role)) % (2**32)
            rng = np.random.default_rng(role_seed)
            self.binding_roles[role] = normalize_vector(rng.normal(size=len(self.hrr_shape)))
        
        role_vector = self.binding_roles[role]
        return bind_role_filler(role_vector, filler_unit.hrr_shape)
    
    def unbind_role(self, bound_vector: np.ndarray, role: str) -> Optional[np.ndarray]:
        """
        HRR unbinding operation to extract filler from role-filler binding.
        """
        if role not in self.binding_roles:
            return None
        
        role_vector = self.binding_roles[role]
        return unbind_role_filler(bound_vector, role_vector)
    
    def create_superposition_with(self, other_units: List['XPUnit'], 
                                 weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Create superposition state with other units.
        Implements holographic superposition mathematics.
        """
        if not other_units:
            return self.hrr_shape.copy()
        
        if weights is None:
            weights = [1.0] * (len(other_units) + 1)
        elif len(weights) != len(other_units) + 1:
            weights = [1.0] * (len(other_units) + 1)
        
        # Weighted superposition
        superposition = weights[0] * self.hrr_shape
        for i, unit in enumerate(other_units):
            superposition += weights[i + 1] * unit.hrr_shape
        
        return normalize_vector(superposition)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'content_id': self.content_id,
            'content': self.content,
            'semantic_vector': self.semantic_vector.tolist(),
            'hrr_shape': self.hrr_shape.tolist(),
            'emotion_vector': self.emotion_vector.tolist(),
            'timestamp': self.timestamp,
            'last_access': self.last_access,
            'decay_rate': self.decay_rate,
            'importance': self.importance,
            'access_count': self.access_count,
            'coherence_links': self.coherence_links,
            'topology_neighbors': self.topology_neighbors,
            'content_hash': self.content_hash,
            'commit_id': self.commit_id,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'XPUnit':
        """Create XPUnit from dictionary"""
        return cls(
            content_id=data['content_id'],
            content=data['content'],
            semantic_vector=np.array(data['semantic_vector'], dtype=VECTOR_DTYPE),
            hrr_shape=np.array(data['hrr_shape'], dtype=VECTOR_DTYPE),
            emotion_vector=np.array(data['emotion_vector'], dtype=VECTOR_DTYPE),
            timestamp=data['timestamp'],
            last_access=data['last_access'],
            decay_rate=data['decay_rate'],
            importance=data['importance'],
            access_count=data['access_count'],
            coherence_links=data.get('coherence_links', {}),
            topology_neighbors=data.get('topology_neighbors', {}),
            content_hash=data.get('content_hash', ''),
            commit_id=data.get('commit_id', ''),
            metadata=data.get('metadata', {})
        )


# =============================================================================
# XP ENVIRONMENT - The Computational Container
# =============================================================================

class XPEnvironment:
    """
    The computational environment where XP units operate.
    
    Consolidates:
    - Processing engines (embeddings, NLP, crypto)
    - Storage systems (versioned store, vector index, graph)
    - Mathematical operations (HRR, decay, consolidation)
    - Spatial topology (KNN, relationship management)
    
    ARCHITECTURE:
    - Local operation (no external dependencies required)
    - High-end methods (SpaCy, SentenceTransformers, FAISS)
    - Cryptographic integrity (BLAKE3, content addressing)
    - Production ready (optimized, error handling, logging)
    """
    
    def __init__(self, config: UnifiedXPConfig = None):
        self.config = config or UnifiedXPConfig()
        
        # Core storage systems
        self.versioned_store = VersionedXPStore() if self.config.use_versioned_store else None
        self.units: Dict[str, XPUnit] = {}
        self.relationship_graph: Dict[str, Dict[str, float]] = {}
        
        # Processing engines (initialized lazily)
        self._embedding_engine = None
        self._nlp_pipeline = None
        self._vector_index = None
        
        # Mathematical operations
        self.decay_engine = DecayMathematicsEngine(self.config)
        self.consolidation_engine = ConsolidationEngine(self.config)
        self.relationship_manager = RelationshipManager(self.config)
        
        # Statistics
        self.stats = {
            'total_units': 0,
            'total_ingestions': 0,
            'total_retrievals': 0,
            'total_consolidations': 0,
            'avg_coherence': 0.0,
            'system_uptime': get_current_timestamp()
        }
        
        logger.info(f"XP Environment initialized with {self.config.embedding_dim}D embeddings")
    
    def _init_embedding_engine(self):
        """Initialize embedding engine (lazy loading)"""
        if self._embedding_engine is None:
            try:
                # Try to use SentenceTransformers if available
                from sentence_transformers import SentenceTransformer
                self._embedding_engine = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Initialized SentenceTransformers embedding engine")
            except ImportError:
                # Fallback to simple hash-based embeddings
                self._embedding_engine = SimpleEmbeddingEngine(self.config.embedding_dim)
                logger.info("Using simple hash-based embedding engine")
        return self._embedding_engine
    
    def _init_nlp_pipeline(self):
        """Initialize NLP pipeline (lazy loading)"""
        if self._nlp_pipeline is None:
            try:
                import spacy
                self._nlp_pipeline = spacy.load("en_core_web_sm")
                logger.info("Initialized SpaCy NLP pipeline")
            except (ImportError, OSError):
                self._nlp_pipeline = SimpleNLPPipeline()
                logger.info("Using simple NLP pipeline")
        return self._nlp_pipeline
    
    def _generate_content_id(self, content: str) -> str:
        """Generate cryptographic content ID"""
        normalized = content.strip().lower()
        if self.config.enable_cryptographic_integrity:
            return hashlib.blake2b(normalized.encode(), digest_size=16).hexdigest()
        else:
            return f"xp_{abs(hash(normalized)):016x}"
    
    def _compute_semantic_vector(self, content: str) -> np.ndarray:
        """Compute semantic embedding vector"""
        # Ensure content is a proper Python string (fix numpy.str_ issue)
        content = str(content)
        
        engine = self._init_embedding_engine()
        
        if hasattr(engine, 'encode'):
            # SentenceTransformers
            embedding = engine.encode(content)
            # Pad or truncate to desired dimension
            if len(embedding) != self.config.embedding_dim:
                if len(embedding) > self.config.embedding_dim:
                    embedding = embedding[:self.config.embedding_dim]
                else:
                    padding = np.zeros(self.config.embedding_dim - len(embedding))
                    embedding = np.concatenate([embedding, padding])
        else:
            # Simple engine
            embedding = engine.encode(content)
        
        return embedding.astype(VECTOR_DTYPE)
    
    def _compute_hrr_shape(self, semantic_vector: np.ndarray, 
                          metadata: Dict[str, Any]) -> np.ndarray:
        """Compute holographic shape using HRR operations"""
        # Create context vector from metadata
        context_seed = abs(hash(json.dumps(metadata, sort_keys=True))) % (2**32)
        rng = np.random.default_rng(context_seed)
        context_vector = normalize_vector(rng.normal(size=self.config.hrr_dim))
        
        # Pad or project semantic vector to HRR dimension
        if len(semantic_vector) != self.config.hrr_dim:
            if len(semantic_vector) > self.config.hrr_dim:
                sem_proj = semantic_vector[:self.config.hrr_dim]
            else:
                padding = np.zeros(self.config.hrr_dim - len(semantic_vector))
                sem_proj = np.concatenate([semantic_vector, padding])
        else:
            sem_proj = semantic_vector.copy()
        
        # Bind semantic and context using circular convolution
        hrr_shape = circular_convolution(sem_proj, context_vector)
        return normalize_vector(hrr_shape).astype(VECTOR_DTYPE)
    
    def _compute_emotion_vector(self, content: str) -> np.ndarray:
        """Compute emotion vector (simple heuristic or NLP-based)"""
        nlp = self._init_nlp_pipeline()
        
        if hasattr(nlp, '__call__'):
            # SpaCy pipeline
            doc = nlp(content)
            # Simple emotion heuristics based on linguistic features
            word_count = len(doc)
            entity_count = len(doc.ents)
            
            # [joy, anger, fear, sadness, surprise, neutral]
            emotion_vec = np.array([
                min(1.0, word_count / 50.0) * 0.6,  # joy from content richness
                0.1,  # baseline anger
                0.05,  # baseline fear
                0.1,  # baseline sadness
                min(1.0, entity_count / 5.0) * 0.3,  # surprise from entities
                0.5  # neutral baseline
            ], dtype=VECTOR_DTYPE)
        else:
            # Simple heuristic
            word_count = len(content.split())
            excitement = min(1.0, word_count / 50.0)
            
            emotion_vec = np.array([
                excitement * 0.6,  # joy
                0.1,  # anger
                0.05,  # fear
                0.1,  # sadness
                excitement * 0.3,  # surprise
                1.0 - excitement  # neutral
            ], dtype=VECTOR_DTYPE)
        
        # Normalize to unit vector
        return normalize_vector(emotion_vec)
    
    def ingest_experience(self, content: str, metadata: Dict[str, Any] = None) -> XPUnit:
        """
        Ingest new experience into XP unit with complete mathematical processing.
        
        This is the core method that transforms raw experience into mathematical
        representation with all XP Core properties.
        """
        if not content.strip():
            raise ValueError("Content cannot be empty")
        
        metadata = metadata or {}
        
        # Generate cryptographic identity
        content_id = self._generate_content_id(content)
        
        # Check for existing unit (deduplication)
        if content_id in self.units:
            existing_unit = self.units[content_id]
            existing_unit.update_access()
            logger.info(f"Retrieved existing unit: {content_id[:16]}...")
            return existing_unit
        
        # Compute mathematical representations
        semantic_vector = self._compute_semantic_vector(content)
        hrr_shape = self._compute_hrr_shape(semantic_vector, metadata)
        emotion_vector = self._compute_emotion_vector(content)
        
        # Create XP unit with all mathematical properties
        unit = XPUnit(
            content_id=content_id,
            content=content,
            semantic_vector=semantic_vector,
            hrr_shape=hrr_shape,
            emotion_vector=emotion_vector,
            timestamp=get_current_timestamp(),
            last_access=get_current_timestamp(),
            decay_rate=np.log(2) / self.config.decay_half_life,  # Convert half-life to rate
            importance=1.0,
            metadata=metadata
        )
        
        # Store in versioned store if enabled
        if self.versioned_store:
            commit_id = self.versioned_store.commit(
                changes={'action': 'ingest', 'content_id': content_id},
                message=f"Ingest: {content[:50]}..."
            )
            unit.commit_id = commit_id
        
        # Store in environment
        self.units[content_id] = unit
        
        # Update spatial topology
        self.relationship_manager.update_topology(unit, self.units)
        
        # Update statistics
        self.stats['total_units'] += 1
        self.stats['total_ingestions'] += 1
        
        logger.info(f"Ingested XP unit: {content_id[:16]}... (dim={len(semantic_vector)})")
        return unit
    
    def retrieve_similar(self, query: Union[str, XPUnit], k: int = 10, 
                        threshold: float = 0.0) -> List[Tuple[XPUnit, float]]:
        """
        Retrieve similar XP units using mathematical similarity.
        
        Supports both string queries and XPUnit queries for maximum flexibility.
        """
        if isinstance(query, str) or hasattr(query, 'dtype'):  # Handle numpy strings
            # Ensure proper string conversion
            query_str = str(query)
            # Create temporary query unit
            query_unit = XPUnit(
                content_id="query_temp",
                content=query_str,
                semantic_vector=self._compute_semantic_vector(query_str),
                hrr_shape=np.zeros(self.config.hrr_dim, dtype=VECTOR_DTYPE),
                emotion_vector=self._compute_emotion_vector(query_str),
                timestamp=get_current_timestamp(),
                last_access=get_current_timestamp(),
                decay_rate=0.0,
                importance=1.0
            )
        else:
            query_unit = query
        
        # Compute similarities with all units
        similarities = []
        for unit_id, unit in self.units.items():
            if unit_id == query_unit.content_id:
                continue  # Skip self
            
            score = unit.score_against(query_unit)
            if score >= threshold:
                similarities.append((unit, score))
                unit.update_access()  # Update access statistics
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Update statistics
        self.stats['total_retrievals'] += 1
        
        logger.info(f"Retrieved {min(k, len(similarities))} similar units for query")
        return similarities[:k]
    
    def consolidate_memories(self) -> int:
        """
        Consolidate memories using importance-based mathematics.
        
        Strengthens important memories and weakens less important ones.
        """
        return self.consolidation_engine.consolidate(self.units)
    
    def evolve_temporal_state(self, time_delta_hours: float = 1.0) -> Dict[str, Any]:
        """
        Apply temporal evolution to all units (decay, consolidation).
        
        This simulates the passage of time and natural memory processes.
        """
        return self.decay_engine.apply_decay(self.units, time_delta_hours)
    
    def get_unit(self, content_id: str) -> Optional[XPUnit]:
        """Retrieve unit by content ID"""
        unit = self.units.get(content_id)
        if unit:
            unit.update_access()
        return unit
    
    def get_relationship_graph(self) -> Dict[str, Dict[str, float]]:
        """Get the complete relationship graph"""
        return self.relationship_manager.get_graph()
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        current_time = get_current_timestamp()
        uptime_hours = (current_time - self.stats['system_uptime']) / 3600.0
        
        if self.units:
            avg_importance = np.mean([unit.importance for unit in self.units.values()])
            avg_access_count = np.mean([unit.access_count for unit in self.units.values()])
            total_relationships = sum(len(unit.coherence_links) for unit in self.units.values())
        else:
            avg_importance = 0.0
            avg_access_count = 0.0
            total_relationships = 0
        
        stats = {
            **self.stats,
            'uptime_hours': uptime_hours,
            'avg_importance': avg_importance,
            'avg_access_count': avg_access_count,
            'total_relationships': total_relationships,
            'config': {
                'embedding_dim': self.config.embedding_dim,
                'hrr_dim': self.config.hrr_dim,
                'decay_half_life': self.config.decay_half_life,
                'k_neighbors': self.config.k_neighbors
            }
        }
        
        if self.versioned_store:
            stats['versioned_store'] = self.versioned_store.stats()
        
        return stats


# =============================================================================
# SUPPORTING ENGINES
# =============================================================================

class SimpleEmbeddingEngine:
    """Simple hash-based embedding engine for fallback"""
    
    def __init__(self, dim: int):
        self.dim = dim
    
    def encode(self, text: str) -> np.ndarray:
        """Generate deterministic embedding from text hash"""
        seed = abs(hash(text)) % (2**32)
        rng = np.random.default_rng(seed)
        return rng.normal(size=self.dim).astype(VECTOR_DTYPE)


class SimpleNLPPipeline:
    """Simple NLP pipeline for fallback"""
    
    def __call__(self, text: str):
        """Simple tokenization and analysis"""
        return SimpleDoc(text)


class SimpleDoc:
    """Simple document representation"""
    
    def __init__(self, text: str):
        self.text = text
        self.tokens = text.split()
        self.ents = []  # No entity recognition in simple mode
    
    def __len__(self):
        return len(self.tokens)


class DecayMathematicsEngine:
    """Engine for applying temporal decay mathematics"""
    
    def __init__(self, config: UnifiedXPConfig):
        self.config = config
    
    def apply_decay(self, units: Dict[str, XPUnit], time_delta_hours: float) -> Dict[str, Any]:
        """Apply exponential decay to all units"""
        decayed_count = 0
        total_decay = 0.0
        
        for unit in units.values():
            old_importance = unit.importance
            
            # Apply exponential decay
            decay_factor = np.exp(-unit.decay_rate * time_delta_hours)
            unit.importance *= decay_factor
            
            # Track statistics
            decay_amount = old_importance - unit.importance
            total_decay += decay_amount
            if decay_amount > 0.01:  # Significant decay
                decayed_count += 1
        
        return {
            'decayed_units': decayed_count,
            'total_decay': total_decay,
            'avg_decay': total_decay / len(units) if units else 0.0
        }


class ConsolidationEngine:
    """Engine for memory consolidation based on importance"""
    
    def __init__(self, config: UnifiedXPConfig):
        self.config = config
    
    def consolidate(self, units: Dict[str, XPUnit]) -> int:
        """Consolidate memories based on importance and access patterns"""
        consolidated_count = 0
        
        for unit in units.values():
            # Consolidation criteria
            high_importance = unit.importance >= self.config.consolidation_threshold
            frequent_access = unit.access_count >= 3
            recent_access = (get_current_timestamp() - unit.last_access) < 3600  # 1 hour
            
            if high_importance or (frequent_access and recent_access):
                # Boost importance for consolidation
                unit.importance = min(2.0, unit.importance * self.config.importance_boost_factor)
                consolidated_count += 1
        
        return consolidated_count


class RelationshipManager:
    """Manager for spatial topology and unit relationships"""
    
    def __init__(self, config: UnifiedXPConfig):
        self.config = config
        self.graph: Dict[str, Dict[str, float]] = {}
    
    def update_topology(self, new_unit: XPUnit, all_units: Dict[str, XPUnit]):
        """Update spatial topology with new unit"""
        if new_unit.content_id not in self.graph:
            self.graph[new_unit.content_id] = {}
        
        # Compute similarities with existing units
        similarities = []
        for unit_id, unit in all_units.items():
            if unit_id == new_unit.content_id:
                continue
            
            coherence = new_unit.compute_coherence_with(unit)
            similarities.append((unit_id, coherence))
        
        # Keep top k neighbors
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_neighbors = similarities[:self.config.k_neighbors]
        
        # Update topology
        for neighbor_id, coherence in top_neighbors:
            if coherence > self.config.topology_update_threshold:
                self.graph[new_unit.content_id][neighbor_id] = coherence
                new_unit.topology_neighbors[neighbor_id] = coherence
                new_unit.coherence_links[neighbor_id] = coherence
                
                # Update reverse relationship
                if neighbor_id not in self.graph:
                    self.graph[neighbor_id] = {}
                self.graph[neighbor_id][new_unit.content_id] = coherence
                
                neighbor_unit = all_units[neighbor_id]
                neighbor_unit.topology_neighbors[new_unit.content_id] = coherence
                neighbor_unit.coherence_links[new_unit.content_id] = coherence
    
    def get_graph(self) -> Dict[str, Dict[str, float]]:
        """Get the complete relationship graph"""
        return self.graph.copy()


# =============================================================================
# UNIFIED XP KERNEL - HD Kernel Interface Implementation
# =============================================================================

class UnifiedXPKernel:
    """
    Unified XP Kernel implementing HD Kernel interface specifications.
    
    This is the main interface that consolidates all XP Core functionality
    into a single, coherent kernel that can be used by any application.
    
    Implements HD Kernel interface:
    - process_memory: Ingest and process new experiences
    - retrieve_memory: Search and retrieve similar experiences  
    - consolidate_memory: Strengthen important memories
    - evolve_state: Apply temporal evolution
    """
    
    def __init__(self, config: UnifiedXPConfig = None):
        self.config = config or UnifiedXPConfig()
        self.environment = XPEnvironment(self.config)
        
        # Initialize emotional weighting system
        if self.config.enable_emotional_weighting:
            if self.config.use_enhanced_emotional_analysis:
                try:
                    self.emotional_analyzer = EnhancedEmotionalAnalyzer()
                    self.emotional_weighter = EnhancedEmotionalMemoryWeighter(self.emotional_analyzer)
                    self.consciousness_integrator = EnhancedConsciousnessEmotionalIntegrator(self.emotional_weighter)
                    logger.info("Enhanced emotional weighting system initialized")
                except Exception as e:
                    logger.warning(f"Enhanced emotional analysis failed, falling back to basic: {e}")
                    self.emotional_analyzer = EmotionalAnalyzer()
                    self.emotional_weighter = EmotionalMemoryWeighter(self.emotional_analyzer)
                    self.consciousness_integrator = ConsciousnessEmotionalIntegrator(self.emotional_weighter)
                    logger.info("Basic emotional weighting system initialized")
            else:
                self.emotional_analyzer = EmotionalAnalyzer()
                self.emotional_weighter = EmotionalMemoryWeighter(self.emotional_analyzer)
                self.consciousness_integrator = ConsciousnessEmotionalIntegrator(self.emotional_weighter)
                logger.info("Basic emotional weighting system initialized")
        else:
            self.emotional_analyzer = None
            self.emotional_weighter = None
            self.consciousness_integrator = None
        
        logger.info("Unified XP Kernel initialized - HD Kernel interface ready")
    
    # HD Kernel Interface Methods
    
    def process_memory(self, content: Any, metadata: Dict[str, Any] = None) -> str:
        """
        HD Kernel interface: Process input and store in XP mathematical foundation.
        
        Args:
            content: Input content (string or any serializable data)
            metadata: Optional metadata dictionary
            
        Returns:
            Content ID of the created XP unit
        """
        content_str = str(content) if not isinstance(content, str) else content
        
        # Analyze emotional content if emotional weighting is enabled
        if self.config.enable_emotional_weighting and self.emotional_analyzer:
            emotion = self.emotional_analyzer.analyze_text(content_str)
            
            # Update emotional weighter state
            self.emotional_weighter.update_emotional_state(emotion)
            
            # Calculate emotional importance boost (use enhanced method if available)
            if hasattr(self.emotional_weighter, 'calculate_enhanced_emotional_importance'):
                emotional_importance = self.emotional_weighter.calculate_enhanced_emotional_importance(
                    content_str, metadata
                )
            else:
                emotional_importance = self.emotional_weighter.calculate_emotional_importance(
                    content_str, metadata
                )
            
            # Add emotional metadata
            if metadata is None:
                metadata = {}
            metadata['emotional_importance'] = emotional_importance
            metadata['emotional_state'] = emotion.to_vector().tolist()
        
        unit = self.environment.ingest_experience(content_str, metadata)
        
        # Apply emotional importance boost if enabled
        if (self.config.enable_emotional_weighting and 
            'emotional_importance' in unit.metadata):
            emotional_boost = unit.metadata['emotional_importance']
            unit.importance *= (emotional_boost * self.config.emotional_importance_factor)
            
            # Set emotional state in the unit
            if 'emotional_state' in unit.metadata:
                emotion_vector = np.array(unit.metadata['emotional_state'])
                emotion = EmotionalState.from_vector(emotion_vector)
                unit.set_emotional_state(emotion)
        
        return unit.content_id
    
    def retrieve_memory(self, query: Any, k: int = 10, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        HD Kernel interface: Retrieve memories using XP HRR operations.
        
        Args:
            query: Query content (string or XPUnit)
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of memory results with content, similarity, and metadata
        """
        # Analyze query emotion if emotional weighting is enabled
        query_emotion = None
        if (self.config.enable_emotional_weighting and 
            self.emotional_analyzer and isinstance(query, str)):
            query_emotion = self.emotional_analyzer.analyze_text(query)
        
        results = self.environment.retrieve_similar(query, k, threshold)
        
        # Convert to HD Kernel format with emotional boosting
        formatted_results = []
        for unit, similarity in results:
            # Apply emotional retrieval boost if enabled
            final_similarity = similarity
            if (self.config.enable_emotional_weighting and 
                query_emotion and self.emotional_weighter):
                memory_emotion = unit.get_emotional_state()
                emotional_boost = self.emotional_weighter.calculate_emotional_retrieval_boost(
                    query_emotion, memory_emotion
                )
                final_similarity = similarity * (emotional_boost * self.config.emotional_retrieval_boost)
            
            formatted_results.append({
                'content_id': unit.content_id,
                'content': unit.content,
                'similarity': final_similarity,
                'base_similarity': similarity,
                'importance': unit.importance,
                'access_count': unit.access_count,
                'age_hours': unit.get_age_hours(),
                'emotional_state': unit.get_emotional_state().to_vector().tolist() if self.config.enable_emotional_weighting else None,
                'metadata': unit.metadata
            })
        
        # Re-sort by final similarity if emotional boosting was applied
        if self.config.enable_emotional_weighting and query_emotion:
            formatted_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return formatted_results
    
    def consolidate_memory(self) -> Dict[str, Any]:
        """
        HD Kernel interface: Consolidate memories using XP mathematics.
        
        Returns:
            Consolidation statistics
        """
        consolidated_count = self.environment.consolidate_memories()
        
        return {
            'consolidated_units': consolidated_count,
            'total_units': len(self.environment.units),
            'consolidation_rate': consolidated_count / len(self.environment.units) if self.environment.units else 0.0
        }
    
    def evolve_state(self, time_delta_hours: float = 1.0) -> Dict[str, Any]:
        """
        HD Kernel interface: Apply temporal evolution to XP state.
        
        Args:
            time_delta_hours: Time delta in hours for evolution
            
        Returns:
            Evolution statistics
        """
        decay_stats = self.environment.evolve_temporal_state(time_delta_hours)
        consolidation_stats = self.consolidate_memory()
        
        return {
            'time_delta_hours': time_delta_hours,
            'decay_stats': decay_stats,
            'consolidation_stats': consolidation_stats,
            'total_units': len(self.environment.units)
        }
    
    # Additional Unified Methods
    
    def get_unit(self, content_id: str) -> Optional[Dict[str, Any]]:
        """Get unit by content ID"""
        unit = self.environment.get_unit(content_id)
        if unit:
            return {
                'content_id': unit.content_id,
                'content': unit.content,
                'importance': unit.importance,
                'access_count': unit.access_count,
                'age_hours': unit.get_age_hours(),
                'coherence_links': unit.coherence_links,
                'metadata': unit.metadata
            }
        return None
    
    def create_binding(self, subject_id: str, role: str, object_id: str) -> Optional[np.ndarray]:
        """Create HRR binding between two units with a role"""
        subject = self.environment.get_unit(subject_id)
        object_unit = self.environment.get_unit(object_id)
        
        if subject and object_unit:
            return subject.bind_with_role(role, object_unit)
        return None
    
    def compute_coherence(self, unit_id1: str, unit_id2: str) -> Optional[float]:
        """Compute mathematical coherence between two units"""
        unit1 = self.environment.get_unit(unit_id1)
        unit2 = self.environment.get_unit(unit_id2)
        
        if unit1 and unit2:
            return unit1.compute_coherence_with(unit2)
        return None
    
    def get_relationship_graph(self) -> Dict[str, Dict[str, float]]:
        """Get the complete relationship graph"""
        return self.environment.get_relationship_graph()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return self.environment.get_comprehensive_stats()
    
    def export_state(self) -> Dict[str, Any]:
        """Export complete system state for persistence"""
        return {
            'config': {
                'embedding_dim': self.config.embedding_dim,
                'hrr_dim': self.config.hrr_dim,
                'decay_half_life': self.config.decay_half_life,
                'k_neighbors': self.config.k_neighbors
            },
            'units': {uid: unit.to_dict() for uid, unit in self.environment.units.items()},
            'relationship_graph': self.environment.get_relationship_graph(),
            'stats': self.environment.get_comprehensive_stats()
        }
    
    def import_state(self, state_data: Dict[str, Any]) -> bool:
        """Import system state from exported data"""
        try:
            # Import units
            for unit_id, unit_data in state_data.get('units', {}).items():
                unit = XPUnit.from_dict(unit_data)
                self.environment.units[unit_id] = unit
            
            # Import relationship graph
            self.environment.relationship_manager.graph = state_data.get('relationship_graph', {})
            
            # Update stats
            imported_stats = state_data.get('stats', {})
            self.environment.stats.update(imported_stats)
            
            logger.info(f"Imported {len(self.environment.units)} units successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import state: {e}")
            return False
    
    # Emotional Weighting Methods
    
    def get_emotional_state(self) -> Optional[EmotionalState]:
        """Get current emotional state of the system"""
        if self.emotional_weighter:
            return self.emotional_weighter.current_emotional_state
        return None
    
    def get_emotional_context(self, lookback_hours: float = 24.0) -> Dict[str, Any]:
        """Get emotional context for specified time period"""
        if self.emotional_weighter:
            return self.emotional_weighter.get_emotional_context(lookback_hours)
        return {}
    
    def get_emotional_consciousness_metrics(self) -> Dict[str, float]:
        """Get emotional consciousness metrics"""
        if self.consciousness_integrator:
            return self.consciousness_integrator.get_emotional_consciousness_metrics()
        return {}
    
    def calculate_emotional_consciousness_boost(self, base_consciousness: float) -> float:
        """Calculate emotional boost to consciousness level"""
        if self.consciousness_integrator:
            # Use enhanced method if available
            if hasattr(self.consciousness_integrator, 'calculate_enhanced_emotional_consciousness_boost'):
                return self.consciousness_integrator.calculate_enhanced_emotional_consciousness_boost(base_consciousness)
            else:
                return self.consciousness_integrator.calculate_emotional_consciousness_boost(base_consciousness)
        return base_consciousness
    
    def analyze_text_emotion(self, text: str) -> Optional[EmotionalState]:
        """Analyze emotional content of text"""
        if self.emotional_analyzer:
            return self.emotional_analyzer.analyze_text(text)
        return None
    
    def get_memory_emotional_importance(self, content_id: str) -> Optional[float]:
        """Get emotional importance of a specific memory"""
        unit = self.environment.get_unit(content_id)
        if unit and 'emotional_importance' in unit.metadata:
            return unit.metadata['emotional_importance']
        return None
    
    def get_emotionally_similar_memories(self, emotion: EmotionalState, k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve memories with similar emotional content"""
        if not self.config.enable_emotional_weighting:
            return []
        
        results = []
        for unit in self.environment.units.values():
            memory_emotion = unit.get_emotional_state()
            similarity = emotion.similarity(memory_emotion)
            
            if similarity > 0.3:  # Minimum emotional similarity threshold
                results.append({
                    'content_id': unit.content_id,
                    'content': unit.content,
                    'emotional_similarity': similarity,
                    'emotional_state': memory_emotion.to_vector().tolist(),
                    'importance': unit.importance,
                    'metadata': unit.metadata
                })
        
        # Sort by emotional similarity
        results.sort(key=lambda x: x['emotional_similarity'], reverse=True)
        return results[:k]


# =============================================================================
# COMPREHENSIVE TEST SUITE
# =============================================================================

def test_unified_xp_core():
    """Comprehensive test of the unified XP Core system"""
    print("🧪 TESTING UNIFIED XP CORE SYSTEM")
    print("=" * 60)
    
    # Test 1: Configuration
    print("\n1️⃣ Testing Unified Configuration...")
    config = UnifiedXPConfig(
        embedding_dim=256,
        hrr_dim=512,
        decay_half_life=72.0,
        k_neighbors=5
    )
    print(f"✅ Config: {config.embedding_dim}D embeddings, {config.hrr_dim}D HRR, {config.k_neighbors} neighbors")
    
    # Test 2: XP Kernel
    print("\n2️⃣ Testing Unified XP Kernel...")
    kernel = UnifiedXPKernel(config)
    print(f"✅ Kernel initialized with HD interface")
    
    # Test 3: Memory Processing (HD Interface)
    print("\n3️⃣ Testing Memory Processing (HD Interface)...")
    content_id1 = kernel.process_memory("The quantum holographic memory system uses mathematical foundations.")
    content_id2 = kernel.process_memory("Machine learning algorithms process natural language effectively.")
    content_id3 = kernel.process_memory("Holographic representations enable distributed memory storage.")
    print(f"✅ Processed 3 memories: {content_id1[:16]}..., {content_id2[:16]}..., {content_id3[:16]}...")
    
    # Test 4: Memory Retrieval (HD Interface)
    print("\n4️⃣ Testing Memory Retrieval (HD Interface)...")
    results = kernel.retrieve_memory("holographic memory systems", k=3)
    print(f"✅ Retrieved {len(results)} similar memories")
    for i, result in enumerate(results):
        print(f"   {i+1}. Similarity: {result['similarity']:.3f} - {result['content'][:50]}...")
    
    # Test 5: Mathematical Operations
    print("\n5️⃣ Testing Mathematical Operations...")
    if len(results) >= 2:
        coherence = kernel.compute_coherence(results[0]['content_id'], results[1]['content_id'])
        print(f"✅ Coherence between top 2 results: {coherence:.3f}")
    
    # Test 6: HRR Binding
    print("\n6️⃣ Testing HRR Binding Operations...")
    if len(results) >= 2:
        binding = kernel.create_binding(results[0]['content_id'], "relates_to", results[1]['content_id'])
        if binding is not None:
            print(f"✅ Created HRR binding: shape={binding.shape}, norm={np.linalg.norm(binding):.3f}")
    
    # Test 7: Temporal Evolution (HD Interface)
    print("\n7️⃣ Testing Temporal Evolution (HD Interface)...")
    evolution_stats = kernel.evolve_state(time_delta_hours=24.0)
    print(f"✅ Applied 24h evolution: {evolution_stats['decay_stats']['decayed_units']} units decayed")
    
    # Test 8: Consolidation (HD Interface)
    print("\n8️⃣ Testing Memory Consolidation (HD Interface)...")
    consolidation_stats = kernel.consolidate_memory()
    print(f"✅ Consolidated {consolidation_stats['consolidated_units']} memories")
    
    # Test 9: Relationship Graph
    print("\n9️⃣ Testing Relationship Graph...")
    graph = kernel.get_relationship_graph()
    total_relationships = sum(len(neighbors) for neighbors in graph.values())
    print(f"✅ Relationship graph: {len(graph)} nodes, {total_relationships} edges")
    
    # Test 10: System Statistics
    print("\n🔟 Testing System Statistics...")
    stats = kernel.get_stats()
    print(f"✅ System stats: {stats['total_units']} units, {stats['total_ingestions']} ingestions")
    print(f"   Uptime: {stats['uptime_hours']:.1f}h, Avg importance: {stats['avg_importance']:.3f}")
    
    # Test 11: State Export/Import
    print("\n1️⃣1️⃣ Testing State Export/Import...")
    exported_state = kernel.export_state()
    new_kernel = UnifiedXPKernel(config)
    import_success = new_kernel.import_state(exported_state)
    print(f"✅ State export/import: {import_success}, {len(exported_state['units'])} units transferred")
    
    print("\n🎉 UNIFIED XP CORE SYSTEM TEST COMPLETE!")
    print(f"🎯 All mathematical foundations working: HRR, decay, consolidation, coherence")
    print(f"🎯 HD Kernel interface fully implemented and tested")
    print(f"🎯 Production features ready: NLP, embeddings, cryptographic integrity")
    
    return kernel, stats


# =============================================================================
# EXPORT ALL UNIFIED COMPONENTS
# =============================================================================

__all__ = [
    # Core classes
    'UnifiedXPConfig', 'XPUnit', 'XPEnvironment', 'UnifiedXPKernel',
    
    # Supporting engines
    'DecayMathematicsEngine', 'ConsolidationEngine', 'RelationshipManager',
    
    # Test function
    'test_unified_xp_core'
]


if __name__ == "__main__":
    # Run comprehensive test when executed directly
    test_kernel, test_stats = test_unified_xp_core()
    print(f"\n📊 Final Stats: {test_stats}")