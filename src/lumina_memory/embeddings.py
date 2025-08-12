"""Embedding providers for Lumina Memory System."""

import logging
from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np

from .core import EmbeddingError

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        pass
    
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        pass
    
    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for single text."""
        return self.embed([text])[0]


class SentenceTransformerEmbedding(EmbeddingProvider):
    """Sentence Transformers embedding provider."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        """Initialize with model name and device."""
        self.model_name = model_name
        self.device = device
        self.model = None
        self._dimension = None
        
        self._load_model()
        logger.info(f"Initializing embedding provider: {model_name} on {device}")
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            # Get dimension by encoding a test string
            test_embedding = self.model.encode("test")
            self._dimension = len(test_embedding)
            
            logger.info(f"Loaded model with dimension: {self._dimension}")
            
        except ImportError as e:
            raise EmbeddingError(f"SentenceTransformers not available: {e}")
        except Exception as e:
            raise EmbeddingError(f"Failed to load model {self.model_name}: {e}")
    
    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return self._dimension
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        if not texts:
            return np.array([])
        
        try:
            # Validate inputs
            for i, text in enumerate(texts):
                if not isinstance(text, str):
                    raise EmbeddingError(f"Text at index {i} is not a string")
                if not text.strip():
                    logger.warning(f"Empty text at index {i}, using placeholder")
                    texts[i] = "[EMPTY]"
            
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            
            # Ensure 2D array
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            
            return embeddings
            
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embeddings: {e}")


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""
    
    def __init__(self, dimension: int = 384):
        """Initialize with specified dimension."""
        self.dimension = dimension
        np.random.seed(42)  # Deterministic for testing
    
    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension.""" 
        return self.dimension
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate mock embeddings based on text hash."""
        if not texts:
            return np.array([])
        
        embeddings = []
        for text in texts:
            # Create deterministic embedding based on text hash
            text_hash = hash(text)
            np.random.seed(abs(text_hash) % (2**32))
            embedding = np.random.normal(0, 1, self.dimension)
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            embeddings.append(embedding)
        
        return np.array(embeddings)
