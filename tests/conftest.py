"""Test configuration."""

import pytest
from lumina_memory.config import LuminaConfig
from lumina_memory.embeddings import MockEmbeddingProvider
from lumina_memory.vector_store import InMemoryVectorStore
from lumina_memory.memory_system import MemorySystem


@pytest.fixture
def config():
    """Create test configuration."""
    config = LuminaConfig(
        embedding_dim=128,  # Smaller for tests
        stm_capacity=10,
        ltm_capacity=100,
        deterministic_mode=True,
        random_seed=42,
    )
    return config


@pytest.fixture
def embedding_provider():
    """Create test embedding provider."""
    return MockEmbeddingProvider(dimension=128)


@pytest.fixture
def vector_store():
    """Create test vector store."""
    return InMemoryVectorStore()


@pytest.fixture
def memory_system(config, embedding_provider, vector_store):
    """Create test memory system."""
    return MemorySystem(embedding_provider, vector_store, config)


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "Artificial intelligence is transforming technology.",
        "Machine learning algorithms learn from data patterns.",
        "Deep learning uses neural networks with many layers.",
        "Natural language processing helps computers understand text.",
        "Computer vision enables machines to interpret images.",
    ]
