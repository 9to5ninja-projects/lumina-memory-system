"""Test memory system functionality."""

import pytest

from lumina_memory.core import MemoryEntry, QueryType
from lumina_memory.memory_system import MemorySystem


def test_memory_system_initialization(memory_system):
    """Test memory system initialization."""
    assert memory_system.embedding_provider is not None
    assert memory_system.vector_store is not None
    assert memory_system.config is not None
    assert len(memory_system.stm) == 0
    assert len(memory_system.ltm) == 0


def test_ingest_single_memory(memory_system, sample_texts):
    """Test ingesting a single memory."""
    text = sample_texts[0]
    memory_id = memory_system.ingest(text)
    
    assert isinstance(memory_id, str)
    assert len(memory_id) > 0
    
    # Check memory was added
    assert len(memory_system.stm) == 1
    assert memory_system.stats["total_memories"] == 1
    assert memory_system.stats["total_ingestions"] == 1


def test_ingest_multiple_memories(memory_system, sample_texts):
    """Test ingesting multiple memories."""
    memory_ids = []
    for text in sample_texts:
        memory_id = memory_system.ingest(text)
        memory_ids.append(memory_id)
    
    assert len(memory_ids) == len(sample_texts)
    assert len(set(memory_ids)) == len(sample_texts)  # All unique
    assert len(memory_system.stm) == len(sample_texts)


def test_recall_memories(memory_system, sample_texts):
    """Test recalling memories."""
    # Ingest memories
    for text in sample_texts:
        memory_system.ingest(text)
    
    # Test recall
    results = memory_system.recall("artificial intelligence", k=3)
    
    assert isinstance(results, list)
    assert len(results) <= 3
    
    if results:
        # Check result structure
        result = results[0]
        assert "id" in result
        assert "content" in result
        assert "similarity" in result
        assert "metadata" in result
        assert "timestamp" in result
        
        # Check similarity is in valid range
        assert 0.0 <= result["similarity"] <= 1.0


def test_recall_with_filters(memory_system):
    """Test recall with metadata filters."""
    # Ingest memories with metadata
    memory_system.ingest("AI research paper", {"category": "research", "topic": "AI"})
    memory_system.ingest("ML blog post", {"category": "blog", "topic": "ML"})
    memory_system.ingest("AI news article", {"category": "news", "topic": "AI"})
    
    # Recall with filter
    results = memory_system.recall(
        "artificial intelligence",
        k=5,
        filters={"category": "research"}
    )
    
    # Should only return research papers
    assert len(results) <= 1
    if results:
        assert results[0]["metadata"]["category"] == "research"


def test_consolidate_memories(memory_system, sample_texts):
    """Test memory consolidation."""
    # Ingest memories
    for i, text in enumerate(sample_texts):
        memory_id = memory_system.ingest(text)
        # Set high importance for some memories
        if i % 2 == 0:
            entry = memory_system._find_entry(memory_id)
            entry.importance_score = 0.8
    
    # Consolidate
    consolidated_count = memory_system.consolidate()
    
    assert consolidated_count >= 0
    assert len(memory_system.ltm) == consolidated_count


def test_forget_memories(memory_system, sample_texts):
    """Test forgetting memories."""
    # Ingest memories
    memory_ids = []
    for text in sample_texts:
        memory_id = memory_system.ingest(text)
        memory_ids.append(memory_id)
    
    initial_count = memory_system.stats["total_memories"]
    
    # Forget first two memories
    forgotten_count = memory_system.forget(memory_ids[:2])
    
    assert forgotten_count >= 0
    assert memory_system.stats["total_memories"] == initial_count - forgotten_count


def test_system_stats(memory_system, sample_texts):
    """Test getting system statistics."""
    # Initially empty
    stats = memory_system.get_stats()
    assert stats["total_memories"] == 0
    assert stats["total_queries"] == 0
    assert stats["stm_size"] == 0
    assert stats["ltm_size"] == 0
    
    # After ingestion
    for text in sample_texts:
        memory_system.ingest(text)
    
    stats = memory_system.get_stats()
    assert stats["total_memories"] == len(sample_texts)
    assert stats["total_ingestions"] == len(sample_texts)
    assert stats["stm_size"] == len(sample_texts)
    
    # After querying
    memory_system.recall("test query", k=1)
    stats = memory_system.get_stats()
    assert stats["total_queries"] == 1


def test_empty_recall(memory_system):
    """Test recall from empty memory system."""
    results = memory_system.recall("test query")
    assert results == []


def test_invalid_ingest(memory_system):
    """Test ingesting invalid content."""
    with pytest.raises(Exception):
        memory_system.ingest("")  # Empty content should fail


@pytest.mark.benchmark
def test_performance_benchmark(memory_system, sample_texts):
    """Test basic performance characteristics."""
    # Ingest many memories
    for i in range(100):
        memory_system.ingest(f"Test memory {i}: {sample_texts[i % len(sample_texts)]}")
    
    # Measure recall performance
    import time
    start_time = time.time()
    
    for _ in range(10):
        memory_system.recall("test query", k=5)
    
    elapsed_time = time.time() - start_time
    avg_query_time = elapsed_time / 10
    
    # Should be reasonably fast (< 100ms per query for mock system)
    assert avg_query_time < 0.1, f"Query time too slow: {avg_query_time:.3f}s"
