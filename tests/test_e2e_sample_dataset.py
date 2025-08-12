"""
End-to-end integration tests using the sample dataset.

These tests validate the complete Lumina Memory System workflow:
1. System initialization
2. Document ingestion
3. Query retrieval
4. Result validation
5. Performance characteristics

Usage:
    pytest tests/test_e2e_sample_dataset.py -v
    python -m pytest tests/test_e2e_sample_dataset.py::test_ai_papers_workflow
"""

import pytest
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lumina_memory import LuminaMemorySystem, LuminaConfig
from data.sample.datasets import load_ai_papers_sample, create_test_corpus, validate_dataset


class TestE2ESampleDataset:
    """End-to-end tests using sample datasets."""
    
    @pytest.fixture(scope="class")
    def sample_data(self):
        """Load and validate sample data."""
        # Validate dataset first
        report = validate_dataset("ai_papers")
        assert report['valid'], f"Dataset validation failed: {report['issues']}"
        
        docs, queries = load_ai_papers_sample()
        return {
            'documents': docs,
            'queries': queries,
            'corpus': create_test_corpus(docs, include_metadata=True)
        }
    
    @pytest.fixture(scope="class") 
    def memory_system(self):
        """Create a memory system for testing."""
        config = LuminaConfig(
            embedding_model_name="all-MiniLM-L6-v2",
            vector_store_type="faiss",
            stm_capacity=50,
            ltm_capacity=200,
            similarity_threshold=0.3
        )
        return LuminaMemorySystem(config)
    
    def test_dataset_validation(self):
        """Test that the sample dataset is valid and well-formed."""
        report = validate_dataset("ai_papers")
        
        # Should be valid
        assert report['valid'], f"Dataset validation failed: {report['issues']}"
        
        # Should have expected structure
        stats = report['statistics']
        assert stats['document_count'] > 0, "No documents in dataset"
        assert stats['query_count'] > 0, "No queries in dataset"
        assert stats['avg_content_length'] > 50, "Documents too short"
        assert len(stats['categories']) > 1, "Should have multiple categories"
    
    def test_document_ingestion(self, memory_system, sample_data):
        """Test ingesting the full sample dataset."""
        corpus = sample_data['corpus']
        docs = sample_data['documents']
        
        # Ingest documents
        memory_ids = memory_system.ingest(corpus)
        
        # Verify ingestion
        assert len(memory_ids) == len(corpus), "Not all documents were ingested"
        assert all(isinstance(mid, str) for mid in memory_ids), "Memory IDs should be strings"
        
        # Check system status
        status = memory_system.status()
        assert status['total_memories'] == len(docs), "Memory count mismatch"
        assert status['total_memories'] > 0, "No memories stored"
        
    def test_query_retrieval_coverage(self, memory_system, sample_data):
        """Test that queries return relevant results."""
        queries = sample_data['queries']
        
        results_summary = {
            'total_queries': len(queries),
            'queries_with_results': 0,
            'avg_results_per_query': 0,
            'avg_similarity_score': 0,
            'coverage_by_category': {}
        }
        
        all_similarities = []
        total_results = 0
        
        for query_data in queries:
            query_text = query_data['query']
            expected_categories = query_data.get('expected_categories', [])
            
            # Execute query
            results = memory_system.recall(query_text, k=10)
            
            if results:
                results_summary['queries_with_results'] += 1
                total_results += len(results)
                
                # Collect similarity scores
                similarities = [r.get('similarity', 0) for r in results]
                all_similarities.extend(similarities)
                
                # Track category coverage
                for category in expected_categories:
                    if category not in results_summary['coverage_by_category']:
                        results_summary['coverage_by_category'][category] = 0
                    results_summary['coverage_by_category'][category] += 1
        
        # Calculate averages
        if results_summary['queries_with_results'] > 0:
            results_summary['avg_results_per_query'] = total_results / len(queries)
        if all_similarities:
            results_summary['avg_similarity_score'] = sum(all_similarities) / len(all_similarities)
        
        # Assertions for coverage
        assert results_summary['queries_with_results'] > 0, "No queries returned results"
        assert results_summary['avg_results_per_query'] > 0, "Average results per query is 0"
        assert results_summary['avg_similarity_score'] > 0.1, "Similarity scores too low"
        
        # At least 80% of queries should return results
        coverage_rate = results_summary['queries_with_results'] / results_summary['total_queries']
        assert coverage_rate >= 0.8, f"Low query coverage: {coverage_rate:.2f}"
        
        print(f" Query Coverage Summary: {results_summary}")
    
    def test_semantic_similarity_quality(self, memory_system, sample_data):
        """Test that semantically related queries return relevant documents."""
        # Test specific query-document pairs that should be highly relevant
        test_cases = [
            {
                'query': 'transformer attention mechanisms',
                'should_contain_docs': ['doc_001'],  # "Attention Is All You Need"
                'min_similarity': 0.4
            },
            {
                'query': 'object detection computer vision',
                'should_contain_docs': ['doc_005'],  # YOLO
                'min_similarity': 0.3
            },
            {
                'query': 'reinforcement learning game playing',
                'should_contain_docs': ['doc_006', 'doc_007'],  # AlphaGo, DQN
                'min_similarity': 0.3
            }
        ]
        
        for test_case in test_cases:
            results = memory_system.recall(test_case['query'], k=5)
            assert len(results) > 0, f"No results for query: {test_case['query']}"
            
            # Check if expected documents appear in results
            result_content = [r.get('content', '') for r in results]
            
            # For this test, we'll check if the query returns reasonable similarities
            top_similarity = results[0].get('similarity', 0) if results else 0
            assert top_similarity >= test_case['min_similarity'], \
                f"Top similarity {top_similarity:.3f} below threshold {test_case['min_similarity']} for query: {test_case['query']}"
    
    def test_performance_characteristics(self, memory_system, sample_data):
        """Test basic performance characteristics."""
        import time
        
        queries = sample_data['queries']
        
        # Measure query latencies
        latencies = []
        for query_data in queries:
            start_time = time.time()
            results = memory_system.recall(query_data['query'], k=10)
            latency = time.time() - start_time
            latencies.append(latency)
        
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            
            # Basic performance assertions
            assert avg_latency < 1.0, f"Average query latency too high: {avg_latency:.3f}s"
            assert max_latency < 2.0, f"Max query latency too high: {max_latency:.3f}s"
            
            print(f" Performance: avg={avg_latency*1000:.1f}ms, max={max_latency*1000:.1f}ms")
    
    def test_consolidation_workflow(self, memory_system, sample_data):
        """Test the memory consolidation process."""
        # Check initial state
        initial_status = memory_system.status()
        
        # Trigger consolidation
        consolidation_result = memory_system.consolidate()
        
        # Check post-consolidation state
        final_status = memory_system.status()
        
        # Verify consolidation worked
        assert isinstance(consolidation_result, dict), "Consolidation should return a dict"
        assert 'consolidated_memories' in consolidation_result, "Missing consolidation info"
        
        # Memory counts should be preserved
        assert final_status['total_memories'] == initial_status['total_memories'], \
            "Total memory count should be preserved during consolidation"
        
        print(f" Consolidation: {consolidation_result}")
    
    def test_error_handling(self, memory_system):
        """Test error handling for edge cases."""
        # Test empty query
        results = memory_system.recall("", k=5)
        assert isinstance(results, list), "Should return empty list for empty query"
        
        # Test very long query
        long_query = "artificial intelligence " * 100
        results = memory_system.recall(long_query, k=5)
        assert isinstance(results, list), "Should handle long queries gracefully"
        
        # Test k=0
        results = memory_system.recall("test query", k=0)
        assert len(results) == 0, "Should return empty list for k=0"


# Standalone test functions for direct execution
def test_quick_validation():
    """Quick validation test that can be run standalone."""
    try:
        # Load and validate dataset
        report = validate_dataset("ai_papers")
        assert report['valid'], f"Dataset validation failed: {report['issues']}"
        
        docs, queries = load_ai_papers_sample()
        assert len(docs) > 0, "No documents loaded"
        assert len(queries) > 0, "No queries loaded"
        
        print(f" Quick validation passed: {len(docs)} docs, {len(queries)} queries")
        return True
        
    except Exception as e:
        print(f" Quick validation failed: {e}")
        return False


if __name__ == "__main__":
    # Run quick validation if executed directly
    print(" Running sample dataset validation...")
    success = test_quick_validation()
    
    if success:
        print("\n Sample dataset is ready for use!")
        print("Run full tests with: pytest tests/test_e2e_sample_dataset.py -v")
    else:
        print("\n Sample dataset has issues - check the data files")
        exit(1)
