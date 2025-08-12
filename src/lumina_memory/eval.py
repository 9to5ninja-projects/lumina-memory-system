"""Evaluation and benchmarking for Lumina Memory System."""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .core import MemoryEntry
from .memory_system import MemorySystem

logger = logging.getLogger(__name__)


class MemoryEvaluator:
    """Evaluate memory system performance."""
    
    def __init__(self, memory_system: MemorySystem):
        """Initialize evaluator."""
        self.memory_system = memory_system
    
    def recall_at_k(
        self,
        queries: List[str],
        ground_truth: List[List[str]],
        k_values: List[int] = [1, 3, 5, 10],
    ) -> Dict[str, float]:
        """
        Calculate Recall@K metrics.
        
        Args:
            queries: List of query strings
            ground_truth: List of relevant document IDs for each query
            k_values: K values to evaluate
            
        Returns:
            Recall@K scores for each K
        """
        if len(queries) != len(ground_truth):
            raise ValueError("Queries and ground truth must have same length")
        
        recall_scores = {f"recall@{k}": 0.0 for k in k_values}
        
        for query, relevant_ids in zip(queries, ground_truth):
            if not relevant_ids:
                continue
            
            # Get results from memory system
            results = self.memory_system.recall(query, k=max(k_values))
            retrieved_ids = [r["id"] for r in results]
            
            # Calculate recall for each K
            for k in k_values:
                top_k_ids = retrieved_ids[:k]
                relevant_retrieved = len(set(top_k_ids) & set(relevant_ids))
                recall = relevant_retrieved / len(relevant_ids)
                recall_scores[f"recall@{k}"] += recall
        
        # Average across queries
        num_queries = len(queries)
        for k in k_values:
            recall_scores[f"recall@{k}"] /= num_queries
        
        return recall_scores
    
    def precision_at_k(
        self,
        queries: List[str],
        ground_truth: List[List[str]],
        k_values: List[int] = [1, 3, 5, 10],
    ) -> Dict[str, float]:
        """Calculate Precision@K metrics."""
        if len(queries) != len(ground_truth):
            raise ValueError("Queries and ground truth must have same length")
        
        precision_scores = {f"precision@{k}": 0.0 for k in k_values}
        
        for query, relevant_ids in zip(queries, ground_truth):
            if not relevant_ids:
                continue
            
            results = self.memory_system.recall(query, k=max(k_values))
            retrieved_ids = [r["id"] for r in results]
            
            for k in k_values:
                if k > len(retrieved_ids):
                    continue
                
                top_k_ids = retrieved_ids[:k]
                relevant_retrieved = len(set(top_k_ids) & set(relevant_ids))
                precision = relevant_retrieved / k if k > 0 else 0.0
                precision_scores[f"precision@{k}"] += precision
        
        # Average across queries
        num_queries = len(queries)
        for k in k_values:
            precision_scores[f"precision@{k}"] /= num_queries
        
        return precision_scores
    
    def ndcg_at_k(
        self,
        queries: List[str],
        ground_truth: List[List[str]],
        k_values: List[int] = [1, 3, 5, 10],
    ) -> Dict[str, float]:
        """Calculate Normalized Discounted Cumulative Gain (NDCG@K)."""
        if len(queries) != len(ground_truth):
            raise ValueError("Queries and ground truth must have same length")
        
        ndcg_scores = {f"ndcg@{k}": 0.0 for k in k_values}
        
        for query, relevant_ids in zip(queries, ground_truth):
            if not relevant_ids:
                continue
            
            results = self.memory_system.recall(query, k=max(k_values))
            
            for k in k_values:
                if k == 0:
                    continue
                
                # Calculate DCG
                dcg = 0.0
                for i, result in enumerate(results[:k]):
                    if result["id"] in relevant_ids:
                        dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
                
                # Calculate IDCG (perfect ranking)
                idcg = 0.0
                for i in range(min(k, len(relevant_ids))):
                    idcg += 1.0 / np.log2(i + 2)
                
                # Calculate NDCG
                ndcg = dcg / idcg if idcg > 0 else 0.0
                ndcg_scores[f"ndcg@{k}"] += ndcg
        
        # Average across queries
        num_queries = len(queries)
        for k in k_values:
            ndcg_scores[f"ndcg@{k}"] /= num_queries
        
        return ndcg_scores
    
    def latency_benchmark(
        self,
        queries: List[str],
        k: int = 5,
        num_runs: int = 10,
    ) -> Dict[str, float]:
        """Benchmark query latency."""
        latencies = []
        
        for query in queries:
            query_latencies = []
            
            for _ in range(num_runs):
                start_time = time.time()
                self.memory_system.recall(query, k=k)
                latency = time.time() - start_time
                query_latencies.append(latency)
            
            latencies.extend(query_latencies)
        
        latencies = np.array(latencies)
        
        return {
            "mean_latency": float(np.mean(latencies)),
            "median_latency": float(np.median(latencies)),
            "p95_latency": float(np.percentile(latencies, 95)),
            "p99_latency": float(np.percentile(latencies, 99)),
            "min_latency": float(np.min(latencies)),
            "max_latency": float(np.max(latencies)),
            "std_latency": float(np.std(latencies)),
        }
    
    def throughput_benchmark(
        self,
        queries: List[str],
        k: int = 5,
        duration_seconds: int = 60,
    ) -> Dict[str, float]:
        """Benchmark query throughput."""
        start_time = time.time()
        query_count = 0
        query_idx = 0
        
        while time.time() - start_time < duration_seconds:
            query = queries[query_idx % len(queries)]
            self.memory_system.recall(query, k=k)
            query_count += 1
            query_idx += 1
        
        elapsed_time = time.time() - start_time
        throughput = query_count / elapsed_time
        
        return {
            "queries_per_second": throughput,
            "total_queries": query_count,
            "elapsed_time": elapsed_time,
        }
    
    def comprehensive_evaluation(
        self,
        queries: List[str],
        ground_truth: Optional[List[List[str]]] = None,
        k_values: List[int] = [1, 3, 5, 10],
        latency_runs: int = 10,
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation."""
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_stats": self.memory_system.get_stats(),
        }
        
        # Latency benchmark
        logger.info("Running latency benchmark...")
        results["latency"] = self.latency_benchmark(queries[:10], num_runs=latency_runs)
        
        # Throughput benchmark
        logger.info("Running throughput benchmark...")
        results["throughput"] = self.throughput_benchmark(queries[:10], duration_seconds=10)
        
        # Accuracy metrics (if ground truth provided)
        if ground_truth:
            logger.info("Running accuracy evaluation...")
            results["recall"] = self.recall_at_k(queries, ground_truth, k_values)
            results["precision"] = self.precision_at_k(queries, ground_truth, k_values)
            results["ndcg"] = self.ndcg_at_k(queries, ground_truth, k_values)
        
        return results
    
    def save_results(self, results: Dict[str, Any], filepath: str = "eval_results.json"):
        """Save evaluation results to file."""
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {filepath}")


def create_synthetic_dataset(
    num_documents: int = 1000,
    num_queries: int = 100,
    doc_length_range: Tuple[int, int] = (20, 100),
) -> Tuple[List[str], List[str], List[List[str]]]:
    """
    Create synthetic dataset for evaluation.
    
    Args:
        num_documents: Number of documents to generate
        num_queries: Number of queries to generate
        doc_length_range: Range of document lengths (in words)
        
    Returns:
        Tuple of (documents, queries, ground_truth)
    """
    import random
    
    # Sample topics and words
    topics = [
        "artificial intelligence", "machine learning", "deep learning",
        "neural networks", "computer vision", "natural language processing",
        "robotics", "data science", "statistics", "programming",
        "python", "javascript", "web development", "mobile apps",
        "cloud computing", "cybersecurity", "blockchain", "quantum computing",
    ]
    
    words = [
        "algorithm", "model", "training", "prediction", "classification",
        "regression", "clustering", "optimization", "accuracy", "performance",
        "dataset", "feature", "vector", "matrix", "tensor", "gradient",
        "learning", "network", "layer", "neuron", "activation", "loss",
        "evaluation", "validation", "testing", "deployment", "production",
    ]
    
    # Generate documents
    documents = []
    for _ in range(num_documents):
        topic = random.choice(topics)
        length = random.randint(*doc_length_range)
        
        doc_words = [topic] + random.choices(words, k=length - 1)
        document = " ".join(doc_words)
        documents.append(document)
    
    # Generate queries and ground truth
    queries = []
    ground_truth = []
    
    for _ in range(num_queries):
        # Create query based on a topic
        topic = random.choice(topics)
        query_words = [topic] + random.choices(words, k=random.randint(2, 5))
        query = " ".join(query_words)
        queries.append(query)
        
        # Find relevant documents (simple keyword matching)
        relevant_docs = []
        for i, doc in enumerate(documents):
            # Simple relevance: share at least 2 words with query
            query_set = set(query.lower().split())
            doc_set = set(doc.lower().split())
            if len(query_set & doc_set) >= 2:
                relevant_docs.append(str(i))
        
        ground_truth.append(relevant_docs)
    
    return documents, queries, ground_truth
