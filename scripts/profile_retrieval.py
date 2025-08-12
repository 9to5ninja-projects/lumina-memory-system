#!/usr/bin/env python3
"""
Performance profiling script for Lumina Memory System retrieval throughput.

This script measures:
- Ingestion rate (documents/second)
- Query throughput (queries/second) 
- Memory usage patterns
- Latency percentiles

Usage:
    python scripts/profile_retrieval.py --docs 1000 --queries 100
    python scripts/profile_retrieval.py --embedding-model all-MiniLM-L6-v2 --batch-size 32
"""

import argparse
import time
import statistics
import psutil
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Import Lumina components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lumina_memory import LuminaMemorySystem, LuminaConfig

class PerformanceProfiler:
    """Profiles retrieval performance of Lumina Memory System."""
    
    def __init__(self, config: LuminaConfig):
        self.config = config
        self.system = None
        self.metrics = {
            "ingestion": [],
            "retrieval": [],
            "memory_usage": [],
            "system_info": {},
        }
        
    def setup_system(self):
        """Initialize the memory system."""
        print(" Initializing Lumina Memory System...")
        start_time = time.time()
        
        self.system = LuminaMemorySystem(self.config)
        
        setup_time = time.time() - start_time
        print(f" System initialized in {setup_time:.2f}s")
        
        # Record system info
        self.metrics["system_info"] = {
            "setup_time": setup_time,
            "python_version": sys.version,
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "embedding_model": self.config.embedding_model_name,
            "vector_store": self.config.vector_store_type,
        }
    
    def generate_test_documents(self, num_docs: int) -> List[str]:
        """Generate test documents of varying lengths."""
        print(f" Generating {num_docs} test documents...")
        
        # Templates for different document types
        templates = [
            "This is a technical document about {topic} and its applications in {field}. "
            "The methodology involves {method} which has proven effective for {purpose}. "
            "Key findings include {finding1}, {finding2}, and {finding3}. "
            "Future research directions should focus on {future}.",
            
            "A comprehensive analysis of {topic} reveals significant insights into {field}. "
            "The study examined {method} across multiple scenarios, demonstrating {purpose}. "
            "Results show {finding1} with statistical significance. Additional observations "
            "include {finding2} and {finding3}. These findings suggest {future}.",
            
            "Research on {topic} has evolved significantly in {field}. Modern approaches "
            "using {method} have enabled new capabilities for {purpose}. Critical discoveries "
            "include {finding1}, leading to better understanding of {finding2}. "
            "The implications for {finding3} are substantial, pointing toward {future}.",
        ]
        
        # Topic pools for variety
        topics = ["artificial intelligence", "machine learning", "neural networks", "data science", 
                 "computer vision", "natural language processing", "robotics", "automation",
                 "quantum computing", "distributed systems", "cloud computing", "cybersecurity"]
        
        fields = ["healthcare", "finance", "education", "manufacturing", "transportation",
                 "telecommunications", "energy", "agriculture", "retail", "government"]
        
        methods = ["deep learning", "statistical analysis", "optimization algorithms", 
                  "ensemble methods", "reinforcement learning", "transfer learning"]
        
        purposes = ["pattern recognition", "prediction accuracy", "cost reduction",
                   "efficiency improvement", "risk mitigation", "quality enhancement"]
        
        findings = ["improved accuracy by 15%", "reduced processing time by 40%", 
                   "enhanced reliability", "better scalability", "lower resource usage",
                   "increased user satisfaction", "stronger security measures"]
        
        futures = ["real-time applications", "edge deployment", "multi-modal integration",
                  "federated learning", "explainable AI", "automated optimization"]
        
        documents = []
        for i in range(num_docs):
            template = templates[i % len(templates)]
            doc = template.format(
                topic=topics[i % len(topics)],
                field=fields[i % len(fields)], 
                method=methods[i % len(methods)],
                purpose=purposes[i % len(purposes)],
                finding1=findings[i % len(findings)],
                finding2=findings[(i + 1) % len(findings)],
                finding3=findings[(i + 2) % len(findings)],
                future=futures[i % len(futures)]
            )
            # Add document ID for tracking
            documents.append(f"[DOC-{i:04d}] {doc}")
        
        return documents
    
    def profile_ingestion(self, documents: List[str], batch_size: int = 10) -> Dict[str, Any]:
        """Profile document ingestion performance."""
        print(f" Profiling ingestion of {len(documents)} documents (batch_size={batch_size})...")
        
        batch_times = []
        memory_usage = []
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Measure memory before batch
            mem_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Time the batch ingestion
            start_time = time.time()
            self.system.ingest(batch)
            batch_time = time.time() - start_time
            
            # Measure memory after batch
            mem_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            batch_times.append(batch_time)
            memory_usage.append(mem_after - mem_before)
            
            if i % (batch_size * 10) == 0:
                print(f"  Processed {i + len(batch)}/{len(documents)} docs "
                      f"({batch_time:.2f}s, {len(batch)/batch_time:.1f} docs/s)")
        
        # Calculate metrics
        total_time = sum(batch_times)
        total_docs = len(documents)
        throughput = total_docs / total_time
        
        metrics = {
            "total_documents": total_docs,
            "total_time": total_time,
            "throughput_docs_per_sec": throughput,
            "avg_batch_time": statistics.mean(batch_times),
            "batch_time_std": statistics.stdev(batch_times) if len(batch_times) > 1 else 0,
            "memory_delta_mb": sum(memory_usage),
            "avg_memory_per_doc_kb": (sum(memory_usage) * 1024) / total_docs,
        }
        
        self.metrics["ingestion"] = metrics
        print(f" Ingestion complete: {throughput:.1f} docs/s")
        return metrics
    
    def generate_test_queries(self, num_queries: int) -> List[str]:
        """Generate test queries that should match ingested documents."""
        print(f" Generating {num_queries} test queries...")
        
        query_templates = [
            "What is {topic}?",
            "How does {method} work?",
            "Applications of {topic} in {field}",
            "Benefits of {method} for {purpose}",
            "Latest research on {topic}",
            "Challenges in {field} using {topic}",
            "Future of {method} in {field}",
            "Comparison of {method} approaches",
            "Best practices for {topic}",
            "Case studies on {method}",
        ]
        
        topics = ["artificial intelligence", "machine learning", "neural networks", 
                 "computer vision", "natural language processing", "deep learning"]
        methods = ["optimization", "statistical analysis", "ensemble methods", 
                  "reinforcement learning", "transfer learning"]
        fields = ["healthcare", "finance", "education", "manufacturing"]
        purposes = ["accuracy", "efficiency", "scalability", "reliability"]
        
        queries = []
        for i in range(num_queries):
            template = query_templates[i % len(query_templates)]
            query = template.format(
                topic=topics[i % len(topics)],
                method=methods[i % len(methods)],
                field=fields[i % len(fields)],
                purpose=purposes[i % len(purposes)]
            )
            queries.append(query)
        
        return queries
    
    def profile_retrieval(self, queries: List[str], k: int = 10) -> Dict[str, Any]:
        """Profile query retrieval performance."""
        print(f" Profiling retrieval for {len(queries)} queries (k={k})...")
        
        query_times = []
        result_counts = []
        similarity_scores = []
        
        for i, query in enumerate(queries):
            start_time = time.time()
            results = self.system.recall(query, k=k)
            query_time = time.time() - start_time
            
            query_times.append(query_time)
            result_counts.append(len(results))
            
            if results:
                similarity_scores.extend([r.get('similarity', 0) for r in results])
            
            if i % 10 == 0:
                print(f"  Query {i + 1}/{len(queries)} "
                      f"({query_time*1000:.1f}ms, {len(results)} results)")
        
        # Calculate percentiles
        percentiles = [50, 90, 95, 99]
        latency_percentiles = {
            f"p{p}": np.percentile(query_times, p) * 1000  # Convert to ms
            for p in percentiles
        }
        
        metrics = {
            "total_queries": len(queries),
            "total_time": sum(query_times),
            "throughput_queries_per_sec": len(queries) / sum(query_times),
            "avg_latency_ms": statistics.mean(query_times) * 1000,
            "latency_std_ms": statistics.stdev(query_times) * 1000,
            "latency_percentiles_ms": latency_percentiles,
            "avg_results_per_query": statistics.mean(result_counts),
            "avg_similarity_score": statistics.mean(similarity_scores) if similarity_scores else 0,
        }
        
        self.metrics["retrieval"] = metrics
        print(f" Retrieval complete: {metrics['throughput_queries_per_sec']:.1f} queries/s")
        return metrics
    
    def save_results(self, output_path: Path):
        """Save profiling results to JSON file."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "embedding_model": self.config.embedding_model_name,
                "vector_store": self.config.vector_store_type,
                "embedding_dim": getattr(self.config, 'embedding_dim', None),
            },
            "metrics": self.metrics
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f" Results saved to {output_path}")
    
    def print_summary(self):
        """Print performance summary."""
        print("\n" + "="*60)
        print(" PERFORMANCE SUMMARY")
        print("="*60)
        
        if "ingestion" in self.metrics:
            ing = self.metrics["ingestion"]
            print(f" INGESTION:")
            print(f"   Documents: {ing['total_documents']:,}")
            print(f"   Throughput: {ing['throughput_docs_per_sec']:.1f} docs/s")
            print(f"   Total time: {ing['total_time']:.2f}s")
            print(f"   Memory delta: {ing['memory_delta_mb']:.1f} MB")
        
        if "retrieval" in self.metrics:
            ret = self.metrics["retrieval"]
            print(f"\n RETRIEVAL:")
            print(f"   Queries: {ret['total_queries']:,}")
            print(f"   Throughput: {ret['throughput_queries_per_sec']:.1f} queries/s")
            print(f"   Avg latency: {ret['avg_latency_ms']:.1f}ms")
            print(f"   P95 latency: {ret['latency_percentiles_ms']['p95']:.1f}ms")
            print(f"   Avg results: {ret['avg_results_per_query']:.1f}")
        
        print(f"\n  SYSTEM:")
        sys_info = self.metrics["system_info"]
        print(f"   CPU cores: {sys_info['cpu_count']}")
        print(f"   Memory: {sys_info['memory_total'] / 1024**3:.1f} GB")
        print(f"   Model: {sys_info['embedding_model']}")
        print(f"   Vector store: {sys_info['vector_store']}")


def main():
    parser = argparse.ArgumentParser(description="Profile Lumina Memory System performance")
    parser.add_argument("--docs", type=int, default=1000, 
                       help="Number of documents to ingest")
    parser.add_argument("--queries", type=int, default=100,
                       help="Number of queries to profile") 
    parser.add_argument("--batch-size", type=int, default=10,
                       help="Batch size for ingestion")
    parser.add_argument("--k", type=int, default=10,
                       help="Number of results to retrieve per query")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2",
                       help="Embedding model to use")
    parser.add_argument("--vector-store", type=str, default="faiss",
                       choices=["faiss", "chromadb"],
                       help="Vector store backend")
    parser.add_argument("--output", type=str, 
                       default=f"profile_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                       help="Output JSON file path")
    
    args = parser.parse_args()
    
    # Create configuration
    config = LuminaConfig(
        embedding_model_name=args.embedding_model,
        vector_store_type=args.vector_store,
        stm_capacity=args.docs // 4,  # 25% in STM
        ltm_capacity=args.docs,
    )
    
    # Initialize profiler
    profiler = PerformanceProfiler(config)
    profiler.setup_system()
    
    try:
        # Generate test data
        documents = profiler.generate_test_documents(args.docs)
        queries = profiler.generate_test_queries(args.queries)
        
        # Profile ingestion
        profiler.profile_ingestion(documents, args.batch_size)
        
        # Profile retrieval
        profiler.profile_retrieval(queries, args.k)
        
        # Print summary and save results
        profiler.print_summary()
        profiler.save_results(Path(args.output))
        
        print(f"\n Profiling complete! Results saved to {args.output}")
        
    except KeyboardInterrupt:
        print("\n  Profiling interrupted by user")
    except Exception as e:
        print(f"\n Profiling failed: {e}")
        raise


if __name__ == "__main__":
    main()
