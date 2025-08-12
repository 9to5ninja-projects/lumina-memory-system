"""Command-line interface for Lumina Memory System."""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.table import Table

from .config import LuminaConfig
from .embeddings import SentenceTransformerEmbedding, MockEmbeddingProvider
from .memory_system import MemorySystem
from .vector_store import FAISSVectorStore, InMemoryVectorStore
from .eval import MemoryEvaluator, create_synthetic_dataset
from .utils import setup_logging, validate_environment, format_memory_stats

console = Console()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Lumina Memory System CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Configuration management")
    config_parser.add_argument("--create", action="store_true", help="Create default config")
    config_parser.add_argument("--show", action="store_true", help="Show current config")
    config_parser.add_argument("--validate", action="store_true", help="Validate config")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run interactive demo")
    demo_parser.add_argument("--mock", action="store_true", help="Use mock embeddings")
    
    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    bench_parser.add_argument("--docs", type=int, default=1000, help="Number of documents")
    bench_parser.add_argument("--queries", type=int, default=100, help="Number of queries")
    bench_parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output file")
    
    # Environment command
    env_parser = subparsers.add_parser("env", help="Check environment")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    setup_logging(level="INFO", use_rich=True)
    
    if args.command == "config":
        handle_config_command(args)
    elif args.command == "demo":
        handle_demo_command(args)
    elif args.command == "benchmark":
        handle_benchmark_command(args)
    elif args.command == "env":
        handle_env_command(args)


def handle_config_command(args):
    """Handle configuration commands."""
    config_path = "lumina_config.json"
    
    if args.create:
        try:
            config = LuminaConfig.from_env()
            config.create_directories()
            config.setup_determinism()
            config.validate()
            config.save(config_path)
            console.print(f" Configuration created: {config_path}")
        except Exception as e:
            console.print(f" Failed to create config: {e}")
            sys.exit(1)
    
    elif args.show:
        try:
            if Path(config_path).exists():
                config = LuminaConfig.load(config_path)
            else:
                config = LuminaConfig.from_env()
            
            # Create table
            table = Table(title="Lumina Configuration")
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="magenta")
            
            for key, value in config.__dict__.items():
                table.add_row(key, str(value))
            
            console.print(table)
        except Exception as e:
            console.print(f" Failed to show config: {e}")
            sys.exit(1)
    
    elif args.validate:
        try:
            config = LuminaConfig.from_env()
            config.validate()
            console.print(" Configuration is valid")
        except Exception as e:
            console.print(f" Configuration validation failed: {e}")
            sys.exit(1)


def handle_demo_command(args):
    """Handle demo command."""
    try:
        # Load configuration
        config = LuminaConfig.from_env()
        config.create_directories()
        config.setup_determinism()
        
        # Create components
        if args.mock:
            embedding_provider = MockEmbeddingProvider(dimension=384)
            vector_store = InMemoryVectorStore()
            console.print(" Using mock components for demo")
        else:
            embedding_provider = SentenceTransformerEmbedding()
            vector_store = FAISSVectorStore(dimension=384)
            console.print(" Using real components for demo")
        
        # Create memory system
        memory = MemorySystem(embedding_provider, vector_store, config)
        
        # Interactive demo
        console.print("\\n Lumina Memory System Demo")
        console.print("Commands: ingest <text>, recall <query>, stats, quit")
        
        while True:
            try:
                command = console.input("\\n> ").strip()
                
                if command.lower() in ["quit", "exit", "q"]:
                    break
                
                elif command.startswith("ingest "):
                    text = command[7:].strip()
                    if text:
                        memory_id = memory.ingest(text)
                        console.print(f" Ingested: {memory_id[:8]}...")
                    else:
                        console.print(" Please provide text to ingest")
                
                elif command.startswith("recall "):
                    query = command[7:].strip()
                    if query:
                        results = memory.recall(query, k=3)
                        if results:
                            console.print(f" Found {len(results)} results:")
                            for i, result in enumerate(results, 1):
                                console.print(
                                    f"  {i}. {result['content'][:100]}... "
                                    f"(sim: {result['similarity']:.3f})"
                                )
                        else:
                            console.print(" No results found")
                    else:
                        console.print(" Please provide query text")
                
                elif command.lower() == "stats":
                    stats = memory.get_stats()
                    console.print("\\n" + format_memory_stats(stats))
                
                else:
                    console.print(" Unknown command")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f" Error: {e}")
        
        console.print(" Goodbye!")
        
    except Exception as e:
        console.print(f" Demo failed: {e}")
        sys.exit(1)


def handle_benchmark_command(args):
    """Handle benchmark command."""
    try:
        console.print(f" Running benchmark with {args.docs} docs, {args.queries} queries...")
        
        # Create synthetic dataset
        documents, queries, ground_truth = create_synthetic_dataset(
            num_documents=args.docs,
            num_queries=args.queries
        )
        
        # Setup system
        config = LuminaConfig.from_env()
        embedding_provider = MockEmbeddingProvider()  # Use mock for speed
        vector_store = InMemoryVectorStore()
        memory = MemorySystem(embedding_provider, vector_store, config)
        
        # Ingest documents
        console.print(" Ingesting documents...")
        for i, doc in enumerate(documents):
            memory.ingest(doc)
            if (i + 1) % 100 == 0:
                console.print(f"  Ingested {i + 1}/{len(documents)} documents")
        
        # Run evaluation
        console.print(" Running benchmarks...")
        evaluator = MemoryEvaluator(memory)
        results = evaluator.comprehensive_evaluation(queries, ground_truth)
        
        # Save results
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        
        # Display summary
        console.print(f"\\n Benchmark Results:")
        console.print(f"  Mean Latency: {results['latency']['mean_latency']:.3f}s")
        console.print(f"  P95 Latency: {results['latency']['p95_latency']:.3f}s")
        console.print(f"  Throughput: {results['throughput']['queries_per_second']:.1f} qps")
        console.print(f"  Recall@5: {results['recall']['recall@5']:.3f}")
        console.print(f"  Precision@5: {results['precision']['precision@5']:.3f}")
        console.print(f"\\n Full results saved to: {args.output}")
        
    except Exception as e:
        console.print(f" Benchmark failed: {e}")
        sys.exit(1)


def handle_env_command(args):
    """Handle environment check command."""
    results = validate_environment()
    
    # Create status table
    table = Table(title="Environment Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="yellow")
    
    table.add_row("Python", " OK", results.get("python_version", "Unknown"))
    
    if results["torch_available"]:
        table.add_row("PyTorch", " OK", results.get("torch_version", "Unknown"))
        if results["cuda_available"]:
            table.add_row("CUDA", " Available", "GPU acceleration enabled")
        else:
            table.add_row("CUDA", " Not Available", "Using CPU only")
    else:
        table.add_row("PyTorch", " Missing", "Install with: pip install torch")
    
    if results["faiss_available"]:
        table.add_row("FAISS", " OK", "Vector search available")
    else:
        table.add_row("FAISS", " Missing", "Install with: pip install faiss-cpu")
    
    if results["sentence_transformers_available"]:
        table.add_row("SentenceTransformers", " OK", "Embeddings available")
    else:
        table.add_row("SentenceTransformers", " Missing", "Install with: pip install sentence-transformers")
    
    console.print(table)
    
    # Show warnings and errors
    if results["warnings"]:
        console.print("\\n Warnings:")
        for warning in results["warnings"]:
            console.print(f"   {warning}")
    
    if results["errors"]:
        console.print("\\n Errors:")
        for error in results["errors"]:
            console.print(f"   {error}")
        sys.exit(1)
    else:
        console.print("\\n Environment looks good!")


if __name__ == "__main__":
    main()
