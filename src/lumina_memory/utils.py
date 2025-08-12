"""Utility functions for Lumina Memory System."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.logging import RichHandler

from .config import LuminaConfig


def setup_logging(
    level: str = "INFO",
    log_to_file: bool = True,
    log_dir: str = "./logs",
    use_rich: bool = True,
) -> None:
    """Setup structured logging for the application."""
    
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler with rich formatting
    if use_rich:
        console = Console()
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        console_handler.setLevel(getattr(logging, level.upper()))
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, level.upper()))
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        log_file = Path(log_dir) / "lumina_memory.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Set library loggers to WARNING to reduce noise
    for lib_name in ["transformers", "sentence_transformers", "torch", "numpy"]:
        logging.getLogger(lib_name).setLevel(logging.WARNING)


def normalize_similarity(score: float, metric: str = "cosine") -> float:
    """
    Normalize similarity scores to 0-1 range.
    
    Args:
        score: Raw similarity score
        metric: Similarity metric used
        
    Returns:
        Normalized similarity score between 0 and 1
    """
    if metric == "cosine":
        # FAISS cosine returns dot product, convert to similarity
        return max(0.0, min(1.0, (score + 1.0) / 2.0))
    elif metric == "euclidean":
        # Distance -> similarity
        return max(0.0, min(1.0, 1.0 / (1.0 + score)))
    else:
        # Inner product - normalize to 0-1 range
        return max(0.0, min(1.0, score))


def validate_environment() -> Dict[str, Any]:
    """
    Validate the runtime environment.
    
    Returns:
        Environment validation results
    """
    results = {
        "python_version": None,
        "torch_available": False,
        "torch_version": None,
        "cuda_available": False,
        "faiss_available": False,
        "sentence_transformers_available": False,
        "warnings": [],
        "errors": [],
    }
    
    try:
        import sys
        results["python_version"] = sys.version
        
        # Check Python version
        if sys.version_info < (3, 10):
            results["errors"].append("Python 3.10+ required")
    except Exception as e:
        results["errors"].append(f"Python version check failed: {e}")
    
    try:
        import torch
        results["torch_available"] = True
        results["torch_version"] = torch.__version__
        results["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        results["warnings"].append("PyTorch not available")
    
    try:
        import faiss
        results["faiss_available"] = True
    except ImportError:
        results["warnings"].append("FAISS not available")
    
    try:
        import sentence_transformers
        results["sentence_transformers_available"] = True
    except ImportError:
        results["warnings"].append("SentenceTransformers not available")
    
    return results


def create_default_config(
    config_path: str = "lumina_config.json",
    force: bool = False,
) -> LuminaConfig:
    """
    Create a default configuration file.
    
    Args:
        config_path: Path to save configuration
        force: Overwrite existing configuration
        
    Returns:
        Created configuration
    """
    config_file = Path(config_path)
    
    if config_file.exists() and not force:
        raise FileExistsError(f"Configuration file exists: {config_path}")
    
    # Create config from environment
    config = LuminaConfig.from_env()
    
    # Create directories
    config.create_directories()
    
    # Setup determinism
    config.setup_determinism()
    
    # Validate
    config.validate()
    
    # Save
    config.save(config_path)
    
    return config


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics."""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            "percent": process.memory_percent(),
        }
    except ImportError:
        return {"error": "psutil not available"}
    except Exception as e:
        return {"error": str(e)}


def format_memory_stats(stats: Dict[str, Any]) -> str:
    """Format memory system statistics for display."""
    lines = [
        "=== Lumina Memory System Statistics ===",
        f"Total Memories: {stats.get('total_memories', 0)}",
        f"Total Queries: {stats.get('total_queries', 0)}",
        f"Total Ingestions: {stats.get('total_ingestions', 0)}",
        f"Average Query Time: {stats.get('avg_query_time', 0):.3f}s",
        f"Memory Hits: {stats.get('memory_hits', 0)}",
        f"Memory Misses: {stats.get('memory_misses', 0)}",
        f"STM Size: {stats.get('stm_size', 0)}",
        f"LTM Size: {stats.get('ltm_size', 0)}",
        f"Vector Store Size: {stats.get('vector_store_size', 0)}",
        f"Embedding Dimension: {stats.get('embedding_dimension', 0)}",
        "",
    ]
    
    if stats.get('memory_hits', 0) + stats.get('memory_misses', 0) > 0:
        hit_rate = stats.get('memory_hits', 0) / (
            stats.get('memory_hits', 0) + stats.get('memory_misses', 0)
        )
        lines.append(f"Hit Rate: {hit_rate:.1%}")
    
    return "\\n".join(lines)
