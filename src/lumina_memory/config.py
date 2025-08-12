"""Configuration management for Lumina Memory System."""

import os
import json
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class LuminaConfig:
    """Central configuration for Lumina Memory System."""
    
    # Core system settings
    embedding_dim: int = 384
    vector_store_type: str = "faiss"  # faiss, chromadb
    similarity_metric: str = "cosine"  # cosine, euclidean, inner_product
    
    # Memory settings
    stm_capacity: int = 1000
    ltm_capacity: int = 10000
    consolidation_threshold: float = 0.7
    
    # Model settings
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    
    # Performance settings
    batch_size: int = 32
    max_workers: int = 4
    cache_size: int = 1000
    
    # Paths (configurable via environment)
    data_dir: str = "./data"
    models_dir: str = "./models"
    logs_dir: str = "./logs" 
    cache_dir: str = "./cache"
    
    # Determinism
    random_seed: int = 42
    deterministic_mode: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    
    @classmethod
    def from_env(cls) -> "LuminaConfig":
        """Create config from environment variables."""
        config = cls()
        
        # Environment variable mappings
        env_mappings = {
            "LUMINA_EMBEDDING_DIM": ("embedding_dim", int),
            "LUMINA_VECTOR_STORE": ("vector_store_type", str),
            "LUMINA_SIMILARITY_METRIC": ("similarity_metric", str),
            "LUMINA_STM_CAPACITY": ("stm_capacity", int),
            "LUMINA_LTM_CAPACITY": ("ltm_capacity", int),
            "LUMINA_MODEL_NAME": ("sentence_transformer_model", str),
            "LUMINA_DEVICE": ("embedding_device", str),
            "LUMINA_BATCH_SIZE": ("batch_size", int),
            "LUMINA_DATA_DIR": ("data_dir", str),
            "LUMINA_MODELS_DIR": ("models_dir", str),
            "LUMINA_LOGS_DIR": ("logs_dir", str),
            "LUMINA_RANDOM_SEED": ("random_seed", int),
            "LUMINA_DETERMINISTIC": ("deterministic_mode", bool),
            "LUMINA_LOG_LEVEL": ("log_level", str),
        }
        
        for env_var, (attr_name, attr_type) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    if attr_type == bool:
                        value = env_value.lower() in ("true", "1", "yes", "on")
                    else:
                        value = attr_type(env_value)
                    setattr(config, attr_name, value)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid value for {env_var}: {env_value}")
        
        return config
    
    def create_directories(self) -> None:
        """Create necessary directories."""
        for dir_path in [self.data_dir, self.models_dir, self.logs_dir, self.cache_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def setup_determinism(self) -> None:
        """Setup deterministic behavior."""
        if self.deterministic_mode:
            random.seed(self.random_seed)
            os.environ["PYTHONHASHSEED"] = str(self.random_seed)
            
            try:
                import numpy as np
                np.random.seed(self.random_seed)
            except ImportError:
                pass
            
            try:
                import torch
                torch.manual_seed(self.random_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.random_seed)
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
            except ImportError:
                pass
    
    def validate(self) -> bool:
        """Validate configuration values."""
        errors = []
        
        if self.embedding_dim <= 0:
            errors.append("embedding_dim must be positive")
            
        if self.vector_store_type not in ["faiss", "chromadb"]:
            errors.append("vector_store_type must be 'faiss' or 'chromadb'")
            
        if self.similarity_metric not in ["cosine", "euclidean", "inner_product"]:
            errors.append("similarity_metric must be valid")
            
        if self.stm_capacity <= 0 or self.ltm_capacity <= 0:
            errors.append("memory capacities must be positive")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
        return True
    
    def save(self, path: str = "lumina_config.json") -> None:
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str = "lumina_config.json") -> "LuminaConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)
