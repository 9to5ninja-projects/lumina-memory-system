"""Test configuration module."""

import json
import tempfile
from pathlib import Path

import pytest

from lumina_memory.config import LuminaConfig


def test_config_defaults():
    """Test default configuration values."""
    config = LuminaConfig()
    
    assert config.embedding_dim == 384
    assert config.vector_store_type == "faiss"
    assert config.similarity_metric == "cosine"
    assert config.stm_capacity == 1000
    assert config.ltm_capacity == 10000
    assert config.random_seed == 42
    assert config.deterministic_mode is True


def test_config_validation():
    """Test configuration validation."""
    # Valid config
    config = LuminaConfig()
    assert config.validate() is True
    
    # Invalid embedding dimension
    config_invalid = LuminaConfig(embedding_dim=-1)
    with pytest.raises(ValueError, match="embedding_dim must be positive"):
        config_invalid.validate()
    
    # Invalid vector store type
    config_invalid = LuminaConfig(vector_store_type="invalid")
    with pytest.raises(ValueError, match="vector_store_type must be"):
        config_invalid.validate()


def test_config_save_load():
    """Test saving and loading configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "test_config.json"
        
        # Create and save config
        original_config = LuminaConfig(
            embedding_dim=256,
            stm_capacity=500,
            random_seed=123
        )
        original_config.save(str(config_path))
        
        # Load config
        loaded_config = LuminaConfig.load(str(config_path))
        
        # Check values
        assert loaded_config.embedding_dim == 256
        assert loaded_config.stm_capacity == 500
        assert loaded_config.random_seed == 123


def test_config_from_env(monkeypatch):
    """Test loading configuration from environment variables."""
    # Set environment variables
    monkeypatch.setenv("LUMINA_EMBEDDING_DIM", "512")
    monkeypatch.setenv("LUMINA_STM_CAPACITY", "2000")
    monkeypatch.setenv("LUMINA_DETERMINISTIC", "false")
    
    config = LuminaConfig.from_env()
    
    assert config.embedding_dim == 512
    assert config.stm_capacity == 2000
    assert config.deterministic_mode is False


def test_config_directory_creation():
    """Test directory creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = LuminaConfig(
            data_dir=f"{tmpdir}/data",
            models_dir=f"{tmpdir}/models",
            logs_dir=f"{tmpdir}/logs",
            cache_dir=f"{tmpdir}/cache"
        )
        
        config.create_directories()
        
        # Check directories exist
        assert Path(config.data_dir).exists()
        assert Path(config.models_dir).exists()
        assert Path(config.logs_dir).exists()
        assert Path(config.cache_dir).exists()
