---
description: Repository Information Overview
alwaysApply: true
---

# Lumina Memory System Information

## Summary
Lumina Memory System is a Python package that provides a memory system with cryptographic integrity for AI applications. It features content-addressed memory storage, vector embeddings, and both short-term and long-term memory capabilities.

## Structure
- **src/lumina_memory/**: Core package code with memory system implementation
- **tests/**: Test files for the package
- **notebooks/**: Jupyter notebooks for setup, quickstart, and benchmarks
- **docs/**: Documentation including memory contract and versioning
- **data/**: Sample data for testing and demonstrations
- **cli/**: Command-line interface tools
- **scripts/**: Utility scripts for development and maintenance
- **.github/**: GitHub workflows and templates

## Language & Runtime
**Language**: Python
**Version**: 3.10, 3.11 (from CI configuration)
**Build System**: Standard Python package structure
**Package Manager**: pip

## Dependencies
**Main Dependencies**:
- numpy
- scipy
- scikit-learn

**Development Dependencies**:
- ruff
- black
- isort
- pytest (implied by .pytest_cache)
- hypothesis (implied by test directory structure)

## Build & Installation
```bash
# Install in development mode
pip install -e .

# Install core dependencies
pip install numpy scipy scikit-learn
```

## Main Components
**Memory System**:
- `MemorySystem`: Main class providing memory functionality
- `EmbeddingProvider`: Handles vector embeddings for content
- `VectorStore`: Manages storage and retrieval of vector embeddings

**Cryptographic Features**:
- Content fingerprinting
- Event hashing and verification
- Encryption and key management
- Holographic reduced representations (HRR)

**Storage**:
- Short-term memory (STM)
- Long-term memory (LTM)
- Event store with versioning

## Testing
**Framework**: pytest with hypothesis
**Test Location**: tests/ directory
**Run Command**:
```bash
pytest
```

## Usage
The package is primarily used through Jupyter notebooks:
```bash
jupyter lab
# Run notebooks sequentially: 01_setup.ipynb → 02_quickstart.ipynb → 03_benchmarks.ipynb
```

Key functionality includes:
- Ingesting content with `memory_system.ingest()`
- Recalling memories with `memory_system.recall()`
- Consolidating memories with `memory_system.consolidate()`
- Managing memory with `memory_system.forget()`