# Lumina Memory: Holographic Memory System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)]()

A production-ready holographic memory system for AI applications with pure functional kernel, event sourcing, and mathematical guarantees.

##  Documentation

- **[Memory Contract](docs/MEMORY_CONTRACT.md)** - Core specification and mathematical invariants
- **[Development Roadmap](DEVELOPMENT_ROADMAP.md)** - Strategic architecture and milestones  
- **[Developer Guide](READ_FIRST.md)** - Setup and contribution guidelines

##  Architecture

Lumina Memory is built on a **pure functional kernel** with **event sourcing**:

- **Kernel**: Pure mathematical operations (superpose, reinforce, decay, forget)
- **Event Store**: Append-only log of all state changes
- **Index**: Materialized view rebuilt deterministically from events
- **Policies**: Higher-level intelligence (novelty gates, consolidation, eviction)

### Key Properties
- **Deterministic**: Same events  same results
- **Immutable**: Operations create new instances, never mutate
- **Versioned**: Safe model upgrades with migration paths
- **Rollback-capable**: Point-in-time recovery from event log

##  Quick Start

### Installation

```bash
# Install from source  
git clone https://github.com/9to5ninja-projects/lumina-memory-system.git
cd lumina-memory-system
pip install -e .

# Install development dependencies
pip install -e .[dev]
```

### Basic Usage

```python
from lumina_memory import MemorySystem, Memory
import numpy as np

# Initialize system
system = MemorySystem()

# Create a memory
memory = Memory(
    id="example_001",
    content="Cats are curious creatures",
    embedding=np.random.rand(384),  # Your embedding
    metadata={"source": "user", "topic": "animals"},
    lineage=[],
    created_at=1692000000.0,
    schema_version="v1.0", 
    model_version="all-MiniLM-L6-v2@sha256:abc123",
    salience=1.0,
    status="active"
)

# Store and query
system.ingest(memory)
results = system.recall("curious cats", k=5)
```

##  Memory Algebra  

The system implements a **memory algebra** with mathematical guarantees:

```python
# Superposition (associative, commutative)
combined = superpose(memory_a, memory_b)

# Reinforcement (monotonic, bounded)
stronger = reinforce(memory, credit=0.5)

# Decay (exponential, deterministic)
aged = decay(memory, dt=24.0, half_life=168.0)

# Forgetting (non-destructive)
forgotten = forget(memory, criteria={"reason": "outdated"})
```

##  Event Sourcing

All operations are captured as events for complete auditability:

```python
# All state changes become events
events = [
    Event(type="INGEST", payload={"memory": memory}, ...),
    Event(type="CONSOLIDATE", payload={"parents": [...], "result": combined}, ...),
    Event(type="FORGET", payload={"memory_id": "...", "reason": "..."}, ...)
]

# Deterministic rebuild from events
index = rebuild_index(event_store, target_offset=1000)
```

##  Testing

The system includes comprehensive property-based testing:

```bash
# Run all tests
pytest tests/ -v

# Run property tests (mathematical invariants)  
pytest tests/test_kernel_properties.py -v

# Run deterministic rebuild tests
pytest tests/test_rebuild_deterministic.py -v

# Local CI check
python local_ci_check.py
```

##  Development

### VS Code Setup
The project includes VS Code tasks and settings:
- `Ctrl+Shift+P`  "Tasks: Run Task"  Select test/lint tasks
- Format on save enabled
- Copilot integration configured

### Branch Strategy
- `main` - Stable releases
- `feature/*` - New capabilities  
- `job/*` - Background processing
- `infra/*` - Infrastructure changes
- `memory-system/*` - Core kernel changes

### Required Checks
- All tests pass (unit + property + integration)
- Code formatting (black, ruff, isort)
- Kernel invariants preserved
- Deterministic rebuild validation

##  Performance

Designed for production workloads:
- **Deterministic**: Reproducible results across runs
- **Scalable**: Event sourcing enables horizontal scaling  
- **Fast**: Optimized vector operations and indexing
- **Safe**: Mathematical guarantees prevent data corruption

##  Security

- **Input validation**: Content filtering and PII detection
- **Poisoning resistance**: Trust-weighted similarity scoring
- **Feedback prevention**: Synthetic content tracking
- **Audit trail**: Complete event history for forensics

##  License

MIT License - see [LICENSE](LICENSE) file for details.

##  Contributing

1. Read the [Memory Contract](docs/MEMORY_CONTRACT.md) 
2. Follow the [Developer Guide](READ_FIRST.md)
3. Create feature branches following naming conventions
4. Ensure all tests pass including property tests
5. Add migration notes for breaking changes

**Note**: Changes to `src/lumina_memory/kernel.py` require special review as they affect mathematical guarantees.
