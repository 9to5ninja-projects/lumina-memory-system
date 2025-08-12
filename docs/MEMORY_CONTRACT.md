# Memory Contract

**Version:** 1.0  
**Status:** Draft  
**Last Updated:** August 11, 2025

## Overview

This document defines the immutable core contract for the Lumina Memory System. This contract establishes the mathematical foundation that all implementations must preserve. Changes to this contract require migration strategies and version compatibility analysis.

## Core Types

### Memory

The fundamental unit of stored information in the system.

```python
@dataclass(frozen=True)
class Memory:
    id: str                    # Unique identifier (never reused)
    content: str              # Canonicalized text or serialized object
    embedding: np.ndarray     # Vector representation of content
    metadata: Dict[str, Any]  # Source, topic, trust, PII flags, etc.
    lineage: List[str]        # Ancestry IDs (if consolidated/merged)
    created_at: float         # Unix timestamp
    schema_version: str       # Schema compatibility version
    model_version: str        # Embedding model name + hash
    salience: float           # [0, +), used for routing/eviction
    status: Status           # "active", "superseded", "tombstone"
```

**Invariants:**
- `id` is globally unique and never reused
- `embedding` dimensions match `model_version` specification
- `lineage` contains only valid Memory IDs
- `salience` is non-negative
- `status` transitions: active  superseded  tombstone (one-way)

### Event

Append-only log entries that capture all state changes.

```python
@dataclass(frozen=True)
class Event:
    ts: float                 # Unix timestamp
    type: EventType          # Event classification
    payload: Dict[str, Any]  # Event-specific data
    actor: str               # System component or user ID
    schema_version: str      # Event schema version
```

**Event Types:**
- `INGEST` - New content added to system
- `RECALL` - Query performed against memory store
- `REINFORCE` - Salience increased for existing memory
- `CONSOLIDATE` - Multiple memories merged via superposition
- `FORGET` - Memory marked as superseded/tombstone
- `POLICY_CHANGE` - System parameters modified
- `MIGRATION` - Schema or model version upgrade

## Kernel Operations

Pure functions that transform Memory instances. **No I/O, no side effects, deterministic.**

### superpose(a: Memory, b: Memory)  Memory

Merge/compose information from two memories.

**Mathematical Properties:**
- **Associative**: `superpose(superpose(a,b), c)  superpose(a, superpose(b,c))`
- **Commutative**: `superpose(a,b)  superpose(b,a)` (w.r.t. embedding and lineage)
- **Idempotent**: `superpose(a,a) = a` when no new information

**Behavior:**
```python
# Lineage composition
result.lineage = sorted(set(a.lineage) | set(b.lineage) | {a.id, b.id})

# Embedding combination (L2-normalized average)
embedding_combined = normalize((a.embedding + b.embedding) / 2)

# Content combination (deterministic order)
if a.content == b.content:
    result.content = a.content
else:
    # Order by ID for deterministic result
    ids_sorted = sorted([a.id, b.id])
    if ids_sorted == [a.id, b.id]:
        result.content = f"{a.content}\n{b.content}"
    else:
        result.content = f"{b.content}\n{a.content}"

# Salience aggregation (maximum preserves strongest signal)
result.salience = max(a.salience, b.salience)

# New deterministic ID
result.id = deterministic_hash("superpose", a.id, b.id, model_version)
```

**Preconditions:**
- `a.model_version == b.model_version`
- `a.schema_version == b.schema_version`
- Both memories have `status == "active"`

### reinforce(m: Memory, credit: float)  Memory

Increase salience by bounded amount.

**Mathematical Properties:**
- **Monotonic**: `result.salience >= m.salience`
- **Bounded**: `result.salience - m.salience <= SALIENCE_CAP`
- **Non-destructive**: Only salience changes

**Behavior:**
```python
delta = max(0.0, min(credit, SALIENCE_CAP))
result = replace(m, salience=m.salience + delta)
```

**Constants:**
- `SALIENCE_CAP = 1.0` (prevent unbounded growth)

### decay(m: Memory, dt: float, half_life: float = 168.0)  Memory

Apply exponential decay to salience.

**Mathematical Properties:**
- **Non-increasing**: `result.salience <= m.salience`
- **Continuous**: Small `dt` changes produce small salience changes
- **Deterministic**: Same inputs always produce same outputs

**Behavior:**
```python
if m.salience <= 0.0:
    return m
factor = 0.5 ** (dt / half_life)
new_salience = max(0.0, m.salience * factor)
result = replace(m, salience=new_salience)
```

### forget(m: Memory, criteria: Dict[str, Any])  Memory

Mark memory for non-destructive removal.

**Mathematical Properties:**
- **Non-destructive**: Core fields (id, content, embedding) unchanged
- **Status transition**: `active  superseded` or `active  tombstone`
- **Irreversible**: Cannot transition back to active

**Behavior:**
```python
mode = criteria.get("mode", "tombstone")
new_status = "superseded" if mode == "supersede" else "tombstone"
result = replace(m, status=new_status)
```

## Global Invariants

### Identity Uniqueness
- Memory IDs are globally unique across all time
- IDs are never reused, even after forgetting

### Deterministic Rebuilds
- Same event log + same versions  identical IndexState
- Same query + same IndexState  identical recall results
- Deterministic ID generation for all operations

### Version Safety
- Memory indices never mix embeddings from different `model_version`
- Schema migrations preserve semantic meaning
- Embedding space contamination is prevented

### Lineage Integrity
- `lineage` field contains only valid Memory IDs
- Lineage forms a directed acyclic graph (DAG)
- Consolidated memories preserve full ancestry

### Content Normalization
- Identical normalized content maps to same `content_hash`
- Duplicates are detected and gated at ingestion
- Content canonicalization is deterministic

## Event Sourcing Model

### IndexState
Materialized view rebuilt from event log:

```python
@dataclass
class IndexState:
    memories: Dict[str, Memory]        # Active memory store
    vector_index: VectorIndex          # Similarity search index
    metadata_index: Dict[str, Any]     # Structured query index
    lineage_graph: Dict[str, List[str]] # Ancestry relationships
    event_offset: int                  # Last processed event
    schema_version: str                # Index schema version
    model_version: str                 # Embedding model version
```

### Rebuild Process
```python
def rebuild_index(event_store: EventStore, target_offset: int) -> IndexState:
    """Deterministically rebuild index from events."""
    index = IndexState.empty()
    for event in event_store.read_from(0, target_offset):
        index = apply_event(index, event)
    return index
```

## Versioning & Migrations

### Model Version Format
```python
model_version = f"{model_name}@{model_hash[:8]}"
# Example: "all-MiniLM-L6-v2@sha256:abc123"
```

### Schema Version Format
```python
schema_version = "v{major}.{minor}"
# Example: "v1.0", "v1.1", "v2.0"
```

### Migration Process
1. **Pre-migration validation**: Verify current state consistency
2. **Parallel embedding**: Generate new embeddings with new model
3. **Migration event**: Write `MIGRATION` event with mapping
4. **Index rebuild**: Create new IndexState with new embeddings
5. **Atomic switch**: Update system pointers
6. **Cleanup**: Archive old embeddings after grace period

### Breaking Changes
- **Major schema version**: Incompatible Memory structure changes
- **Model version**: Different embedding dimensions or semantics
- **Event schema**: Incompatible event payload formats

## Failure Modes & Mitigation

### Catastrophic Forgetting
**Problem**: Important memories lost due to aggressive eviction

**Mitigation:**
- Novelty gates prevent duplicate ingestion
- Topic reservoirs maintain diversity
- Consolidated lineage preserves information density
- Salience-based eviction protects high-value memories

### Poisoning/Injection Attacks
**Problem**: Malicious content corrupts memory system

**Mitigation:**
- Content validation at ingestion gates
- Trust-weighted similarity scoring
- PII detection and redaction
- Anomaly detection on embedding distributions

### Feedback Collapse
**Problem**: System retrieves its own generated content

**Mitigation:**
- Source metadata tracking (`synthetic: true`)
- Synthetic content quotas in retrieval
- Generation lineage tracking
- Temporal filtering for recent synthetics

### Version Drift
**Problem**: Embedding spaces become incompatible over time

**Mitigation:**
- Model version pinning with hashes
- Migration validation on representative datasets
- Rollback capabilities to previous versions
- Gradual model updates with A/B testing

## Rollback Strategy

### Snapshot Creation
```python
@dataclass
class Snapshot:
    index_state: IndexState
    event_offset: int
    timestamp: float
    model_version: str
    schema_version: str
```

### Rollback Process
1. **Identify target snapshot**: Choose stable checkpoint
2. **Stop ingestion**: Prevent new events during rollback
3. **Restore index**: Load snapshot IndexState
4. **Replay events**: Apply events from snapshot to current
5. **Validate consistency**: Verify deterministic rebuild
6. **Resume operations**: Re-enable ingestion and queries

### Rollback Triggers
- Property test failures in production
- Significant performance degradation
- Data corruption detected
- Failed migration requiring revert

## Examples

### Basic Memory Creation
```python
memory = Memory(
    id="mem_abc123",
    content="The quick brown fox jumps over the lazy dog.",
    embedding=np.array([0.1, 0.2, -0.3, 0.4]),
    metadata={"source": "user_input", "topic": "example"},
    lineage=[],
    created_at=1692000000.0,
    schema_version="v1.0",
    model_version="all-MiniLM-L6-v2@sha256:abc123",
    salience=0.5,
    status="active"
)
```

### Superposition Example
```python
# Original memories
mem_a = Memory(id="a", content="Cats are mammals", ...)
mem_b = Memory(id="b", content="Cats are carnivores", ...)

# Superposed result
mem_c = superpose(mem_a, mem_b)
# mem_c.content = "Cats are mammals\nCats are carnivores"
# mem_c.lineage = ["a", "b"]
# mem_c.id = "mem_deterministic_hash_abc"
```

### Event Log Example
```python
events = [
    Event(
        ts=1692000000.0,
        type="INGEST",
        payload={"memory_id": "mem_123", "content": "New information"},
        actor="user_456",
        schema_version="v1.0"
    ),
    Event(
        ts=1692000060.0,
        type="CONSOLIDATE",
        payload={"parent_ids": ["mem_123", "mem_124"], "result_id": "mem_125"},
        actor="consolidation_job",
        schema_version="v1.0"
    )
]
```

## Compliance Testing

All implementations must pass:
- **Property tests**: Mathematical invariants verified with Hypothesis
- **Deterministic rebuild**: Same events  same results
- **Version compatibility**: Migration and rollback procedures
- **Performance benchmarks**: Recall@K and latency requirements

---

**This contract is immutable once ratified. All changes require formal migration strategy.**
