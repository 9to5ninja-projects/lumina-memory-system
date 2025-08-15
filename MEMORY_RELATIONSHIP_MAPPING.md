# Memory Class Relationship Mapping

## Field-by-Field Comparison

### Core Identity Fields
| Field | Main Branch Memory | XP Core MemoryUnit | Bridge Memory | Main Branch MemoryEntry |
|-------|-------------------|-------------------|---------------|------------------------|
| ID | `id: str` | `id: bytes` (BLAKE3) | `node_id: int` | `id: str` (UUID) |
| Content | `content: str` | ❌ (encrypted?) | ❌ (implicit) | `content: str` |

### Vector Representations  
| Field | Main Branch Memory | XP Core MemoryUnit | Main Branch MemoryEntry |
|-------|-------------------|-------------------|------------------------|
| Vector | `embedding: np.ndarray` | `shape: np.ndarray` ("HOLOGRAPHIC CORE") | `embedding: Optional[np.ndarray]` |

### Importance/Salience
| Field | Main Branch Memory | XP Core MemoryUnit | Main Branch MemoryEntry |
|-------|-------------------|-------------------|------------------------|
| Importance | `salience: float` | `salience: np.float32` | `importance_score: float` |

### Temporal Fields
| Field | Main Branch Memory | XP Core MemoryUnit | Main Branch MemoryEntry |
|-------|-------------------|-------------------|------------------------|
| Creation | `created_at: float` | `created_at` (timestamp) | `timestamp: datetime` |
| Access | ❌ | `accessed_at` | `access_count: int` |
| Update | ❌ | `updated_at` | ❌ |

### Mathematical Properties
| Field | Main Branch Memory | XP Core MemoryUnit | Main Branch MemoryEntry |
|-------|-------------------|-------------------|------------------------|
| Lineage | `lineage: List[str]` | ❌ | ❌ |
| Status | `status: Status` | ❌ | ❌ |
| Superposition | `vec_sum, weight, leaf_count` | ❌ | ❌ |
| Decay | ❌ | `half_life, decay_floor` | ❌ |
| Similarity | ❌ | `sim: int` (SimHash) | ❌ |

### Security/Metadata
| Field | Main Branch Memory | XP Core MemoryUnit | Main Branch MemoryEntry |
|-------|-------------------|-------------------|------------------------|
| Metadata | `metadata: Dict[str, object]` | `metadata` | `metadata: Dict[str, Any]` |
| Encryption | ❌ | `encryption` (key_id, nonce, etc.) | ❌ |
| Policy | ❌ | `policy` | ❌ |
| Consent | ❌ | `consent` | ❌ |
| Provenance | ❌ | `provenance` | ❌ |
| Audit | ❌ | `audit_id` | ❌ |

## Relationship Analysis

### SAME CONCEPT vs DIFFERENT CONCEPT?

**Evidence they're RELATED:**
- All have ID, content (or encrypted equivalent), vector representation
- All have importance/salience scoring
- All have metadata
- All represent "a unit of stored memory"

**Evidence they're DIFFERENT:**
- Main Branch Memory: Mathematical superposition focus
- XP Core MemoryUnit: Security/encryption/governance focus  
- Main Branch MemoryEntry: Simple storage focus

### EVOLUTIONARY STAGES?

**Progression Theory:**
1. **MemoryEntry** (core.py) - Basic storage stub
2. **Memory** (kernel.py) - Mathematical operations added
3. **MemoryUnit** (XP Core) - Full security/governance added

## ✅ **KEY FINDING: EVOLUTIONARY STAGES PATTERN**

**Status: LEAVE AS IS** - These represent progressive evolution, not conflicts

**Pattern Identified:**
1. **MemoryEntry** (core.py) → Basic storage stub
2. **Memory** (kernel.py) → Mathematical operations layer  
3. **Bridge Memory** (node-based) → Spatial representation layer
4. **MemoryUnit** (XP Core) → Full governance + holographic layer

**Conclusion:** These are **complementary stages** in the same system, not competing implementations.

## Next Steps

Continue systematic class analysis for remaining classes to identify:
- Which classes follow this evolutionary pattern
- Which classes are actual conflicts/duplicates  
- Which classes are mathematical stubs vs implementation
