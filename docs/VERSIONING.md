# Versioning Strategy

## System Versioning

Lumina Memory follows semantic versioning with milestone-based progression:

### Version Format
```
v{MAJOR}.{MINOR}.{PATCH}-{STAGE}
```

- **MAJOR**: Breaking changes to Memory Contract or kernel invariants  
- **MINOR**: New milestone completion (M1-M12)
- **PATCH**: Bug fixes, documentation updates  
- **STAGE**: alpha, beta, rc, stable

### Milestone-Version Mapping

| Version | Milestone | Description | Status |
|---------|-----------|-------------|---------|
| v0.1.0-alpha | M1 | Memory Contract & Documentation |  Complete |
| v0.2.0-alpha | M2 | Kernel Module (Pure Functions) |  Planned |
| v0.3.0-alpha | M3 | Property Tests (Hypothesis) |  Planned |
| v0.4.0-alpha | M4 | Event Store & Deterministic Rebuild |  Planned |
| v0.5.0-alpha | M5 | Embedding Versioning & Migrations |  Planned |
| v0.6.0-alpha | M6 | Novelty Gate & Topic Reservoirs |  Planned |
| v0.7.0-alpha | M7 | Consolidation Job & Lineage |  Planned |
| v0.8.0-alpha | M8 | Decay/Forget Policies |  Planned |
| v0.9.0-alpha | M9 | Observability & Metrics |  Planned |
| v0.10.0-beta | M10 | Threat Gates & Production Ready |  Planned |
| v1.0.0-rc.1 | - | Release Candidate |  Planned |
| v1.0.0 | - | Stable Production Release |  Planned |

## Contract Versioning

The Memory Contract has independent versioning:

- **Contract v1.0**: Current specification (immutable once ratified)
- **Contract v2.0**: Future major schema changes (if needed)

## Component Versioning

### Schema Versions
```python
schema_version = "v1.0"  # Memory dataclass structure
event_schema_version = "v1.0"  # Event log format  
```

### Model Versions  
```python
model_version = "all-MiniLM-L6-v2@sha256:abc12345"
# Format: {model_name}@{content_hash}
```

## Breaking Changes

**Major Version Bumps Required For:**
- Memory Contract modifications
- Kernel operation signature changes
- Event schema incompatibilities
- Storage format changes

**Minor Version Bumps For:**
- New milestone completions
- Backward-compatible feature additions
- Policy algorithm improvements

**Patch Version Bumps For:**
- Bug fixes
- Documentation updates
- Performance optimizations
- Test improvements

## Version Control Strategy

### Git Tags
Each milestone completion gets a git tag:
```bash
git tag v0.1.0-alpha  # M1 completion
git tag v0.2.0-alpha  # M2 completion
```

### Branch Strategy
- `main`: Current stable version
- `develop`: Integration branch for next version
- `feature/*`: Individual milestone development
- `hotfix/*`: Critical production fixes

### Release Process
1. Complete milestone on feature branch
2. Merge to `develop` 
3. Run full test suite + benchmarks
4. Merge to `main`
5. Tag version
6. Update documentation
7. Create GitHub release

---

**Current Version: v0.1.0-alpha (M1: Memory Contract Complete)**
