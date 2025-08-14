## What changed
<!-- Brief summary of the changes -->

## Why  
<!-- Link to issue: Closes #123 -->

## Invariants & Risks
<!-- Check all that apply and provide details -->

### Kernel Invariants
- **Affected?**  No  Yes  Details:
  - [ ] Associativity preserved
  - [ ] Commutativity preserved  
  - [ ] Idempotency preserved
  - [ ] Monotonicity preserved
  - [ ] Determinism preserved

### Migration & Rollback
- **Migration needed?**  No  Yes  Script/path:
- **Rollback plan**: <!-- How to revert to snapshot N -->
- **Threat/poisoning implications**: <!-- Security considerations -->

## Evidence

### Tests
```bash
# Paste pytest -q summary here
```

### Benchmarks  
<!-- Paste metrics deltas -->
- Recall@10: baseline  new
- nDCG: baseline  new  
- p50/p95 latency: baseline  new

### Deterministic Rebuild
- [ ]  Verified on seed log  same results before/after

### Additional Evidence
<!-- Screenshots, logs, performance traces -->

## Checklist
- [ ] Tests added/updated for new functionality
- [ ] Property tests pass
- [ ] Documentation updated
- [ ] Breaking changes documented in migration guide
- [ ] Rollback tested
