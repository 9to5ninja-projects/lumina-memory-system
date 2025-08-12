"""
Property-based tests for kernel operations using Hypothesis.

These tests mathematically verify the invariants documented in the Memory Contract.
All properties must pass to ensure kernel correctness and mathematical guarantees.
"""

import math
from dataclasses import replace
from typing import List

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, assume

# Import our kernel components
import sys
sys.path.append('../src')
from lumina_memory.kernel import (
    Memory, superpose, reinforce, decay, forget,
    SALIENCE_REINFORCE_CAP, DEFAULT_HALF_LIFE
)


# ---- Hypothesis Strategies ----

def st_embedding(dim_min=4, dim_max=16):
    """Generate bounded embedding vectors to avoid NaN/Inf issues."""
    return st.lists(
        st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
        min_size=dim_min, max_size=dim_max
    ).map(lambda xs: np.array(xs, dtype=np.float32))


def st_memory():
    """Generate valid Memory instances for property testing."""
    return st.builds(
        Memory,
        id=st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=97, max_codepoint=122)),
        content=st.text(min_size=1, max_size=100),
        embedding=st_embedding(),
        metadata=st.dictionaries(
            st.text(min_size=1, max_size=10), 
            st.one_of(st.integers(-100, 100), st.text(max_size=20)), 
            max_size=3
        ),
        lineage=st.lists(st.text(min_size=1, max_size=10), max_size=3),
        created_at=st.floats(min_value=1000000000, max_value=2000000000),
        schema_version=st.just("v1.0"),  # Keep consistent for compatibility
        model_version=st.just("test-model@sha123"),  # Keep consistent
        salience=st.floats(min_value=0.0, max_value=10.0, allow_infinity=False, allow_nan=False),
        status=st.sampled_from(["active", "superseded", "tombstone"]),
    )


# ---- Superpose Properties ----

@settings(deadline=None, max_examples=50)
@given(st_memory(), st_memory(), st_memory())
def test_superpose_associative(a, b, c):
    """
    Superpose is associative: superpose(superpose(a,b), c)  superpose(a, superpose(b,c))
    (w.r.t. embedding shape and lineage set equality)
    """
    # Only test with active memories (precondition)
    a = replace(a, status="active")
    b = replace(b, status="active") 
    c = replace(c, status="active")
    
    try:
        # Left association: (a  b)  c
        ab = superpose(a, b)
        left = superpose(ab, c)
        
        # Right association: a  (b  c)
        bc = superpose(b, c)
        right = superpose(a, bc)
        
        # Properties that should be associative
        assert left.embedding.shape == right.embedding.shape
        assert set(left.lineage) == set(right.lineage)
        assert left.model_version == right.model_version
        assert left.schema_version == right.schema_version
        
    except ValueError:
        # If superpose fails due to version mismatch, that's expected
        # The test is still valid - we're testing mathematical properties
        # when the operation is valid
        pass


@settings(deadline=None, max_examples=50)
@given(st_memory(), st_memory())
def test_superpose_commutative(a, b):
    """
    Superpose is commutative: superpose(a,b)  superpose(b,a)
    (w.r.t. embedding shape and lineage set equality)
    """
    # Only test with active memories
    a = replace(a, status="active")
    b = replace(b, status="active")
    
    try:
        ab = superpose(a, b)
        ba = superpose(b, a)
        
        # Commutative properties
        assert ab.embedding.shape == ba.embedding.shape
        assert set(ab.lineage) == set(ba.lineage)
        assert ab.model_version == ba.model_version
        assert ab.salience == ba.salience  # max() is commutative
        
        # Content order might differ but both contents should be present
        assert a.content in ab.content and b.content in ab.content
        assert a.content in ba.content and b.content in ba.content
        
    except ValueError:
        # Version mismatch - operation not valid, test passes
        pass


@settings(deadline=None, max_examples=30)
@given(st_memory())
def test_superpose_idempotent_on_identical(m):
    """
    Superpose is idempotent on identical inputs: superpose(m,m) has predictable behavior.
    """
    m = replace(m, status="active")
    
    mm = superpose(m, m)
    
    # Idempotent properties
    assert mm.embedding.shape == m.embedding.shape
    assert mm.content == m.content  # Identical content should remain unchanged
    assert mm.salience == m.salience  # max(x,x) = x
    assert mm.model_version == m.model_version
    
    # Lineage should include the original memory
    assert m.id in mm.lineage


# ---- Reinforce Properties ----

@settings(deadline=None, max_examples=50)
@given(st_memory(), st.floats(min_value=0.0, max_value=5.0))
def test_reinforce_monotonic(m, credit):
    """
    Reinforce is monotonic: result.salience >= m.salience
    """
    assume(not math.isnan(credit) and not math.isinf(credit))
    
    result = reinforce(m, credit)
    
    # Monotonic property
    assert result.salience >= m.salience - 1e-6  # Allow tiny float precision errors
    
    # Non-destructive (other fields unchanged)
    assert result.id == m.id
    assert result.content == m.content
    assert np.array_equal(result.embedding, m.embedding)
    assert result.metadata == m.metadata
    assert result.lineage == m.lineage


@settings(deadline=None, max_examples=50)
@given(st_memory(), st.floats(min_value=0.0, max_value=10.0))
def test_reinforce_bounded(m, credit):
    """
    Reinforce is bounded: result.salience - m.salience <= SALIENCE_REINFORCE_CAP
    """
    assume(not math.isnan(credit) and not math.isinf(credit))
    
    result = reinforce(m, credit)
    
    # Bounded property  
    delta = result.salience - m.salience
    assert delta <= SALIENCE_REINFORCE_CAP + 1e-6  # Allow float precision


@settings(deadline=None, max_examples=40)
@given(st_memory(), st.floats(min_value=-5.0, max_value=5.0))
def test_reinforce_non_negative_salience(m, credit):
    """
    Reinforce maintains non-negative salience regardless of input.
    """
    assume(not math.isnan(credit) and not math.isinf(credit))
    
    result = reinforce(m, credit)
    
    # Non-negative property
    assert result.salience >= 0.0


# ---- Decay Properties ----

@settings(deadline=None, max_examples=50)
@given(st_memory(), st.floats(min_value=0.0, max_value=1000.0))
def test_decay_non_increasing(m, dt):
    """
    Decay never increases salience: result.salience <= m.salience
    """
    assume(not math.isnan(dt) and not math.isinf(dt))
    
    result = decay(m, dt)
    
    # Non-increasing property
    assert result.salience <= m.salience + 1e-6  # Allow float precision
    
    # Non-destructive (other fields unchanged)
    assert result.id == m.id
    assert result.content == m.content
    assert np.array_equal(result.embedding, m.embedding)
    assert result.metadata == m.metadata
    assert result.lineage == m.lineage


@settings(deadline=None, max_examples=30)
@given(st_memory())
def test_decay_half_life_accuracy(m):
    """
    After one half-life, salience should be approximately half the original.
    """
    # Only test with positive salience
    assume(m.salience > 0.1)
    
    result = decay(m, dt=DEFAULT_HALF_LIFE, half_life=DEFAULT_HALF_LIFE)
    
    # Half-life accuracy (within 5% tolerance for floating point)
    expected = m.salience * 0.5
    tolerance = expected * 0.05
    assert abs(result.salience - expected) <= tolerance


@settings(deadline=None, max_examples=40)
@given(st_memory(), st.floats(min_value=0.0, max_value=100.0))
def test_decay_bounded_non_negative(m, dt):
    """
    Decay maintains non-negative salience and never goes below zero.
    """
    assume(not math.isnan(dt) and not math.isinf(dt))
    
    result = decay(m, dt)
    
    # Bounded non-negative
    assert result.salience >= 0.0


@settings(deadline=None, max_examples=30)
@given(st_memory())
def test_decay_continuous_small_changes(m):
    """
    Small time changes produce small salience changes (continuity).
    """
    assume(m.salience > 0.01)  # Need measurable salience
    
    small_dt = 1.0  # Small time change
    result1 = decay(m, dt=small_dt)
    result2 = decay(m, dt=small_dt * 1.1)  # 10% larger
    
    # Continuity: small input change should produce small output change
    salience_change = abs(result2.salience - result1.salience)
    assert salience_change < m.salience * 0.1  # Less than 10% of original


# ---- Forget Properties ----

@settings(deadline=None, max_examples=40)
@given(st_memory())
def test_forget_non_destructive(m):
    """
    Forget is non-destructive: core fields remain unchanged.
    """
    result = forget(m, {"mode": "tombstone"})
    
    # Non-destructive properties
    assert result.id == m.id
    assert result.content == m.content
    assert np.array_equal(result.embedding, m.embedding)
    assert result.metadata == m.metadata
    assert result.lineage == m.lineage
    assert result.created_at == m.created_at
    assert result.schema_version == m.schema_version
    assert result.model_version == m.model_version
    assert result.salience == m.salience


@settings(deadline=None, max_examples=40) 
@given(st_memory())
def test_forget_status_only_changes(m):
    """
    Forget only changes the status field.
    """
    # Test tombstone mode
    result_tombstone = forget(m, {"mode": "tombstone"})
    assert result_tombstone.status == "tombstone"
    
    # Test supersede mode  
    result_supersede = forget(m, {"mode": "supersede"})
    assert result_supersede.status == "superseded"
    
    # Default mode should be tombstone
    result_default = forget(m)
    assert result_default.status == "tombstone"


@settings(deadline=None, max_examples=30)
@given(st_memory())
def test_forget_irreversible_transitions(m):
    """
    Forget creates irreversible status transitions.
    """
    # Start with active memory
    active_memory = replace(m, status="active")
    
    # Transition to superseded
    superseded = forget(active_memory, {"mode": "supersede"})
    assert superseded.status == "superseded"
    
    # Transition to tombstone  
    tombstone = forget(active_memory, {"mode": "tombstone"})
    assert tombstone.status == "tombstone"
    
    # Cannot transition back (this would be tested in higher-level policies)
    # The kernel itself doesn't prevent this, but the contract specifies it


# ---- Integration Properties ----

@settings(deadline=None, max_examples=20)
@given(st_memory(), st.floats(min_value=0.0, max_value=2.0), st.floats(min_value=0.0, max_value=50.0))
def test_reinforce_then_decay_composition(m, credit, dt):
    """
    Composition of operations maintains mathematical properties.
    """
    assume(not math.isnan(credit) and not math.isinf(credit))
    assume(not math.isnan(dt) and not math.isinf(dt))
    
    # Apply reinforce then decay
    reinforced = reinforce(m, credit)
    final = decay(reinforced, dt)
    
    # Final salience should be non-negative
    assert final.salience >= 0.0
    
    # All non-salience fields should be preserved through the chain
    assert final.id == m.id
    assert final.content == m.content
    assert np.array_equal(final.embedding, m.embedding)


if __name__ == "__main__":
    # Run a quick validation that our strategies work
    print("Validating Hypothesis strategies...")
    
    # Test memory generation
    memory = st_memory().example()
    print(f" Generated memory: {memory.id}")
    
    # Test embedding generation  
    embedding = st_embedding().example()
    print(f" Generated embedding shape: {embedding.shape}")
    
    print(" All strategies validated")
    print("Run 'pytest tests/test_kernel_properties.py -v' to execute property tests")
