"""
Minimal validation tests for M2: Kernel implementation.
Comprehensive property tests will be added in M3.
"""

import numpy as np
import pytest
from lumina_memory.kernel import Memory, superpose, reinforce, decay, forget


class TestM2KernelBasics:
    """Basic kernel functionality validation for M2."""
    
    def test_memory_creation(self):
        """Memory can be created with required fields."""
        memory = Memory(
            id="test_001",
            content="Test content",
            embedding=np.array([0.1, 0.2, 0.3]),
            metadata={"source": "test"},
            lineage=[],
            created_at=1692000000.0,
            schema_version="v1.0",
            model_version="test@sha123",
            salience=1.0,
            status="active"
        )
        
        assert memory.id == "test_001"
        assert memory.content == "Test content"
        assert memory.salience == 1.0
        assert memory.status == "active"
    
    def test_memory_immutability(self):
        """Memory instances are immutable (frozen dataclass)."""
        memory = Memory(
            id="test_002",
            content="Test",
            embedding=np.array([0.1]),
            metadata={},
            lineage=[],
            created_at=1692000000.0,
            schema_version="v1.0", 
            model_version="test@sha123",
        )
        
        # Should raise exception when trying to modify
        with pytest.raises(Exception):  # FrozenInstanceError
            memory.salience = 2.0
    
    def test_superpose_basic(self):
        """Superpose can combine two memories."""
        mem_a = Memory(
            id="a", content="Content A", embedding=np.array([1.0, 0.0]),
            metadata={"topic": "test"}, lineage=[], created_at=1692000000.0,
            schema_version="v1.0", model_version="test@sha123", salience=0.5
        )
        
        mem_b = Memory(
            id="b", content="Content B", embedding=np.array([0.0, 1.0]),
            metadata={"topic": "test"}, lineage=[], created_at=1692000001.0,
            schema_version="v1.0", model_version="test@sha123", salience=0.7
        )
        
        result = superpose(mem_a, mem_b)
        
        # Basic checks
        assert result.id != mem_a.id and result.id != mem_b.id
        assert "Content A" in result.content and "Content B" in result.content
        assert result.salience == 0.7  # max(0.5, 0.7)
        assert set(result.lineage) == {"a", "b"}
        assert result.model_version == "test@sha123"
        assert result.status == "active"
    
    def test_superpose_version_mismatch_error(self):
        """Superpose rejects different model versions."""
        mem_a = Memory(
            id="a", content="Test", embedding=np.array([1.0]),
            metadata={}, lineage=[], created_at=1692000000.0,
            schema_version="v1.0", model_version="model_v1@sha123"
        )
        
        mem_b = Memory(
            id="b", content="Test", embedding=np.array([1.0]),
            metadata={}, lineage=[], created_at=1692000000.0,
            schema_version="v1.0", model_version="model_v2@sha456"
        )
        
        with pytest.raises(ValueError, match="different model_version"):
            superpose(mem_a, mem_b)
    
    def test_reinforce_basic(self):
        """Reinforce can increase salience."""
        memory = Memory(
            id="test", content="Test", embedding=np.array([1.0]),
            metadata={}, lineage=[], created_at=1692000000.0,
            schema_version="v1.0", model_version="test@sha123",
            salience=0.5
        )
        
        result = reinforce(memory, 0.3)
        
        assert result.salience == 0.8  # 0.5 + 0.3
        assert result.id == memory.id  # Same memory, just reinforced
        assert result.content == memory.content
    
    def test_reinforce_capped(self):
        """Reinforce respects salience cap."""
        memory = Memory(
            id="test", content="Test", embedding=np.array([1.0]),
            metadata={}, lineage=[], created_at=1692000000.0,
            schema_version="v1.0", model_version="test@sha123",
            salience=0.8
        )
        
        # Try to add more than cap allows
        result = reinforce(memory, 5.0)  # Should be capped to 1.0
        
        assert result.salience == 1.8  # 0.8 + 1.0 (capped)
    
    def test_decay_basic(self):
        """Decay can reduce salience."""
        memory = Memory(
            id="test", content="Test", embedding=np.array([1.0]),
            metadata={}, lineage=[], created_at=1692000000.0,
            schema_version="v1.0", model_version="test@sha123",
            salience=1.0
        )
        
        # After one half-life, should be ~0.5
        result = decay(memory, dt=168.0)  # DEFAULT_HALF_LIFE
        
        assert result.salience < 1.0  # Decayed
        assert abs(result.salience - 0.5) < 0.01  # Approximately half
        assert result.id == memory.id
    
    def test_forget_basic(self):
        """Forget can mark memory as tombstone."""
        memory = Memory(
            id="test", content="Test", embedding=np.array([1.0]),
            metadata={}, lineage=[], created_at=1692000000.0,
            schema_version="v1.0", model_version="test@sha123",
            status="active"
        )
        
        result = forget(memory)
        
        assert result.status == "tombstone"
        assert result.id == memory.id  # Same memory
        assert result.content == memory.content  # Non-destructive
    
    def test_forget_supersede_mode(self):
        """Forget can mark memory as superseded."""
        memory = Memory(
            id="test", content="Test", embedding=np.array([1.0]),
            metadata={}, lineage=[], created_at=1692000000.0,
            schema_version="v1.0", model_version="test@sha123",
            status="active"
        )
        
        result = forget(memory, {"mode": "supersede"})
        
        assert result.status == "superseded"
        assert result.id == memory.id
        assert result.content == memory.content


if __name__ == "__main__":
    # Run tests directly for M2 validation
    import sys
    import traceback
    
    test_class = TestM2KernelBasics()
    methods = [method for method in dir(test_class) if method.startswith("test_")]
    
    passed = 0
    failed = 0
    
    for method_name in methods:
        try:
            method = getattr(test_class, method_name)
            method()
            print(f" {method_name}")
            passed += 1
        except Exception as e:
            print(f" {method_name}: {e}")
            traceback.print_exc()
            failed += 1
    
    print(f"\nM2 Kernel Validation: {passed} passed, {failed} failed")
    if failed == 0:
        print(" M2 Kernel implementation validated!")
    else:
        sys.exit(1)
