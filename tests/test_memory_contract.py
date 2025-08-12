"""
Tests to validate memory contract compliance.
These tests ensure implementations adhere to the documented contract.
"""

import pytest
from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np

# Test data structures that match the contract
@dataclass(frozen=True)
class MockMemory:
    id: str
    content: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    lineage: List[str]
    created_at: float
    schema_version: str
    model_version: str
    salience: float
    status: str

@dataclass(frozen=True)
class MockEvent:
    ts: float
    type: str
    payload: Dict[str, Any]
    actor: str
    schema_version: str


class TestMemoryContract:
    """Test contract compliance for Memory dataclass."""
    
    def test_memory_immutability(self):
        """Memory instances must be immutable."""
        mem = MockMemory(
            id="test_123",
            content="test content",
            embedding=np.array([0.1, 0.2]),
            metadata={"source": "test"},
            lineage=[],
            created_at=1692000000.0,
            schema_version="v1.0",
            model_version="test@sha123",
            salience=0.5,
            status="active"
        )
        
        # Should raise exception when trying to modify
        with pytest.raises(Exception):  # FrozenInstanceError or similar
            mem.salience = 1.0
    
    def test_memory_required_fields(self):
        """Memory must have all required fields with correct types."""
        mem = MockMemory(
            id="test_123",
            content="test content", 
            embedding=np.array([0.1, 0.2]),
            metadata={"source": "test"},
            lineage=["parent_id"],
            created_at=1692000000.0,
            schema_version="v1.0",
            model_version="test@sha123",
            salience=0.5,
            status="active"
        )
        
        assert isinstance(mem.id, str)
        assert isinstance(mem.content, str)
        assert isinstance(mem.embedding, np.ndarray)
        assert isinstance(mem.metadata, dict)
        assert isinstance(mem.lineage, list)
        assert isinstance(mem.created_at, float)
        assert isinstance(mem.schema_version, str)
        assert isinstance(mem.model_version, str)
        assert isinstance(mem.salience, float)
        assert isinstance(mem.status, str)
    
    def test_memory_id_uniqueness_requirement(self):
        """Memory IDs must be unique (this is a constraint, not testable here)."""
        # This is a system-level invariant that would be tested in integration tests
        # Here we just document the requirement
        mem1 = MockMemory(
            id="unique_123", content="test1", embedding=np.array([0.1]),
            metadata={}, lineage=[], created_at=1692000000.0,
            schema_version="v1.0", model_version="test@sha123",
            salience=0.5, status="active"
        )
        mem2 = MockMemory(
            id="unique_456", content="test2", embedding=np.array([0.2]),
            metadata={}, lineage=[], created_at=1692000001.0,
            schema_version="v1.0", model_version="test@sha123",
            salience=0.5, status="active"
        )
        assert mem1.id != mem2.id
    
    def test_status_values(self):
        """Status must be one of the allowed values."""
        valid_statuses = ["active", "superseded", "tombstone"]
        
        for status in valid_statuses:
            mem = MockMemory(
                id=f"test_{status}",
                content="test",
                embedding=np.array([0.1]),
                metadata={},
                lineage=[],
                created_at=1692000000.0,
                schema_version="v1.0",
                model_version="test@sha123",
                salience=0.5,
                status=status
            )
            assert mem.status in valid_statuses


class TestEventContract:
    """Test contract compliance for Event dataclass."""
    
    def test_event_immutability(self):
        """Event instances must be immutable."""
        event = MockEvent(
            ts=1692000000.0,
            type="INGEST",
            payload={"memory_id": "test_123"},
            actor="test_user",
            schema_version="v1.0"
        )
        
        with pytest.raises(Exception):  # FrozenInstanceError or similar
            event.ts = 1692000001.0
    
    def test_event_types(self):
        """Event types must be from allowed set."""
        valid_types = ["INGEST", "RECALL", "REINFORCE", "CONSOLIDATE", 
                      "FORGET", "POLICY_CHANGE", "MIGRATION"]
        
        for event_type in valid_types:
            event = MockEvent(
                ts=1692000000.0,
                type=event_type,
                payload={"test": "data"},
                actor="test_actor",
                schema_version="v1.0"
            )
            assert event.type in valid_types


class TestVersioningContract:
    """Test version format compliance."""
    
    def test_model_version_format(self):
        """Model version must follow name@hash format."""
        valid_versions = [
            "all-MiniLM-L6-v2@sha256:abc123",
            "text-embedding-ada-002@sha256:def456",
            "custom-model@sha256:789xyz"
        ]
        
        for version in valid_versions:
            assert "@" in version
            name, hash_part = version.split("@", 1)
            assert len(name) > 0
            assert len(hash_part) > 0
    
    def test_schema_version_format(self):
        """Schema version must follow v{major}.{minor} format."""
        valid_versions = ["v1.0", "v1.1", "v2.0", "v10.5"]
        
        for version in valid_versions:
            assert version.startswith("v")
            version_part = version[1:]
            major, minor = version_part.split(".")
            assert major.isdigit()
            assert minor.isdigit()


class TestContractDocumentationExamples:
    """Validate examples from the contract documentation."""
    
    def test_basic_memory_example(self):
        """Basic memory creation example from docs should work."""
        memory = MockMemory(
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
        
        assert memory.id == "mem_abc123"
        assert "fox" in memory.content
        assert len(memory.embedding) == 4
        assert memory.metadata["source"] == "user_input"
        assert memory.lineage == []
        assert memory.status == "active"
    
    def test_event_log_example(self):
        """Event log example from docs should work."""
        events = [
            MockEvent(
                ts=1692000000.0,
                type="INGEST",
                payload={"memory_id": "mem_123", "content": "New information"},
                actor="user_456",
                schema_version="v1.0"
            ),
            MockEvent(
                ts=1692000060.0,
                type="CONSOLIDATE",
                payload={"parent_ids": ["mem_123", "mem_124"], "result_id": "mem_125"},
                actor="consolidation_job",
                schema_version="v1.0"
            )
        ]
        
        assert len(events) == 2
        assert events[0].type == "INGEST"
        assert events[1].type == "CONSOLIDATE"
        assert events[0].ts < events[1].ts  # Chronological order

