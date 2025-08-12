"""
Test configuration and fixtures for Lumina Memory System tests.
"""

import pytest
import sys
import os
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from lumina_memory.test_store import event_store_factory as _event_store_factory

@pytest.fixture
def event_store_factory():
    """Provide event store factory for tests."""
    return _event_store_factory
