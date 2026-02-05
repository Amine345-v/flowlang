"""
Test configuration and fixtures for FlowLang test suite.
"""
import os
import sys
import pytest
from pathlib import Path
from typing import Generator, Dict, Any

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flowlang.parser import parse
from flowlang.semantic import SemanticAnalyzer
from flowlang.runtime import Runtime


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return the path to the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def sample_flow() -> str:
    """Return a simple flow for testing."""
    return """
    flow example_flow(team1, team2) {
        checkpoint "start" {
            team1: do_something()
            team2: process_data()
        }
        
        checkpoint "next" {
            team1: finalize()
        }
    }
    """


@pytest.fixture
def parser_func():
    """Return the parse function."""
    return parse


@pytest.fixture
def analyzer() -> SemanticAnalyzer:
    """Return a semantic analyzer instance."""
    return SemanticAnalyzer()


@pytest.fixture
def runtime() -> Runtime:
    """Return a configured runtime instance."""
    return Runtime()


@pytest.fixture
def mock_ai_provider(monkeypatch):
    """Mock AI provider for testing."""
    from unittest.mock import MagicMock
    
    mock = MagicMock()
    mock.generate.return_value = {"result": "test response"}
    
    # Patch the AI provider in the runtime
    monkeypatch.setattr("flowlang.ai_providers.get_ai_provider", lambda: mock)
    return mock
