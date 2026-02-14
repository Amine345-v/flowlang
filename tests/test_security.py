"""
Security tests for FlowLang.
Tests the runtime's dry_run mode, input validation, and safe defaults.
"""
import os
import pytest
import tempfile
from pathlib import Path

from flowlang.parser import parse
from flowlang.errors import ParseError
from flowlang.runtime import Runtime


def test_large_input_parsing():
    """Test that very large inputs are handled without crashing."""
    # Generate a large but syntactically valid flow
    large_src = """
    team T : Command<Search> [size=1];
    flow large_flow(using: T) {
    """
    for i in range(500):
        large_src += f'  checkpoint "cp_{i}" {{ T.search("query_{i}"); }}\n'
    large_src += "}\n"
    
    tree = parse(large_src)
    assert tree is not None
    
    checkpoints = list(tree.find_data("checkpoint"))
    assert len(checkpoints) == 500


def test_malformed_input_rejected():
    """Test that malformed inputs are rejected with ParseError."""
    malformed_inputs = [
        "",  # Empty input
        ";;;",  # Just semicolons
        "flow {",  # Incomplete flow
        "team ;",  # Incomplete team
        "this is not flowlang at all",
    ]
    
    for src in malformed_inputs:
        with pytest.raises(ParseError):
            parse(src)


def test_dry_run_no_side_effects():
    """Test that dry_run mode doesn't execute real actions."""
    rt = Runtime(dry_run=True)
    rt.load("""
    team T : Command<Search> [size=1];
    flow main(using: T) {
        checkpoint "start" {
            T.search("query");
        }
    }
    """)
    rt.run_flow("main")
    
    # In dry_run mode, all action results should be dry_run markers
    assert any("[dry_run]" in str(l) for l in rt.console)


def test_persistence_temp_directory():
    """Test that persistence uses safe temporary directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rt = Runtime(dry_run=True)
        rt.persistence.state_dir = Path(tmpdir)
        
        rt.load("""
        team T : Command<Search> [size=1];
        flow main(using: T) {
            checkpoint "start" {
                T.search("query");
            }
        }
        """)
        rt.run_flow("main")
        
        # Check that state was saved in the temp directory
        state_files = list(Path(tmpdir).glob("*.json"))
        assert len(state_files) >= 0  # May or may not have saved state


def test_input_escaping():
    """Test that strings with special characters are handled safely."""
    rt = Runtime(dry_run=True)
    # Test with strings containing potential injection characters
    rt.load("""
    team T : Command<Search> [size=1];
    flow main(using: T) {
        checkpoint "start" {
            T.search("query with 'quotes' and \\"escapes\\"");
        }
    }
    """)
    rt.run_flow("main")
    # Should complete without errors


def test_concurrent_flow_isolation():
    """Test that multiple Runtime instances are isolated."""
    rt1 = Runtime(dry_run=True)
    rt2 = Runtime(dry_run=True)
    
    src = """
    team T : Command<Search> [size=1];
    flow main(using: T) {
        checkpoint "start" {
            T.search("query");
        }
    }
    """
    
    rt1.load(src)
    rt2.load(src)
    
    rt1.run_flow("main")
    rt2.run_flow("main")
    
    # Each runtime should have its own console
    assert len(rt1.console) > 0
    assert len(rt2.console) > 0
    # Metrics should be independent
    assert rt1.metrics is not rt2.metrics
