"""
Semantic analysis tests for FlowLang.
Tests the SemanticAnalyzer using the actual parse() function.
"""
import pytest
from pathlib import Path

from flowlang.parser import parse
from flowlang.semantic import SemanticAnalyzer
from flowlang.errors import SemanticError


def test_team_declaration_validation():
    """Test validation of team declarations."""
    src = """
    team devs : Command<Search> [size=2];
    team testers : Command<Judge> [size=1];
    flow F(using: devs, testers) {
        checkpoint "start" {
            devs.search("query");
            testers.judge("code", "quality");
        }
    }
    """
    
    tree = parse(src)
    analyzer = SemanticAnalyzer(tree)
    analyzer.analyze()
    
    # Check teams were registered
    assert "devs" in analyzer.teams
    assert "testers" in analyzer.teams


def test_duplicate_team_declaration():
    """Test that duplicate team declarations are caught."""
    src = """
    team devs : Command<Search> [size=1];
    team devs : Command<Search> [size=2];
    """
    
    tree = parse(src)
    analyzer = SemanticAnalyzer(tree)
    
    with pytest.raises(SemanticError, match="Duplicate team"):
        analyzer.analyze()


def test_undefined_team_reference():
    """Test that references to undefined teams are caught."""
    src = """
    team devs : Command<Search> [size=1];
    flow test_flow(using: devs) {
        checkpoint "start" {
            nonexistent.search("query");
        }
    }
    """
    
    tree = parse(src)
    analyzer = SemanticAnalyzer(tree)
    
    with pytest.raises(SemanticError, match="Unknown team"):
        analyzer.analyze()


def test_strict_typing_mismatch():
    """Test that team type mismatches are caught."""
    src = """
    team devs : Command<Search> [size=1];
    flow test_flow(using: devs) {
        checkpoint "start" {
            devs.judge("code", "quality");
        }
    }
    """
    
    tree = parse(src)
    analyzer = SemanticAnalyzer(tree)
    
    with pytest.raises(SemanticError, match="cannot perform"):
        analyzer.analyze()


def test_all_verb_types_valid():
    """Test that all valid verb-type combinations pass analysis."""
    src = """
    team s : Command<Search> [size=1];
    team t : Command<Try> [size=1];
    team j : Command<Judge> [size=1];
    team c : Command<Communicate> [size=1];
    flow test_flow(using: s, t, j, c) {
        checkpoint "test" {
            s.search("query");
            t.try("experiment");
            j.judge("data", "criteria");
            c.ask("question");
        }
    }
    """
    
    tree = parse(src)
    analyzer = SemanticAnalyzer(tree)
    # Should not raise
    analyzer.analyze()


def test_semantic_ok_example():
    """Test that example1.flow passes semantic analysis."""
    p = Path(__file__).resolve().parents[1] / "examples" / "example1.flow"
    tree = parse(p)
    # should not raise
    SemanticAnalyzer(tree).analyze()


def test_semantic_field_error():
    """Test that accessing non-existent fields on result types raises error."""
    src = (
        'result JudgeResult { confidence: number; score: number; pass: boolean; };\n'
        'type Command<Judge>;\n'
        'team J: Command<Judge> [size=1];\n'
        'flow F(using: J) {\n'
        '  checkpoint "C" {\n'
        '    X = J.judge("t", "crit");\n'
        '    bad = X.no_such_field;\n'
        '  }\n'
        '}\n'
    )
    tree = parse(src)
    with pytest.raises(SemanticError):
        SemanticAnalyzer(tree).analyze()
