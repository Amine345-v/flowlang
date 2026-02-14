"""
Unit tests for the FlowLang parser.
Tests the parse() function and ParseError from flowlang.parser.
"""
import pytest
from pathlib import Path

from flowlang.parser import parse, ParseError


def test_parse_simple_flow():
    """Test parsing a simple flow definition."""
    flow_src = """
    team dev_team : Command<Search> [size=1];
    flow simple_flow(using: dev_team) {
        checkpoint "start" {
            dev_team.search("query");
        }
    }
    """
    
    tree = parse(flow_src)
    assert tree is not None
    assert tree.data == "program"
    
    # Find flow declaration
    flows = list(tree.find_data("flow_decl"))
    assert len(flows) == 1
    assert str(flows[0].children[0]) == "simple_flow"
    
    # Find checkpoints
    checkpoints = list(tree.find_data("checkpoint"))
    assert len(checkpoints) == 1
    
    # Find actions
    actions = list(tree.find_data("action_stmt"))
    assert len(actions) == 1
    assert str(actions[0].children[0]) == "dev_team"


def test_parse_complex_flow():
    """Test parsing a more complex flow with multiple checkpoints and actions."""
    flow_src = """
    team dev : Command<Search> [size=2];
    team qa : Command<Judge> [size=1];
    team ops : Command<Try> [size=1];
    flow complex_flow(using: dev, qa, ops) {
        checkpoint "develop" {
            dev.search("requirements");
            qa.judge("code_quality", "high");
        }
        
        checkpoint "deploy" {
            ops.try("staging deployment");
            qa.judge("staging_test", "pass");
        }
        
        checkpoint "release" {
            ops.try("production deployment");
        }
    }
    """
    
    tree = parse(flow_src)
    
    flows = list(tree.find_data("flow_decl"))
    assert len(flows) == 1
    assert str(flows[0].children[0]) == "complex_flow"
    
    checkpoints = list(tree.find_data("checkpoint"))
    assert len(checkpoints) == 3


def test_parse_errors():
    """Test parsing invalid flow definitions raises ParseError."""
    # Missing flow closing brace
    with pytest.raises(ParseError):
        parse("flow test_flow(using: T) {")
    
    # Invalid syntax
    with pytest.raises(ParseError):
        parse("this is not valid flowlang")
    
    # Empty team with no type
    with pytest.raises(ParseError):
        parse("team ;")


def test_parse_chain_definition():
    """Test parsing chain declarations."""
    src = """
    chain deployment_flow {
        nodes: [build, test, deploy];
        propagation: causal(decay=0.5, backprop=true, forward=true);
    }
    """
    
    tree = parse(src)
    chains = list(tree.find_data("chain_decl"))
    assert len(chains) == 1
    assert str(chains[0].children[0]) == "deployment_flow"


def test_parse_process_definition():
    """Test parsing process declarations with policies."""
    src = """
    process deployment "Deploy Flow" {
        root: "Root";
        branch "Pipeline" -> ["Build", "Test", "Deploy"];
        node "Build" { type: "ci"; };
        policy: { require_reason: true; };
        audit: enabled;
    }
    """
    
    tree = parse(src)
    processes = list(tree.find_data("process_decl"))
    assert len(processes) == 1
    assert str(processes[0].children[0]) == "deployment"


def test_parse_result_type():
    """Test parsing result type declarations."""
    src = """
    result SearchResult {
        items: list;
        total: number;
    };
    """
    
    tree = parse(src)
    results = list(tree.find_data("result_decl"))
    assert len(results) == 1
    assert str(results[0].children[0]) == "SearchResult"


def test_parse_team_with_options():
    """Test parsing team declarations with various options."""
    src = """
    team searchers : Command<Search> [size=3, distribution=round_robin];
    """
    
    tree = parse(src)
    teams = list(tree.find_data("team_decl"))
    assert len(teams) == 1
    assert str(teams[0].children[0]) == "searchers"
