from pathlib import Path
import pytest

from flowlang.parser import parse


def test_parse_example_file():
    p = Path(__file__).resolve().parents[1] / "examples" / "example1.flow"
    tree = parse(p)
    assert tree is not None


def test_parse_min_program():
    src = (
        'result R { x: number; };\n'
        'type Command<Search>;\n'
        'team T: Command<Search> [size=1];\n'
        'chain C { nodes: [A,B]; propagation: causal(decay=0.5, backprop=true, forward=true); };\n'
        'process P "X" { root: "R"; };\n'
        'flow F(using: T) { checkpoint "C1" { _ = T.search("q"); } }\n'
    )
    tree = parse(src)
    assert tree is not None
