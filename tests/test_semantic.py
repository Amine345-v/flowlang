import pytest
from pathlib import Path
from flowlang.parser import parse
from flowlang.semantic import SemanticAnalyzer, SemanticError


def test_semantic_ok_example():
    p = Path(__file__).resolve().parents[1] / "examples" / "example1.flow"
    tree = parse(p)
    # should not raise
    SemanticAnalyzer(tree).analyze()


def test_semantic_field_error():
    # wrong field on JudgeResult
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
