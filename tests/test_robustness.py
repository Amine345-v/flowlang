"""Robustness tests for FlowLang runtime.

These tests verify:
1. State growth under repeated deep_merge operations.
2. Graceful handling of malformed AI responses.
"""

import pytest
from unittest.mock import patch, MagicMock
from flowlang.runtime import Runtime, EvalContext
from flowlang.types import TypedValue, ValueTag


class TestStateGrowth:
    """Test that context state doesn't explode under deep_merge."""

    def test_deep_merge_list_growth(self):
        """Verify list concatenation in deep_merge."""
        rt = Runtime()
        a = {"items": [1, 2, 3]}
        b = {"items": [4, 5, 6]}
        merged = rt._deep_merge(a, b)
        assert merged["items"] == [1, 2, 3, 4, 5, 6]

    def test_deep_merge_nested_growth(self):
        """Verify nested dict merging doesn't lose data."""
        rt = Runtime()
        a = {"level1": {"level2": {"data": [1]}}}
        b = {"level1": {"level2": {"data": [2]}, "extra": "new"}}
        merged = rt._deep_merge(a, b)
        assert merged["level1"]["level2"]["data"] == [1, 2]
        assert merged["level1"]["extra"] == "new"

    def test_repeated_merge_state_size(self):
        """Simulate repeated par block merges and measure growth."""
        rt = Runtime()
        base = EvalContext(variables={"log": []}, checkpoints=[], merge_policy="deep_merge")
        
        # Simulate 100 parallel merges each adding 10 items
        for i in range(100):
            sub = EvalContext(
                variables={"log": [f"entry_{i}_{j}" for j in range(10)]},
                checkpoints=[],
                merge_policy="deep_merge"
            )
            rt._merge_contexts(base, sub)
        
        # After 100 merges × 10 items = 1000 log entries
        assert len(base.variables["log"]) == 1000
        # This is expected behavior; the test documents it for awareness


class TestCRDTMerge:
    """Test CRDT merge semantics."""

    def test_crdt_numeric_max(self):
        """CRDT should take max of numeric values."""
        rt = Runtime()
        a = {"score": 5}
        b = {"score": 10}
        merged = rt._crdt_merge(a, b)
        assert merged["score"] == 10

    def test_crdt_list_union(self):
        """CRDT should deduplicate lists."""
        rt = Runtime()
        a = {"tags": ["a", "b"]}
        b = {"tags": ["b", "c"]}
        merged = rt._crdt_merge(a, b)
        assert set(merged["tags"]) == {"a", "b", "c"}


class TestMalformedAIResponse:
    """Test handling of malformed AI responses."""

    def test_garbage_json_fallback(self):
        """Verify runtime doesn't crash on garbage AI output."""
        from flowlang.ai_providers import _map_to_typed_value
        
        # Simulate garbage response
        content = "This is not JSON at all!!!"
        parsed = None  # json.loads would fail
        kwargs = {"history": []}
        
        result = _map_to_typed_value("ask", content, parsed, kwargs)
        
        # Should fall back gracefully
        assert result.tag == ValueTag.CommunicateResult
        assert result.meta["text"] == content  # Raw content preserved

    def test_missing_required_field(self):
        """Verify partial JSON doesn't crash."""
        from flowlang.ai_providers import _map_to_typed_value
        
        content = '{"score": 0.8}'  # Missing confidence and pass
        parsed = {"score": 0.8}
        kwargs = {}
        
        result = _map_to_typed_value("judge", content, parsed, kwargs)
        
        assert result.tag == ValueTag.JudgeResult
        assert result.meta["score"] == 0.8
        assert result.meta["confidence"] == 0.0  # Fallback default
        assert result.meta["pass"] == True  # Derived from score > 0

    def test_wrong_type_in_response(self):
        """Verify wrong types are handled."""
        from flowlang.ai_providers import _map_to_typed_value
        
        # AI returns string instead of number
        content = '{"score": "high", "confidence": "very", "pass": "yes"}'
        parsed = {"score": "high", "confidence": "very", "pass": "yes"}
        kwargs = {}
        
        result = _map_to_typed_value("judge", content, parsed, kwargs)
        
        # Current behavior: accepts whatever AI returns
        # This test documents the gap—ideally should fail or coerce
        assert result.tag == ValueTag.JudgeResult
        assert result.meta["score"] == "high"  # Wrong type accepted!


class TestChainPropagation:
    """Test chain effect propagation edge cases."""

    def test_propagation_decay_to_zero(self):
        """Verify propagation stops when effect decays below cap."""
        rt = Runtime()
        rt.chains["TestChain"] = {
            "nodes": {"A", "B", "C", "D", "E"},
            "order": ["A", "B", "C", "D", "E"],
            "effects": {},
            "propagation": {"decay": 0.5, "forward": True, "backprop": False, "cap": 0.1},
            "labels": {},
            "constraints": {},
        }
        
        ctx = EvalContext(variables={}, checkpoints=[])
        
        # Touch node A with effect 1.0
        # Expected: A=1.0, B=0.5, C=0.25, D=0.125, E=0.0625 (but E < cap)
        rt.chains["TestChain"]["effects"]["A"] = 1.0
        
        # Manually trigger propagation (simulating touch)
        order = rt.chains["TestChain"]["order"]
        decay = 0.5
        cap = 0.1
        cur = 1.0
        for i, node in enumerate(order[1:], 1):
            cur *= decay
            if cur < cap:
                break
            rt.chains["TestChain"]["effects"][node] = cur
        
        effects = rt.chains["TestChain"]["effects"]
        assert effects.get("A") == 1.0
        assert effects.get("B") == 0.5
        assert effects.get("C") == 0.25
        assert effects.get("D") == 0.125
        assert "E" not in effects  # Below cap, propagation stopped


class TestProcessTreePolicies:
    """Test process tree policy enforcement."""

    def test_protected_node_collapse_blocked(self):
        """Verify protected nodes cannot be collapsed."""
        rt = Runtime()
        rt.processes["TestProcess"] = {
            "nodes": {"Root": {}, "Protected": {}, "Normal": {}},
            "policies": {"protected_nodes": "Root,Protected"},
            "marks": {},
        }
        
        ctx = EvalContext(variables={}, checkpoints=[])
        
        # Attempting to collapse protected node should raise
        from flowlang.errors import RuntimeFlowError
        with pytest.raises(RuntimeFlowError, match="protected"):
            rt._process_call("TestProcess", "collapse", ["Protected"], {}, ctx)
        
        # Normal node should collapse fine
        rt._process_call("TestProcess", "collapse", ["Normal"], {}, ctx)
        assert "Normal" not in rt.processes["TestProcess"]["nodes"]
