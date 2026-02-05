"""Tests for schema validation in FlowLang.

These tests verify that AI responses are strictly validated
and that invalid responses raise SchemaValidationError.
"""

import pytest
from flowlang.schemas import (
    validate_response,
    JudgeResult,
    SearchResult,
    TryResult,
    CommunicateResult,
)
from flowlang.errors import SchemaValidationError
from flowlang.ai_providers import _map_to_typed_value
from flowlang.types import ValueTag


class TestJudgeResultValidation:
    """Test JudgeResult schema validation."""

    def test_valid_judge_response(self):
        """Valid response should pass validation."""
        data = {"score": 0.85, "confidence": 0.9, "pass": True}
        result = validate_response("judge", data)
        assert isinstance(result, JudgeResult)
        assert result.score == 0.85
        assert result.confidence == 0.9
        assert result.pass_result is True

    def test_string_numbers_coerced(self):
        """String numbers should be coerced to float."""
        data = {"score": "0.75", "confidence": "0.8", "pass": "true"}
        result = validate_response("judge", data)
        assert result.score == 0.75
        assert result.confidence == 0.8
        assert result.pass_result is True

    def test_score_out_of_range_fails(self):
        """Score > 1 should fail validation."""
        data = {"score": 1.5, "confidence": 0.5, "pass": True}
        with pytest.raises(SchemaValidationError, match="score"):
            validate_response("judge", data)

    def test_invalid_string_fails(self):
        """Non-numeric string should fail."""
        data = {"score": "high", "confidence": 0.5, "pass": True}
        with pytest.raises(SchemaValidationError, match="Cannot convert"):
            validate_response("judge", data)


class TestSearchResultValidation:
    """Test SearchResult schema validation."""

    def test_valid_search_response(self):
        """Valid response should pass."""
        data = {"hits": ["result1", "result2", "result3"]}
        result = validate_response("search", data)
        assert isinstance(result, SearchResult)
        assert len(result.hits) == 3

    def test_single_string_coerced_to_list(self):
        """Single string should become a list."""
        data = {"hits": "single result"}
        result = validate_response("search", data)
        assert result.hits == ["single result"]

    def test_none_becomes_empty_list(self):
        """None should become empty list."""
        data = {"hits": None}
        result = validate_response("search", data)
        assert result.hits == []


class TestTryResultValidation:
    """Test TryResult schema validation."""

    def test_valid_try_response(self):
        """Valid response should pass."""
        data = {"output": "experiment completed", "metrics": {"time": 1.5}}
        result = validate_response("try", data)
        assert isinstance(result, TryResult)
        assert result.output == "experiment completed"
        assert result.metrics["time"] == 1.5

    def test_missing_metrics_defaults(self):
        """Missing metrics should default to empty dict."""
        data = {"output": "result"}
        result = validate_response("try", data)
        assert result.metrics == {}


class TestCommunicateResultValidation:
    """Test CommunicateResult schema validation."""

    def test_valid_ask_response(self):
        """Valid response should pass."""
        data = {"text": "Hello!", "history": ["Hi", "Hello!"]}
        result = validate_response("ask", data)
        assert isinstance(result, CommunicateResult)
        assert result.text == "Hello!"
        assert len(result.history) == 2

    def test_missing_history_defaults(self):
        """Missing history should default to empty list."""
        data = {"text": "response"}
        result = validate_response("ask", data)
        assert result.history == []


class TestMapToTypedValue:
    """Test the updated _map_to_typed_value function."""

    def test_valid_judge_creates_typed_value(self):
        """Valid judge response creates correct TypedValue."""
        content = '{"score": 0.9, "confidence": 0.85, "pass": true}'
        parsed = {"score": 0.9, "confidence": 0.85, "pass": True}
        result = _map_to_typed_value("judge", content, parsed, {})
        
        assert result.tag == ValueTag.JudgeResult
        assert result.meta["score"] == 0.9
        assert result.meta["confidence"] == 0.85
        assert result.meta["pass"] is True

    def test_invalid_judge_raises_error(self):
        """Invalid judge response should raise SchemaValidationError."""
        content = '{"score": "invalid"}'
        parsed = {"score": "invalid"}
        
        with pytest.raises(SchemaValidationError):
            _map_to_typed_value("judge", content, parsed, {})

    def test_raw_content_judge_fails(self):
        """Raw content (no JSON) for judge should fail."""
        content = "This is not JSON"
        parsed = None
        
        with pytest.raises(SchemaValidationError, match="must be valid JSON"):
            _map_to_typed_value("judge", content, parsed, {})

    def test_raw_content_ask_succeeds(self):
        """Raw content for ask should succeed (text fallback)."""
        content = "Plain text response"
        parsed = None
        
        result = _map_to_typed_value("ask", content, parsed, {"history": ["q1"]})
        assert result.tag == ValueTag.CommunicateResult
        assert result.meta["text"] == "Plain text response"

    def test_search_with_valid_data(self):
        """Search with valid data should work."""
        content = '{"hits": ["a", "b"]}'
        parsed = {"hits": ["a", "b"]}
        
        result = _map_to_typed_value("search", content, parsed, {})
        assert result.tag == ValueTag.SearchResult
        assert result.meta["hits"] == ["a", "b"]
