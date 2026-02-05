"""Pydantic schemas for FlowLang AI response validation.

All AI provider responses MUST validate against these schemas.
Invalid responses raise SchemaValidationError.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict


class JudgeResult(BaseModel):
    """Schema for judge verb responses."""
    model_config = ConfigDict(populate_by_name=True)
    
    score: float = Field(ge=0.0, le=1.0, description="Score between 0 and 1")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence between 0 and 1")
    pass_result: bool = Field(alias="pass", description="Whether the judgment passed")
    
    @field_validator('score', 'confidence', mode='before')
    @classmethod
    def coerce_to_float(cls, v: Any) -> float:
        """Attempt to coerce string numbers to float."""
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                raise ValueError(f"Cannot convert '{v}' to float")
        return float(v)
    
    @field_validator('pass_result', mode='before')
    @classmethod
    def coerce_to_bool(cls, v: Any) -> bool:
        """Coerce various truthy values to bool."""
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ('true', 'yes', '1', 'pass', 'passed')
        return bool(v)


class SearchResult(BaseModel):
    """Schema for search verb responses."""
    hits: List[str] = Field(default_factory=list, description="List of search results")
    
    @field_validator('hits', mode='before')
    @classmethod
    def ensure_list(cls, v: Any) -> List[str]:
        """Ensure hits is a list of strings."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return [str(item) for item in v]
        return [str(v)]


class TryResult(BaseModel):
    """Schema for try verb responses."""
    output: str = Field(default="", description="Output of the experiment")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Metrics dictionary")
    
    @field_validator('output', mode='before')
    @classmethod
    def coerce_to_str(cls, v: Any) -> str:
        """Coerce to string."""
        if v is None:
            return ""
        return str(v)
    
    @field_validator('metrics', mode='before')
    @classmethod
    def ensure_dict(cls, v: Any) -> Dict[str, Any]:
        """Ensure metrics is a dict."""
        if v is None:
            return {}
        if isinstance(v, dict):
            return v
        return {"value": v}


class CommunicateResult(BaseModel):
    """Schema for ask/communicate verb responses."""
    text: str = Field(default="", description="Response text")
    history: List[str] = Field(default_factory=list, description="Conversation history")
    
    @field_validator('text', mode='before')
    @classmethod
    def coerce_to_str(cls, v: Any) -> str:
        """Coerce to string."""
        if v is None:
            return ""
        return str(v)
    
    @field_validator('history', mode='before')
    @classmethod
    def ensure_list(cls, v: Any) -> List[str]:
        """Ensure history is a list."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return [str(item) for item in v]
        return []


# Schema registry for verb -> schema mapping
VERB_SCHEMAS = {
    "ask": CommunicateResult,
    "search": SearchResult,
    "try": TryResult,
    "judge": JudgeResult,
}


def validate_response(verb: str, data: Dict[str, Any]) -> BaseModel:
    """Validate AI response against the appropriate schema.
    
    Args:
        verb: The verb that was executed (ask, search, try, judge)
        data: The parsed JSON response from the AI
        
    Returns:
        Validated Pydantic model instance
        
    Raises:
        SchemaValidationError: If validation fails
    """
    from .errors import SchemaValidationError
    
    schema_cls = VERB_SCHEMAS.get(verb)
    if schema_cls is None:
        # Unknown verb, return raw data wrapped in a generic model
        return TryResult(output=str(data), metrics={})
    
    try:
        return schema_cls.model_validate(data)
    except Exception as e:
        raise SchemaValidationError(
            f"AI response for '{verb}' failed schema validation: {e}\n"
            f"Received data: {data}"
        )
