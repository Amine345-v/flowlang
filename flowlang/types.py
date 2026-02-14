from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Optional, List, Set
from datetime import datetime

try:
    from pydantic import BaseModel, Field, field_validator, ValidationError
    _HAS_PYDANTIC = True
except ImportError:
    _HAS_PYDANTIC = False

class CommandKind(str, Enum):
    Search = "Search"
    Try = "Try"
    Judge = "Judge"
    Communicate = "Communicate"

@dataclass
class TeamType:
    name: str
    kind: CommandKind
    options: Dict[str, Any]

# Simple value typing for semantic passes
class ValueTag(str, Enum):
    Number = "Number"
    String = "String"
    Boolean = "Boolean"
    List = "List"
    Dict = "Dict"
    Unknown = "Unknown"
    SearchResult = "SearchResult"
    TryResult = "TryResult"
    JudgeResult = "JudgeResult"  # has .confidence
    CommunicateResult = "CommunicateResult"
    Option = "Option"
    Union = "Union"
    Order = "Order"

@dataclass
class TypedValue:
    tag: ValueTag
    value: Any = None
    meta: Optional[Dict[str, Any]] = None

# ─── CFP Strict Enums ───────────────────────────────────────────
class EchoSignature(str, Enum):
    """How far the ripple goes if this feature is modified."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class ImpactKind(str, Enum):
    """The constitutional weight of a feature."""
    GUIDANCE = "guidance"
    CONSTRAINT = "constraint"
    REQUIREMENT = "requirement"

# ─── CriticalFeature: The Constitutional Lock ───────────────────
if _HAS_PYDANTIC:
    class CriticalFeature(BaseModel, frozen=True):
        """CFP-validated trace (أثر إلزامي). Once created, cannot be mutated.
        This is the 'Constitutional Lock' — no AI output enters the System Tree
        without passing Pydantic validation."""
        name: str = Field(min_length=1, description="Feature name, cannot be empty")
        value: Any = Field(description="The trace value")
        feature_id: Optional[str] = None
        ancestry_link: Optional[str] = None
        feature_type: Optional[str] = None
        impact_zones: frozenset = Field(default_factory=frozenset)
        echo_signature: EchoSignature = EchoSignature.LOW
        confidence: float = Field(ge=0.0, le=1.0, default=1.0)
        impact: ImpactKind = ImpactKind.GUIDANCE
        origin_node: Optional[str] = None

        @field_validator("impact_zones", mode="before")
        @classmethod
        def coerce_impact_zones(cls, v):
            """Accept list/set/frozenset and normalize to frozenset."""
            if isinstance(v, (list, set)):
                return frozenset(v)
            return v
        
        @field_validator("echo_signature", mode="before")
        @classmethod 
        def coerce_echo_signature(cls, v):
            """Accept string and normalize to EchoSignature enum."""
            if isinstance(v, str):
                return EchoSignature(v.upper())
            return v

        @field_validator("impact", mode="before")
        @classmethod
        def coerce_impact(cls, v):
            """Accept string and normalize to ImpactKind enum."""
            if isinstance(v, str):
                try:
                    return ImpactKind(v.lower())
                except ValueError:
                    return ImpactKind.GUIDANCE
            return v
else:
    # Fallback: plain dataclass if pydantic is not available
    @dataclass(frozen=True)
    class CriticalFeature:
        """CFP trace (fallback without Pydantic validation)."""
        name: str = ""
        value: Any = None
        feature_id: Optional[str] = None
        ancestry_link: Optional[str] = None
        feature_type: Optional[str] = None
        impact_zones: frozenset = field(default_factory=frozenset)
        echo_signature: str = "LOW"
        confidence: float = 1.0
        impact: str = "guidance"
        origin_node: Optional[str] = None

# ─── Helper: Parse a dict into a CriticalFeature safely ─────────
def parse_critical_feature(raw: dict, fallback_origin: Optional[str] = None) -> Optional[CriticalFeature]:
    """The 'Sieve' (المنخل): validates raw dict and returns a CriticalFeature or None."""
    try:
        # Normalize keys
        data = {
            "name": raw.get("name", "unknown"),
            "value": raw.get("value"),
            "feature_id": raw.get("feature_id"),
            "ancestry_link": raw.get("ancestry_link") or fallback_origin,
            "feature_type": raw.get("feature_type") or raw.get("type"),
            "impact_zones": raw.get("impact_zones", []),
            "echo_signature": raw.get("echo_signature", "LOW"),
            "confidence": raw.get("confidence", 1.0),
            "impact": raw.get("impact", "guidance"),
            "origin_node": fallback_origin,
        }
        if _HAS_PYDANTIC:
            return CriticalFeature.model_validate(data)
        else:
            return CriticalFeature(**data)
    except Exception:
        # Rejected by the Constitutional Lock — this trace is invalid
        return None

# ─── Order: Mutable work unit with lifecycle ─────────────────────
@dataclass
class Order:
    """Represents a discrete unit of work (an Order) in the system.
    Orders are mutable because they have lifecycle states (created → processing → completed)."""
    id: str
    payload: Any
    kind: CommandKind
    state: str = "created"  # created, processing, completed, failed
    history: List[str] = field(default_factory=list) # Monolith dialogue history
    chain_node: Optional[str] = None # Associated chain node (System Sequence)
    process_node: Optional[str] = None # Associated process node (The Maestro)
    audit_trail: List[Dict[str, Any]] = field(default_factory=list) # Who touched it?
    critical_features: List[CriticalFeature] = field(default_factory=list) # Commanding Traces

    def log_activity(self, team: str, action: str, result: Any, member_idx: Optional[int] = None):
        if isinstance(result, dict) and "critical_features" in result:
            features = result["critical_features"]
            if isinstance(features, list):
                for f in features:
                    if isinstance(f, dict):
                        validated = parse_critical_feature(f, fallback_origin=self.chain_node)
                        if validated is not None:
                            self.critical_features.append(validated)
                        # else: silently rejected by the Constitutional Lock

        self.audit_trail.append({
            "timestamp": datetime.now().isoformat(),
            "team": team,
            "team_member": member_idx,
            "action": action,
            "result_summary": str(result)[:100]
        })
        # "Load" cycle: Transition back to processing if it was completed or created
        self.state = "processing"

    def complete(self, result: Any = None):
        # "Unload" cycle: Transition to completed
        self.state = "completed"
        if result:
            self.audit_trail.append({
                "timestamp": datetime.now().isoformat(),
                "status": "completed",
                "final_result": str(result)[:200]
            })

    def fail(self, error: str):
        self.state = "failed"
        self.audit_trail.append({
            "timestamp": datetime.now().isoformat(),
            "status": "failed",
            "error": error
        })
