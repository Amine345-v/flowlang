from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
from datetime import datetime

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

@dataclass
class Order:
    """Represents a discrete unit of work (an Order) in the system."""
    id: str
    payload: Any
    kind: CommandKind
    state: str = "created"  # created, processing, completed, failed
    history: List[str] = field(default_factory=list) # Monolith dialogue history
    chain_node: Optional[str] = None # Associated chain node (System Sequence)
    process_node: Optional[str] = None # Associated process node (The Maestro)
    audit_trail: List[Dict[str, Any]] = field(default_factory=list) # Who touched it?
    
    def log_activity(self, team: str, action: str, result: Any, member_idx: Optional[int] = None):
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
