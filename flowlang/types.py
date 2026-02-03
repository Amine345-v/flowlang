from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, Optional

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

@dataclass
class TypedValue:
    tag: ValueTag
    meta: Optional[Dict[str, Any]] = None

