# Minimal AST wrapper types for FlowLang prototype
from dataclasses import dataclass
from typing import Any, List, Dict, Optional

@dataclass
class Program:
    tree: Any  # Keep original Lark tree for simplicity

# Execution context structures
@dataclass
class TeamSpec:
    name: str
    kind: str  # Search | Try | Judge | Communicate
    options: Dict[str, Any]

@dataclass
class ChainSpec:
    name: str
    nodes: List[str]

@dataclass
class FlowSpec:
    name: str
    params: List[str]  # team names

@dataclass
class RuntimeContext:
    variables: Dict[str, Any]
    checkpoint_reports: List[Any]

