from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union

# -------- FlowLang Intermediate Representation (FlowIR) ---------

@dataclass
class ActionOp:
    team: str
    verb: str
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ContextUpdateOp:
    values: List[Any] = field(default_factory=list)

@dataclass
class ChainTouchOp:
    chain: str
    node: str
    effect: Any

@dataclass
class SystemCallOp:
    target: str
    op: str
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DeployOp:
    model: str
    env: str

@dataclass
class FlowControlOp:
    kind: str  # 'back_to' or 'end'
    target: Optional[str] = None

@dataclass
class ParBlockOp:
    branches: List[List['OpIR']] = field(default_factory=list)

@dataclass
class RaceBlockOp:
    branches: List[List['OpIR']] = field(default_factory=list)

OpIR = Union[ActionOp, ContextUpdateOp, ChainTouchOp, SystemCallOp, DeployOp, FlowControlOp, ParBlockOp, RaceBlockOp]

@dataclass
class CheckpointIR:
    name: str
    ops: List[OpIR] = field(default_factory=list)

@dataclass
class FlowIR:
    name: str
    teams: List[str] = field(default_factory=list)
    merge_policy: str = "last_wins"
    checkpoints: List[CheckpointIR] = field(default_factory=list)

@dataclass
class ProgramIR:
    flows: List[FlowIR] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
