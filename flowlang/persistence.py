import os
import json
import pickle
import glob
from typing import Any, Dict, Optional, List
from datetime import datetime

try:
    from pydantic import BaseModel, Field
    _HAS_PYDANTIC = True
except ImportError:
    _HAS_PYDANTIC = False
    from dataclasses import dataclass, field, asdict

# ─── FlowState: Validated execution snapshot ─────────────────────
if _HAS_PYDANTIC:
    class FlowState(BaseModel):
        """Serializable snapshot of a FlowLang execution state (Pydantic-validated)."""
        flow_name: str
        timestamp: str
        eval_context: Dict[str, Any] = Field(default_factory=dict)
        checkpoints: List[str] = Field(default_factory=list)
        chains: Dict[str, Any] = Field(default_factory=dict)
        processes: Dict[str, Any] = Field(default_factory=dict)
        pc: int = 0

        def to_dict(self) -> Dict[str, Any]:
            return self.model_dump()

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'FlowState':
            return cls.model_validate(data)
else:
    @dataclass
    class FlowState:
        """Serializable snapshot of a FlowLang execution state (fallback)."""
        flow_name: str
        timestamp: str
        eval_context: Dict[str, Any] = field(default_factory=dict)
        checkpoints: List[str] = field(default_factory=list)
        chains: Dict[str, Any] = field(default_factory=dict)
        processes: Dict[str, Any] = field(default_factory=dict)
        pc: int = 0

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'FlowState':
            return cls(**data)

        def to_dict(self) -> Dict[str, Any]:
            return asdict(self)


class PersistenceManager:
    """Manages saving and loading FlowLang state."""
    
    def __init__(self, base_path: str = "./.flowlang_state"):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)
        
    def save_state(self, name: str, ctx_vars: Dict[str, Any], checkpoints: List[str], chains: Dict[str, Any] = None, processes: Dict[str, Any] = None) -> str:
        """Save state to disk. Returns the filename."""
        safe_name = "".join(c for c in name if c.isalnum() or c in ('-', '_'))
        timestamp = datetime.now().isoformat().replace(":", "-")
        filename = f"{safe_name}_{timestamp}.json"
        path = os.path.join(self.base_path, filename)
        
        state = FlowState(
            flow_name=name,
            timestamp=timestamp,
            eval_context=ctx_vars,
            checkpoints=checkpoints,
            chains=chains or {},
            processes=processes or {}
        )
        
        pkl_path = path.replace(".json", ".pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(state, f)
            
        return pkl_path

    def load_state(self, path: str) -> FlowState:
        """Load state from a file path."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"State file not found: {path}")
            
        with open(path, "rb") as f:
            state = pickle.load(f)
            
        if not isinstance(state, FlowState):
             raise ValueError("File content is not a valid FlowState")
             
        return state

    def list_states(self, name_filter: str = "*") -> List[str]:
        """List available state files, sorted by newest first."""
        pattern = os.path.join(self.base_path, f"{name_filter}*.pkl")
        files = glob.glob(pattern)
        files.sort(key=os.path.getmtime, reverse=True)
        return files

    def get_latest_state(self, name: str) -> Optional[str]:
        """Get the most recent state file for a given flow name."""
        files = self.list_states(name)
        return files[0] if files else None

    def export_map_json(self, name: str, state: FlowState) -> str:
        """Exports a simplified JSON map for the JOL-IDE to visualize the System Tree."""
        export_path = os.getenv("FLOWLANG_IDE_EXPORT_PATH", self.base_path)
        os.makedirs(export_path, exist_ok=True)
        
        system_map = {
            "project": name,
            "timestamp": state.timestamp,
            "chains": state.chains,
            "processes": state.processes,
            "traces": []
        }
        
        def _serialize_impact(impact):
            return impact.value if hasattr(impact, 'value') else str(impact)
        
        for var_name, var_val in state.eval_context.items():
            if hasattr(var_val, "critical_features"):
                for cf in var_val.critical_features:
                    system_map["traces"].append({
                        "name": cf.name,
                        "value": cf.value,
                        "impact": _serialize_impact(cf.impact),
                        "origin": cf.origin_node
                    })
            elif isinstance(var_val, list):
                for item in var_val:
                    if hasattr(item, "critical_features"):
                        for cf in item.critical_features:
                            system_map["traces"].append({
                                "name": cf.name,
                                "value": cf.value,
                                "impact": _serialize_impact(cf.impact),
                                "origin": cf.origin_node
                            })

        filename = f"{name}_map.json"
        path = os.path.join(export_path, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(system_map, f, indent=2, ensure_ascii=False)
            
        return path

