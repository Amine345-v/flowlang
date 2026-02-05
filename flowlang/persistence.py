import os
import json
import pickle
import glob
from typing import Any, Dict, Optional, List
from datetime import datetime
from dataclasses import dataclass, field, asdict

@dataclass
class FlowState:
    """Serializable snapshot of a FlowLang execution state."""
    flow_name: str
    timestamp: str
    eval_context: Dict[str, Any]
    checkpoints: List[str]
    pc: int = 0  # Program counter or instruction index if applicable
    
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
        
    def save_state(self, name: str, ctx_vars: Dict[str, Any], checkpoints: List[str]) -> str:
        """Save state to disk. Returns the filename."""
        # Sanitize name
        safe_name = "".join(c for c in name if c.isalnum() or c in ('-', '_'))
        timestamp = datetime.now().isoformat().replace(":", "-")
        filename = f"{safe_name}_{timestamp}.json"
        path = os.path.join(self.base_path, filename)
        
        # Create state object
        state = FlowState(
            flow_name=name,
            timestamp=timestamp,
            eval_context=ctx_vars,  # Assumes ctx_vars is JSON-serializable. If not, we might need pickle.
            checkpoints=checkpoints
        )
        
        # Try JSON first, fallback to pickle if needed? 
        # Requirement says JSON/pickle. Let's use JSON for readability, maybe fallback or custom encoder.
        # But EvalContext might contain complex objects (Result objects are pydantic or dicts).
        # We'll use a robust JSON encoder or just pickle for simplicity and correctness of python objects.
        # Let's use pickle for reliability of internal state.
        
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
