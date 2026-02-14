"""
SystemTreeEngine: DAG-enforced graph engine for FlowLang's System Tree.

This module provides proper graph data structures for:
- Ancestry verification (فحص النسب)
- Echo propagation (صدى التعديل)
- Cycle detection (اكتشاف الحلقات)
- Tree completion checks (ملأ الشجرة)

The System Tree is a Directed Acyclic Graph (DAG). Any attempt to create
a cycle is rejected — ensuring structural integrity at the mathematical level.
"""

from typing import Any, Dict, List, Optional, Set

try:
    import networkx as nx
    _HAS_NETWORKX = True
except ImportError:
    _HAS_NETWORKX = False

try:
    from .types import CriticalFeature
except ImportError:
    CriticalFeature = None

try:
    from flowlang_core import EchoEngine
    _HAS_RUST_ENGINE = True
except ImportError:
    _HAS_RUST_ENGINE = False


class SystemTreeEngine:
    """DAG-enforced System Tree with echo propagation and ancestry verification.
    
    This is the 'Map' (الخريطة) that the user sees in the IDE.
    Every node is a validated CriticalFeature, and every edge is a causal link.
    """
    
    def __init__(self):
        if not _HAS_NETWORKX:
            raise ImportError("NetworkX is required for the System Tree Engine. Install: pip install networkx>=3.0")
        self.graph: nx.DiGraph = nx.DiGraph()
        self._mandatory_features: Set[str] = set()
        
        # Rust Acceleration (The "Muscle")
        self.rust_engine = EchoEngine() if _HAS_RUST_ENGINE else None
    
    # ─── Node Management ─────────────────────────────────────────
    
    def add_trace(self, feature: Any) -> bool:
        """Add a validated trace to the tree. Returns False if rejected (cycle or duplicate)."""
        fid = getattr(feature, 'feature_id', None) or getattr(feature, 'name', str(id(feature)))
        
        # Reject duplicate node IDs to prevent silent overwrites
        if fid in self.graph.nodes:
            return False
        
        # Add the node
        self.graph.add_node(fid, data=feature)
        if self.rust_engine:
            # PHYSICS OF LOGIC: Calculate Mass based on Echo Signature
            # LOW -> High Mass (Damps volatility)
            # CRITICAL -> Low Mass (Transmits efficiently)
            sig = str(getattr(feature, 'echo_signature', 'MEDIUM'))
            mass = 1.0
            if sig == 'LOW': mass = 10.0
            elif sig == 'MEDIUM': mass = 5.0
            elif sig == 'HIGH': mass = 1.0
            elif sig == 'CRITICAL': mass = 0.1
            
            self.rust_engine.add_node(fid, mass)
        
        # Add the ancestry edge
        ancestry = getattr(feature, 'ancestry_link', None)
        if ancestry and ancestry in self.graph.nodes:
            self.graph.add_edge(ancestry, fid)
            
            # DAG enforcement: reject cycles
            if not nx.is_directed_acyclic_graph(self.graph):
                self.graph.remove_edge(ancestry, fid)
                self.graph.remove_node(fid)
                if self.rust_engine:
                    # Sync rollback (Rust assumes adding edge might fail, but here we manually rollback)
                    # Actually Rust engine has its own cycle check on add_edge.
                    # We should probably trust Rust engine more for perf?
                    # For now just sync removal.
                    self.rust_engine.remove_node(fid)
                return False
            
            # Sync edge to Rust if Python check passed
            if self.rust_engine:
                # Rust add_edge returns False if cycle, but we already checked in NetworkX
                self.rust_engine.add_edge(ancestry, fid)
        
        return True
    
    def remove_trace(self, feature_id: str):
        """Remove a trace and all its descendants."""
        if feature_id in self.graph.nodes:
            descendants = list(nx.descendants(self.graph, feature_id))
            self.graph.remove_nodes_from([feature_id] + descendants)
            
            if self.rust_engine:
                self.rust_engine.remove_node(feature_id)
                # Rust engine removes connected edges, but descendants needed?
                # Rust remove_node removes edges but doesn't cascade delete descendants automatically.
                # We need to manually remove descendants from Rust too.
                for d in descendants:
                    self.rust_engine.remove_node(d)
    
    # ─── Echo Propagation ────────────────────────────────────────
    
    def get_echo_path(self, feature_id: str) -> List[str]:
        """Get all downstream nodes affected by a change at this node.
        This is the 'Causal Echo' — the ripple that flows forward."""
        if self.rust_engine:
             return self.rust_engine.get_descendants(feature_id)
             
        if feature_id not in self.graph.nodes:
            return []
        return list(nx.descendants(self.graph, feature_id))
    
    def get_reverse_echo(self, feature_id: str) -> List[str]:
        """Get all upstream nodes that feed into this node."""
        if feature_id not in self.graph.nodes:
            return []
        return list(nx.ancestors(self.graph, feature_id))
    
    # ─── Ancestry Verification (for Judge Engine) ────────────────
    
    def verify_ancestry(self, feature_id: str, claimed_parent: str) -> bool:
        """Formal check: is claimed_parent actually an ancestor of feature_id?
        This is the Judge's 'Ancestry Check' (فحص النسب)."""
        if feature_id not in self.graph.nodes:
            return False
        if claimed_parent not in self.graph.nodes:
            return False
        return self.graph.has_edge(claimed_parent, feature_id) or \
               claimed_parent in nx.ancestors(self.graph, feature_id)
    
    def get_causal_chain(self, feature_id: str) -> List[str]:
        """Get the full causal chain from ROOT to this feature.
        This is what the user sees when they click a node in the IDE."""
        if feature_id not in self.graph.nodes:
            return []
        # Find shortest path from any root to this node
        roots = [n for n in self.graph.nodes if self.graph.in_degree(n) == 0]
        for root in roots:
            try:
                path = nx.shortest_path(self.graph, root, feature_id)
                return path
            except nx.NetworkXNoPath:
                continue
        return [feature_id]
    
    # ─── Tree Completion (for Judge Engine) ──────────────────────
    
    def set_mandatory_features(self, features: List[str]):
        """Define which features MUST exist for the tree to be 'complete'."""
        self._mandatory_features = set(features)
    
    def get_missing_mandatory(self) -> List[str]:
        """Check which mandatory features are missing from the tree."""
        existing_names = set()
        for nid in self.graph.nodes:
            data = self.graph.nodes[nid].get("data")
            if data:
                name = getattr(data, 'name', nid)
                existing_names.add(name)
            existing_names.add(nid)
        return [m for m in self._mandatory_features if m not in existing_names]
    
    # ─── Visualization / Export ──────────────────────────────────
    
    def to_dict(self) -> Dict[str, Any]:
        """Export the tree as a dict for IDE visualization."""
        nodes = []
        for nid in self.graph.nodes:
            data = self.graph.nodes[nid].get("data")
            node_info = {
                "id": nid,
                "parents": list(self.graph.predecessors(nid)),
                "children": list(self.graph.successors(nid)),
                "depth": len(list(nx.ancestors(self.graph, nid))),
            }
            if data:
                node_info["name"] = getattr(data, 'name', nid)
                node_info["value"] = getattr(data, 'value', None)
                node_info["echo_signature"] = str(getattr(data, 'echo_signature', 'LOW'))
            nodes.append(node_info)
        
        return {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "is_dag": nx.is_directed_acyclic_graph(self.graph),
            "nodes": nodes
        }
    
    @property
    def node_count(self) -> int:
        return self.graph.number_of_nodes()
    
    @property 
    def is_valid_dag(self) -> bool:
        """Check if the graph is a Directed Acyclic Graph (DAG)."""
        return nx.is_directed_acyclic_graph(self.graph)
