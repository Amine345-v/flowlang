use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::algo::is_cyclic_directed;
use petgraph::Direction;
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Clone, Debug)]
pub struct NodeData {
    pub name: String,
    pub mass: f64,
}

/// Core propagation engine backed by petgraph DiGraph.
/// Now with "Physics of Logic": Mass-based damping.
pub struct PropagationEngine {
    graph: DiGraph<NodeData, ()>,
    /// Fast lookup: node name → graph index
    name_to_idx: HashMap<String, NodeIndex>,
}

impl PropagationEngine {
    pub fn new() -> Self {
        PropagationEngine {
            graph: DiGraph::new(),
            name_to_idx: HashMap::new(),
        }
    }

    /// Add a node with mass. Returns true if new, false if already exists.
    pub fn add_node(&mut self, name: String, mass: f64) -> bool {
        if self.name_to_idx.contains_key(&name) {
            // Update mass if it exists? For now, immutable identity.
            return false;
        }
        let data = NodeData { name: name.clone(), mass };
        let idx = self.graph.add_node(data);
        self.name_to_idx.insert(name, idx);
        true
    }

    /// Add a directed edge. Returns false if it would create a cycle (DAG enforcement).
    /// Note: Nodes must be created via add_node first, or this will create them with default mass (1.0).
    pub fn add_edge(&mut self, from: String, to: String) -> bool {
        // Ensure both nodes exist with default mass 1.0 (High/Medium) if not already present
        if !self.name_to_idx.contains_key(&from) {
            self.add_node(from.clone(), 1.0);
        }
        if !self.name_to_idx.contains_key(&to) {
            self.add_node(to.clone(), 1.0);
        }

        let from_idx = self.name_to_idx[&from];
        let to_idx = self.name_to_idx[&to];

        // Add the edge
        self.graph.add_edge(from_idx, to_idx, ());

        // Check if this created a cycle
        if is_cyclic_directed(&self.graph) {
            if let Some(edge) = self.graph.find_edge(from_idx, to_idx) {
                self.graph.remove_edge(edge);
            }
            return false;
        }

        true
    }

    /// Check if the graph is a valid DAG.
    pub fn is_dag(&self) -> bool {
        !is_cyclic_directed(&self.graph)
    }

    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    pub fn nodes(&self) -> Vec<String> {
        self.graph.node_weights().map(|n| n.name.clone()).collect()
    }

    pub fn edges(&self) -> Vec<(String, String)> {
        self.graph
            .edge_indices()
            .filter_map(|e| {
                let (a, b) = self.graph.edge_endpoints(e)?;
                Some((self.graph[a].name.clone(), self.graph[b].name.clone()))
            })
            .collect()
    }

    pub fn has_edge(&self, from: &str, to: &str) -> bool {
        if let (Some(&f), Some(&t)) = (self.name_to_idx.get(from), self.name_to_idx.get(to)) {
            self.graph.contains_edge(f, t)
        } else {
            false
        }
    }

    pub fn remove_node(&mut self, name: &str) -> bool {
        if let Some(&idx) = self.name_to_idx.get(name) {
            self.graph.remove_node(idx);
            self.name_to_idx.remove(name);
            self.rebuild_index_map();
            true
        } else {
            false
        }
    }

    pub fn get_descendants(&self, name: &str) -> Vec<String> {
        let Some(&start) = self.name_to_idx.get(name) else {
            return vec![];
        };

        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        for neighbor in self.graph.neighbors_directed(start, Direction::Outgoing) {
            if visited.insert(neighbor) {
                queue.push_back(neighbor);
            }
        }

        while let Some(node) = queue.pop_front() {
            result.push(self.graph[node].name.clone());
            for neighbor in self.graph.neighbors_directed(node, Direction::Outgoing) {
                if visited.insert(neighbor) {
                    queue.push_back(neighbor);
                }
            }
        }

        result
    }

    pub fn get_ancestors(&self, name: &str) -> Vec<String> {
        let Some(&start) = self.name_to_idx.get(name) else {
            return vec![];
        };

        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        for neighbor in self.graph.neighbors_directed(start, Direction::Incoming) {
             if visited.insert(neighbor) {
                 queue.push_back(neighbor);
             }
        }

        while let Some(node) = queue.pop_front() {
            result.push(self.graph[node].name.clone());
            for neighbor in self.graph.neighbors_directed(node, Direction::Incoming) {
                if visited.insert(neighbor) {
                    queue.push_back(neighbor);
                }
            }
        }
        result
    }

    pub fn verify_ancestry(&self, node: &str, claimed_parent: &str) -> bool {
        let ancestors = self.get_ancestors(node);
        ancestors.contains(&claimed_parent.to_string())
    }

    /// THE DAMPING KERNEL: Mass-based physics propagation
    /// E_out = E_in * decay * (1.0 / (1.0 + mass))
    pub fn propagate(
        &self,
        order: &[String],
        source_node: &str,
        effect: f64,
        decay: f64,
        cap: Option<f64>,
        forward: bool,
        backward: bool,
    ) -> HashMap<String, f64> {
        let mut results = HashMap::new();

        let source_idx = match order.iter().position(|n| n == source_node) {
            Some(idx) => idx,
            None => return results,
        };

        // Source effect is absolute (internal resonance)
        results.insert(source_node.to_string(), effect);

        // Helper to get mass
        let get_mass = |name: &str| -> f64 {
            if let Some(&idx) = self.name_to_idx.get(name) {
                self.graph[idx].mass
            } else {
                1.0 // Default mass if not in graph (shouldn't happen for chained nodes usually)
            }
        };

        // Forward diffusion
        if forward {
            let mut cur = effect;
            for j in (source_idx + 1)..order.len() {
                let target_node = &order[j];
                let mass = get_mass(target_node);
                
                // PHYSICS: Inertia Damping
                // Higher mass = harder to move = more damping
                let inertia_factor = 1.0 / (1.0 + mass);
                
                cur *= decay * inertia_factor;

                if let Some(c) = cap {
                    if cur < c { break; }
                }

                let entry = results.entry(target_node.clone()).or_insert(0.0);
                *entry = entry.max(cur);
            }
        }

        // Backward diffusion
        if backward {
            let mut cur = effect;
            for j in (0..source_idx).rev() {
                let target_node = &order[j];
                let mass = get_mass(target_node);
                
                let inertia_factor = 1.0 / (1.0 + mass);
                
                cur *= decay * inertia_factor;

                if let Some(c) = cap {
                    if cur < c { break; }
                }

                let entry = results.entry(target_node.clone()).or_insert(0.0);
                *entry = entry.max(cur);
            }
        }

        results
    }

    fn rebuild_index_map(&mut self) {
        self.name_to_idx.clear();
        for idx in self.graph.node_indices() {
            let name = self.graph[idx].name.clone();
            self.name_to_idx.insert(name, idx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mass_damping() {
        let mut engine = PropagationEngine::new();
        // A -> B -> C
        // B has High Mass (damping)
        engine.add_node("A".into(), 0.0); // Super conductor
        engine.add_node("B".into(), 9.0); // Heavy mass (1/(1+9) = 0.1 transmission)
        engine.add_node("C".into(), 0.0); 

        let order = vec!["A".into(), "B".into(), "C".into()];
        
        // Propagate 1.0 from A. Decay 1.0 (no natural decay, pure mass damping)
        let res = engine.propagate(&order, "A", 1.0, 1.0, None, true, false);
        
        // A = 1.0
        assert_eq!(res["A"], 1.0);
        
        // B = 1.0 * decay(1.0) * inertia(1/(1+9)=0.1) = 0.1
        assert!((res["B"] - 0.1).abs() < 1e-10);
        
        // C = 0.1 * decay(1.0) * inertia(1/(1+0)=1.0) = 0.1
        assert!((res["C"] - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_critical_transmission() {
        let mut engine = PropagationEngine::new();
        // A (Critical) -> B (Critical)
        engine.add_node("A".into(), 0.1); 
        engine.add_node("B".into(), 0.1); 

        let order = vec!["A".into(), "B".into()];
        
        // Propagate 1.0. Mass 0.1 -> factor = 1/(1.1) ~= 0.909
        let res = engine.propagate(&order, "A", 1.0, 1.0, None, true, false);
        
        assert_eq!(res["A"], 1.0);
        assert!(res["B"] > 0.9); // High transmission
    }
}
