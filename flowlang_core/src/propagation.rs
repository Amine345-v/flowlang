use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::algo::is_cyclic_directed;
use petgraph::Direction;
use std::collections::{HashMap, HashSet, VecDeque};

/// Core propagation engine backed by petgraph DiGraph.
/// This replaces the Python/NetworkX hot path with zero-copy Rust performance.
pub struct PropagationEngine {
    graph: DiGraph<String, ()>,
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

    /// Add a node. Returns true if new, false if already exists.
    pub fn add_node(&mut self, name: String) -> bool {
        if self.name_to_idx.contains_key(&name) {
            return false;
        }
        let idx = self.graph.add_node(name.clone());
        self.name_to_idx.insert(name, idx);
        true
    }

    /// Add a directed edge. Returns false if it would create a cycle (DAG enforcement).
    pub fn add_edge(&mut self, from: String, to: String) -> bool {
        // Ensure both nodes exist
        self.add_node(from.clone());
        self.add_node(to.clone());

        let from_idx = self.name_to_idx[&from];
        let to_idx = self.name_to_idx[&to];

        // Add the edge
        self.graph.add_edge(from_idx, to_idx, ());

        // Check if this created a cycle
        if is_cyclic_directed(&self.graph) {
            // Remove the edge that caused the cycle
            // Find and remove the last edge we added
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
        self.graph.node_weights().cloned().collect()
    }

    pub fn edges(&self) -> Vec<(String, String)> {
        self.graph
            .edge_indices()
            .filter_map(|e| {
                let (a, b) = self.graph.edge_endpoints(e)?;
                Some((self.graph[a].clone(), self.graph[b].clone()))
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
            // Rebuild index map since petgraph may reassign indices
            self.rebuild_index_map();
            true
        } else {
            false
        }
    }

    /// Get all downstream descendants via BFS.
    pub fn get_descendants(&self, name: &str) -> Vec<String> {
        let Some(&start) = self.name_to_idx.get(name) else {
            return vec![];
        };

        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        // Start from all outgoing neighbors
        for neighbor in self.graph.neighbors_directed(start, Direction::Outgoing) {
            if visited.insert(neighbor) {
                queue.push_back(neighbor);
            }
        }

        while let Some(node) = queue.pop_front() {
            result.push(self.graph[node].clone());
            for neighbor in self.graph.neighbors_directed(node, Direction::Outgoing) {
                if visited.insert(neighbor) {
                    queue.push_back(neighbor);
                }
            }
        }

        result
    }

    /// Get all upstream ancestors via reverse BFS.
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
            result.push(self.graph[node].clone());
            for neighbor in self.graph.neighbors_directed(node, Direction::Incoming) {
                if visited.insert(neighbor) {
                    queue.push_back(neighbor);
                }
            }
        }

        result
    }

    /// Check if claimed_parent is an ancestor of node.
    pub fn verify_ancestry(&self, node: &str, claimed_parent: &str) -> bool {
        let ancestors = self.get_ancestors(node);
        ancestors.contains(&claimed_parent.to_string())
    }

    /// THE HOT PATH: Bidirectional decay propagation on an ordered chain.
    ///
    /// Given:
    ///   - order: [A, B, C, D, E] — the chain's node sequence
    ///   - source_node: "C" — where the effect originates
    ///   - effect: 1.0 — initial effect magnitude
    ///   - decay: 0.6 — multiplicative decay per hop
    ///   - cap: Some(0.01) — stop propagating below this threshold
    ///   - forward: true — propagate C→D→E
    ///   - backward: true — propagate C→B→A
    ///
    /// Returns:
    ///   { "C": 1.0, "D": 0.6, "E": 0.36, "B": 0.6, "A": 0.36 }
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

        // Find source index in the order
        let source_idx = match order.iter().position(|n| n == source_node) {
            Some(idx) => idx,
            None => return results,
        };

        // Set source effect
        results.insert(source_node.to_string(), effect);

        // Forward diffusion: source → end of chain
        if forward {
            let mut cur = effect;
            for j in (source_idx + 1)..order.len() {
                cur *= decay;

                // Logical Firewall: stop below threshold
                if let Some(c) = cap {
                    if cur < c {
                        break;
                    }
                }

                let entry = results.entry(order[j].clone()).or_insert(0.0);
                *entry = entry.max(cur);
            }
        }

        // Backward diffusion: source → start of chain
        if backward {
            let mut cur = effect;
            for j in (0..source_idx).rev() {
                cur *= decay;

                if let Some(c) = cap {
                    if cur < c {
                        break;
                    }
                }

                let entry = results.entry(order[j].clone()).or_insert(0.0);
                *entry = entry.max(cur);
            }
        }

        results
    }

    /// Rebuild the name→index map after graph mutations.
    fn rebuild_index_map(&mut self) {
        self.name_to_idx.clear();
        for idx in self.graph.node_indices() {
            let name = self.graph[idx].clone();
            self.name_to_idx.insert(name, idx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_propagation() {
        let engine = PropagationEngine::new();

        let order = vec!["A".into(), "B".into(), "C".into(), "D".into(), "E".into()];
        let result = engine.propagate(&order, "C", 1.0, 0.5, None, true, true);

        assert_eq!(result["C"], 1.0);
        assert!((result["D"] - 0.5).abs() < 1e-10);
        assert!((result["E"] - 0.25).abs() < 1e-10);
        assert!((result["B"] - 0.5).abs() < 1e-10);
        assert!((result["A"] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_cap_stops_propagation() {
        let engine = PropagationEngine::new();

        let order = vec!["A".into(), "B".into(), "C".into(), "D".into(), "E".into()];
        let result = engine.propagate(&order, "A", 1.0, 0.5, Some(0.3), true, false);

        assert_eq!(result["A"], 1.0);
        assert!((result["B"] - 0.5).abs() < 1e-10);
        // C = 0.25 < 0.3 cap → should NOT be in results
        assert!(!result.contains_key("C"));
    }

    #[test]
    fn test_dag_enforcement() {
        let mut engine = PropagationEngine::new();
        assert!(engine.add_edge("A".into(), "B".into()));
        assert!(engine.add_edge("B".into(), "C".into()));
        // This would create A→B→C→A cycle
        assert!(!engine.add_edge("C".into(), "A".into()));
        assert!(engine.is_dag());
    }

    #[test]
    fn test_descendants() {
        let mut engine = PropagationEngine::new();
        engine.add_edge("A".into(), "B".into());
        engine.add_edge("B".into(), "C".into());
        engine.add_edge("B".into(), "D".into());

        let desc = engine.get_descendants("A");
        assert!(desc.contains(&"B".to_string()));
        assert!(desc.contains(&"C".to_string()));
        assert!(desc.contains(&"D".to_string()));
    }
}
