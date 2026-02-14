use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::algo::{is_cyclic_directed, toposort};
use petgraph::Direction;
use std::collections::{HashMap, HashSet, VecDeque};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  NODE DATA — Every node in the System Tree has physical properties
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[derive(Clone, Debug)]
pub struct NodeData {
    pub name: String,
    /// Physical "mass" — controls how much energy this node absorbs.
    /// Low mass (0.1) = Superconductor (CRITICAL features)
    /// High mass (10.0) = Insulator (LOW features)
    pub mass: f64,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  CASCADE REPORT — Result of cascade failure analysis
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[derive(Clone, Debug)]
pub struct CascadeReport {
    pub source: String,
    pub affected_count: usize,
    pub total_nodes: usize,
    pub affected_ratio: f64,
    pub is_critical: bool,
    pub max_energy: f64,
    pub affected_nodes: Vec<String>,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  PROPAGATION ENGINE — The Deep Tech Core
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pub struct PropagationEngine {
    graph: DiGraph<NodeData, ()>,
    name_to_idx: HashMap<String, NodeIndex>,
}

impl PropagationEngine {
    pub fn new() -> Self {
        PropagationEngine {
            graph: DiGraph::new(),
            name_to_idx: HashMap::new(),
        }
    }

    // ─── Node Management ────────────────────────────────────────

    pub fn add_node(&mut self, name: String, mass: f64) -> bool {
        if self.name_to_idx.contains_key(&name) {
            return false;
        }
        let data = NodeData { name: name.clone(), mass };
        let idx = self.graph.add_node(data);
        self.name_to_idx.insert(name, idx);
        true
    }

    pub fn add_edge(&mut self, from: String, to: String) -> bool {
        if !self.name_to_idx.contains_key(&from) {
            self.add_node(from.clone(), 1.0);
        }
        if !self.name_to_idx.contains_key(&to) {
            self.add_node(to.clone(), 1.0);
        }

        let from_idx = self.name_to_idx[&from];
        let to_idx = self.name_to_idx[&to];

        self.graph.add_edge(from_idx, to_idx, ());

        if is_cyclic_directed(&self.graph) {
            if let Some(edge) = self.graph.find_edge(from_idx, to_idx) {
                self.graph.remove_edge(edge);
            }
            return false;
        }
        true
    }

    pub fn is_dag(&self) -> bool {
        !is_cyclic_directed(&self.graph)
    }

    pub fn node_count(&self) -> usize { self.graph.node_count() }
    pub fn edge_count(&self) -> usize { self.graph.edge_count() }

    pub fn nodes(&self) -> Vec<String> {
        self.graph.node_weights().map(|n| n.name.clone()).collect()
    }

    pub fn edges(&self) -> Vec<(String, String)> {
        self.graph.edge_indices().filter_map(|e| {
            let (a, b) = self.graph.edge_endpoints(e)?;
            Some((self.graph[a].name.clone(), self.graph[b].name.clone()))
        }).collect()
    }

    pub fn has_edge(&self, from: &str, to: &str) -> bool {
        if let (Some(&f), Some(&t)) = (self.name_to_idx.get(from), self.name_to_idx.get(to)) {
            self.graph.contains_edge(f, t)
        } else { false }
    }

    pub fn remove_node(&mut self, name: &str) -> bool {
        if let Some(&idx) = self.name_to_idx.get(name) {
            self.graph.remove_node(idx);
            self.name_to_idx.remove(name);
            self.rebuild_index_map();
            true
        } else { false }
    }

    // ─── Graph Traversal ────────────────────────────────────────

    pub fn get_descendants(&self, name: &str) -> Vec<String> {
        let Some(&start) = self.name_to_idx.get(name) else { return vec![]; };
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        for n in self.graph.neighbors_directed(start, Direction::Outgoing) {
            if visited.insert(n) { queue.push_back(n); }
        }
        while let Some(node) = queue.pop_front() {
            result.push(self.graph[node].name.clone());
            for n in self.graph.neighbors_directed(node, Direction::Outgoing) {
                if visited.insert(n) { queue.push_back(n); }
            }
        }
        result
    }

    pub fn get_ancestors(&self, name: &str) -> Vec<String> {
        let Some(&start) = self.name_to_idx.get(name) else { return vec![]; };
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        for n in self.graph.neighbors_directed(start, Direction::Incoming) {
            if visited.insert(n) { queue.push_back(n); }
        }
        while let Some(node) = queue.pop_front() {
            result.push(self.graph[node].name.clone());
            for n in self.graph.neighbors_directed(node, Direction::Incoming) {
                if visited.insert(n) { queue.push_back(n); }
            }
        }
        result
    }

    pub fn verify_ancestry(&self, node: &str, claimed_parent: &str) -> bool {
        self.get_ancestors(node).contains(&claimed_parent.to_string())
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    //  ALGORITHM 1: Chain-based Propagation (Phase 1 - Damping Kernel)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
        results.insert(source_node.to_string(), effect);

        let get_mass = |name: &str| -> f64 {
            self.name_to_idx.get(name)
                .map(|&idx| self.graph[idx].mass)
                .unwrap_or(1.0)
        };

        if forward {
            let mut cur = effect;
            for j in (source_idx + 1)..order.len() {
                let mass = get_mass(&order[j]);
                let inertia = 1.0 / (1.0 + mass);
                cur *= decay * inertia;
                if let Some(c) = cap { if cur < c { break; } }
                let entry = results.entry(order[j].clone()).or_insert(0.0);
                *entry = entry.max(cur);
            }
        }
        if backward {
            let mut cur = effect;
            for j in (0..source_idx).rev() {
                let mass = get_mass(&order[j]);
                let inertia = 1.0 / (1.0 + mass);
                cur *= decay * inertia;
                if let Some(c) = cap { if cur < c { break; } }
                let entry = results.entry(order[j].clone()).or_insert(0.0);
                *entry = entry.max(cur);
            }
        }
        results
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    //  ALGORITHM 2: DAG-aware Topological Propagation (Phase 2)
    //  "The Real Echo" — propagates through the FULL graph topology,
    //  not just a linear chain. This is what handles millions of deps.
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    pub fn propagate_dag(
        &self,
        source_node: &str,
        effect: f64,
        decay: f64,
        cap: Option<f64>,
    ) -> HashMap<String, f64> {
        let mut energies: HashMap<NodeIndex, f64> = HashMap::new();

        let Some(&source_idx) = self.name_to_idx.get(source_node) else {
            return HashMap::new();
        };

        // Step 1: Topological sort of the entire graph
        let topo_order = match toposort(&self.graph, None) {
            Ok(order) => order,
            Err(_) => return HashMap::new(), // Graph has cycle (shouldn't happen)
        };

        // Step 2: Find source position in topological order
        let source_pos = match topo_order.iter().position(|&n| n == source_idx) {
            Some(pos) => pos,
            None => return HashMap::new(),
        };

        // Step 3: Seed the source
        energies.insert(source_idx, effect);

        // Step 4: Forward propagation in topological order
        // Only process nodes AFTER the source in topological order
        for i in (source_pos + 1)..topo_order.len() {
            let node = topo_order[i];
            let node_mass = self.graph[node].mass;
            let inertia = 1.0 / (1.0 + node_mass);

            // Collect energy from ALL parents (predecessors)
            let mut incoming_energy = 0.0f64;
            for parent in self.graph.neighbors_directed(node, Direction::Incoming) {
                if let Some(&parent_energy) = energies.get(&parent) {
                    // Each incoming edge transmits with decay * inertia
                    incoming_energy += parent_energy * decay * inertia;
                }
            }

            // Only store if above cap threshold
            if incoming_energy > 0.0 {
                if let Some(c) = cap {
                    if incoming_energy < c { continue; }
                }
                energies.insert(node, incoming_energy);
            }
        }

        // Convert NodeIndex keys to String keys
        energies.iter()
            .map(|(&idx, &val)| (self.graph[idx].name.clone(), val))
            .collect()
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    //  ALGORITHM 3: Cascade Failure Detection
    //  "The Guardian" — detects when a single change threatens
    //  to destabilize the entire system.
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    pub fn detect_cascade(
        &self,
        source_node: &str,
        effect: f64,
        decay: f64,
        cap: Option<f64>,
        critical_threshold: f64,  // e.g. 0.1 = 10% of nodes affected = CRITICAL
    ) -> CascadeReport {
        let total = self.graph.node_count();
        if total == 0 {
            return CascadeReport {
                source: source_node.to_string(),
                affected_count: 0, total_nodes: 0, affected_ratio: 0.0,
                is_critical: false, max_energy: 0.0, affected_nodes: vec![],
            };
        }

        // Run DAG propagation
        let energies = self.propagate_dag(source_node, effect, decay, cap);
        let affected_count = energies.len();
        let affected_ratio = affected_count as f64 / total as f64;
        let max_energy = energies.values().cloned().fold(0.0f64, f64::max);

        let mut affected_nodes: Vec<String> = energies.keys().cloned().collect();
        affected_nodes.sort();

        CascadeReport {
            source: source_node.to_string(),
            affected_count,
            total_nodes: total,
            affected_ratio,
            is_critical: affected_ratio > critical_threshold,
            max_energy,
            affected_nodes,
        }
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    //  ALGORITHM 4: System Energy (Lyapunov Stability)
    //  "The Thermometer" — measures total energy in the system.
    //  In a stable system, energy MUST decrease over time/distance.
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    pub fn system_energy(
        &self,
        source_node: &str,
        effect: f64,
        decay: f64,
    ) -> f64 {
        let energies = self.propagate_dag(source_node, effect, decay, None);
        energies.values().sum()
    }

    /// Verify Lyapunov stability: total energy at hop N+1 < energy at hop N.
    /// Returns (is_stable, energy_per_level)
    pub fn verify_stability(
        &self,
        source_node: &str,
        effect: f64,
        decay: f64,
    ) -> (bool, Vec<f64>) {
        let Some(&source_idx) = self.name_to_idx.get(source_node) else {
            return (true, vec![]);
        };

        let topo_order = match toposort(&self.graph, None) {
            Ok(order) => order,
            Err(_) => return (false, vec![]),
        };

        let _source_pos = match topo_order.iter().position(|&n| n == source_idx) {
            Some(pos) => pos,
            None => return (true, vec![]),
        };

        // Compute energies
        let energies = self.propagate_dag(source_node, effect, decay, None);

        // Group by topological "level" (distance from source in topo order)
        // We use BFS levels from source
        let mut levels: HashMap<NodeIndex, usize> = HashMap::new();
        levels.insert(source_idx, 0);
        let mut queue = VecDeque::new();
        queue.push_back(source_idx);
        while let Some(node) = queue.pop_front() {
            let level = levels[&node];
            for neighbor in self.graph.neighbors_directed(node, Direction::Outgoing) {
                if !levels.contains_key(&neighbor) {
                    levels.insert(neighbor, level + 1);
                    queue.push_back(neighbor);
                }
            }
        }

        // Sum energy per level
        let max_level = levels.values().cloned().max().unwrap_or(0);
        let mut energy_per_level = vec![0.0f64; max_level + 1];
        for (&idx, &level) in &levels {
            if let Some(&energy) = energies.get(&self.graph[idx].name) {
                if level < energy_per_level.len() {
                    energy_per_level[level] += energy;
                }
            }
        }

        // Lyapunov check: each level's energy must be <= previous level
        let mut is_stable = true;
        for i in 1..energy_per_level.len() {
            if energy_per_level[i] > energy_per_level[i - 1] * 1.001 {
                // Allow tiny floating point tolerance
                is_stable = false;
                break;
            }
        }

        (is_stable, energy_per_level)
    }

    // ─── Internal ───────────────────────────────────────────────

    fn rebuild_index_map(&mut self) {
        self.name_to_idx.clear();
        for idx in self.graph.node_indices() {
            self.name_to_idx.insert(self.graph[idx].name.clone(), idx);
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  TESTS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(test)]
mod tests {
    use super::*;

    fn build_diamond() -> PropagationEngine {
        // Diamond DAG:    A
        //               /   \
        //              B     C
        //               \   /
        //                 D
        let mut e = PropagationEngine::new();
        e.add_node("A".into(), 0.0);
        e.add_node("B".into(), 0.0);
        e.add_node("C".into(), 0.0);
        e.add_node("D".into(), 0.0);
        e.add_edge("A".into(), "B".into());
        e.add_edge("A".into(), "C".into());
        e.add_edge("B".into(), "D".into());
        e.add_edge("C".into(), "D".into());
        e
    }

    #[test]
    fn test_dag_propagation_diamond() {
        let e = build_diamond();
        let res = e.propagate_dag("A", 100.0, 0.5, None);

        assert_eq!(res["A"], 100.0);
        // B = 100 * 0.5 * 1.0 = 50
        assert!((res["B"] - 50.0).abs() < 1e-10);
        // C = 100 * 0.5 * 1.0 = 50
        assert!((res["C"] - 50.0).abs() < 1e-10);
        // D receives from BOTH B and C:
        // D = (50 * 0.5) + (50 * 0.5) = 25 + 25 = 50
        assert!((res["D"] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_dag_mass_damping() {
        let mut e = PropagationEngine::new();
        // A (light) -> B (HEAVY) -> C (light)
        e.add_node("A".into(), 0.0);
        e.add_node("B".into(), 9.0);  // mass 9 -> factor 0.1
        e.add_node("C".into(), 0.0);
        e.add_edge("A".into(), "B".into());
        e.add_edge("B".into(), "C".into());

        let res = e.propagate_dag("A", 100.0, 1.0, None);
        // B = 100 * 1.0 * (1/10) = 10
        assert!((res["B"] - 10.0).abs() < 1e-10);
        // C = 10 * 1.0 * 1.0 = 10
        assert!((res["C"] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_cascade_detection() {
        let e = build_diamond();
        let report = e.detect_cascade("A", 100.0, 0.5, None, 0.5);
        // All 4 nodes affected (A, B, C, D)
        assert_eq!(report.affected_count, 4);
        assert!((report.affected_ratio - 1.0).abs() < 1e-10);
        assert!(report.is_critical); // 100% > 50% threshold
    }

    #[test]
    fn test_lyapunov_stability() {
        let e = build_diamond();
        let (is_stable, levels) = e.verify_stability("A", 100.0, 0.5);

        // With decay < 1.0 and mass >= 0, system MUST be stable
        assert!(is_stable, "System should be Lyapunov stable");
        // Level 0 = 100, Level 1 = 100 (50+50), Level 2 = 50
        // This is stable because levels don't increase beyond initial
        assert!(levels.len() >= 2);
    }

    #[test]
    fn test_large_dag() {
        // Build a wide DAG: 1 root -> 1000 children -> 1 sink
        let mut e = PropagationEngine::new();
        e.add_node("ROOT".into(), 0.0);
        e.add_node("SINK".into(), 0.0);
        for i in 0..1000 {
            let name = format!("N{}", i);
            e.add_node(name.clone(), 0.5);
            e.add_edge("ROOT".into(), name.clone());
            e.add_edge(name, "SINK".into());
        }

        let res = e.propagate_dag("ROOT", 100.0, 0.8, None);
        // ROOT = 100
        assert_eq!(res["ROOT"], 100.0);
        // Each child = 100 * 0.8 * (1/1.5) ~= 53.3
        let child_energy = res.get("N0").unwrap_or(&0.0);
        assert!(*child_energy > 50.0);
        // SINK receives from ALL 1000 children
        let sink_energy = res.get("SINK").unwrap_or(&0.0);
        assert!(*sink_energy > 1000.0); // Sum of 1000 children
    }
}
