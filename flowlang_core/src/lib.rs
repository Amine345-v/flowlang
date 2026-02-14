use pyo3::prelude::*;
use std::collections::HashMap;

mod propagation;

use propagation::PropagationEngine;

/// High-performance echo propagation engine for FlowLang.
/// This includes the Deep Tech "Damping Kernel".
#[pyclass]
pub struct EchoEngine {
    engine: PropagationEngine,
}

#[pymethods]
impl EchoEngine {
    #[new]
    fn new() -> Self {
        EchoEngine {
            engine: PropagationEngine::new(),
        }
    }

    /// Add a named node to the graph with a specific mass (inertia).
    /// Low mass = high criticality (transmits well).
    /// High mass = low importance (dampens well).
    #[pyo3(signature = (name, mass=1.0))]
    fn add_node(&mut self, name: String, mass: f64) -> bool {
        self.engine.add_node(name, mass)
    }

    /// Add a directed edge (from → to). Returns false if it would create a cycle.
    fn add_edge(&mut self, from: String, to: String) -> bool {
        self.engine.add_edge(from, to)
    }

    /// Check if the graph is a valid DAG (no cycles).
    fn is_dag(&self) -> bool {
        self.engine.is_dag()
    }

    /// Get the number of nodes.
    fn node_count(&self) -> usize {
        self.engine.node_count()
    }

    /// Get the number of edges.
    fn edge_count(&self) -> usize {
        self.engine.edge_count()
    }

    /// Get all node names.
    fn nodes(&self) -> Vec<String> {
        self.engine.nodes()
    }

    /// Get all edges as (from, to) pairs.
    fn edges(&self) -> Vec<(String, String)> {
        self.engine.edges()
    }

    /// Check if an edge exists.
    fn has_edge(&self, from: String, to: String) -> bool {
        self.engine.has_edge(&from, &to)
    }

    /// Remove a node and all its connected edges.
    fn remove_node(&mut self, name: String) -> bool {
        self.engine.remove_node(&name)
    }

    /// Get all downstream descendants of a node (the "echo path").
    fn get_descendants(&self, name: String) -> Vec<String> {
        self.engine.get_descendants(&name)
    }

    /// Get all upstream ancestors of a node (the "reverse echo").
    fn get_ancestors(&self, name: String) -> Vec<String> {
        self.engine.get_ancestors(&name)
    }

    /// Verify if claimed_parent is actually an ancestor of node.
    fn verify_ancestry(&self, node: String, claimed_parent: String) -> bool {
        self.engine.verify_ancestry(&node, &claimed_parent)
    }

    /// THE HOT PATH: Bidirectional decay propagation.
    /// Uses the Damping Kernel for physics-based inertia.
    #[pyo3(signature = (order, source_node, effect, decay=0.6, cap=None, forward=true, backward=true))]
    fn propagate(
        &self,
        order: Vec<String>,
        source_node: String,
        effect: f64,
        decay: f64,
        cap: Option<f64>,
        forward: bool,
        backward: bool,
    ) -> HashMap<String, f64> {
        self.engine.propagate(
            &order,
            &source_node,
            effect,
            decay,
            cap,
            forward,
            backward,
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "EchoEngine(nodes={}, edges={}, is_dag={})",
            self.engine.node_count(),
            self.engine.edge_count(),
            self.engine.is_dag()
        )
    }
}

#[pymodule]
fn flowlang_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EchoEngine>()?;
    Ok(())
}
