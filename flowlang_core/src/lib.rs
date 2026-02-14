use pyo3::prelude::*;
use std::collections::HashMap;

mod propagation;

use propagation::PropagationEngine;

/// High-performance echo propagation engine for FlowLang.
/// Phase 2: Million-Scale Stability Engine with Deep Tech algorithms.
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

    // ─── Node/Edge Management ───────────────────────────────────

    #[pyo3(signature = (name, mass=1.0))]
    fn add_node(&mut self, name: String, mass: f64) -> bool {
        self.engine.add_node(name, mass)
    }

    fn add_edge(&mut self, from: String, to: String) -> bool {
        self.engine.add_edge(from, to)
    }

    fn is_dag(&self) -> bool { self.engine.is_dag() }
    fn node_count(&self) -> usize { self.engine.node_count() }
    fn edge_count(&self) -> usize { self.engine.edge_count() }
    fn nodes(&self) -> Vec<String> { self.engine.nodes() }

    fn edges(&self) -> Vec<(String, String)> { self.engine.edges() }

    fn has_edge(&self, from: String, to: String) -> bool {
        self.engine.has_edge(&from, &to)
    }

    fn remove_node(&mut self, name: String) -> bool {
        self.engine.remove_node(&name)
    }

    fn get_descendants(&self, name: String) -> Vec<String> {
        self.engine.get_descendants(&name)
    }

    fn get_ancestors(&self, name: String) -> Vec<String> {
        self.engine.get_ancestors(&name)
    }

    fn verify_ancestry(&self, node: String, claimed_parent: String) -> bool {
        self.engine.verify_ancestry(&node, &claimed_parent)
    }

    // ─── Algorithm 1: Chain Propagation (Damping Kernel) ────────

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
        self.engine.propagate(&order, &source_node, effect, decay, cap, forward, backward)
    }

    // ─── Algorithm 2: DAG Topological Propagation ───────────────

    #[pyo3(signature = (source_node, effect, decay=0.6, cap=None))]
    fn propagate_dag(
        &self,
        source_node: String,
        effect: f64,
        decay: f64,
        cap: Option<f64>,
    ) -> HashMap<String, f64> {
        self.engine.propagate_dag(&source_node, effect, decay, cap)
    }

    // ─── Algorithm 3: Cascade Failure Detection ─────────────────

    #[pyo3(signature = (source_node, effect, decay=0.6, cap=None, critical_threshold=0.1))]
    fn detect_cascade(
        &self,
        source_node: String,
        effect: f64,
        decay: f64,
        cap: Option<f64>,
        critical_threshold: f64,
    ) -> HashMap<String, PyObject> {
        Python::with_gil(|py| {
            let report = self.engine.detect_cascade(
                &source_node, effect, decay, cap, critical_threshold,
            );
            let mut result = HashMap::new();
            result.insert("source".to_string(), report.source.into_pyobject(py).unwrap().into_any().unbind());
            result.insert("affected_count".to_string(), report.affected_count.into_pyobject(py).unwrap().into_any().unbind());
            result.insert("total_nodes".to_string(), report.total_nodes.into_pyobject(py).unwrap().into_any().unbind());
            result.insert("affected_ratio".to_string(), report.affected_ratio.into_pyobject(py).unwrap().into_any().unbind());
            result.insert("is_critical".to_string(), pyo3::types::PyBool::new(py, report.is_critical).to_owned().into_any().unbind());
            result.insert("max_energy".to_string(), report.max_energy.into_pyobject(py).unwrap().into_any().unbind());
            result.insert("affected_nodes".to_string(), report.affected_nodes.into_pyobject(py).unwrap().into_any().unbind());
            result
        })
    }

    // ─── Algorithm 4: System Energy & Stability ─────────────────

    #[pyo3(signature = (source_node, effect, decay=0.6))]
    fn system_energy(&self, source_node: String, effect: f64, decay: f64) -> f64 {
        self.engine.system_energy(&source_node, effect, decay)
    }

    #[pyo3(signature = (source_node, effect, decay=0.6))]
    fn verify_stability(&self, source_node: String, effect: f64, decay: f64) -> (bool, Vec<f64>) {
        self.engine.verify_stability(&source_node, effect, decay)
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
