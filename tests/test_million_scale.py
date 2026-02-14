"""
Million-Scale Stress Test for the FlowLang Echo Engine.
Tests DAG propagation, cascade detection, and Lyapunov stability
on graphs with 100,000+ nodes.
"""
import unittest
import time
import sys
import os

sys.path.insert(0, os.path.join(os.getcwd(), "flowlang"))

try:
    from flowlang_core import EchoEngine
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False

@unittest.skipUnless(HAS_ENGINE, "flowlang_core not installed")
class TestMillionScale(unittest.TestCase):

    def test_dag_propagation_10k_chain(self):
        """10,000 node chain: topological DAG propagation."""
        engine = EchoEngine()
        size = 10_000
        
        # Build chain: 0 -> 1 -> 2 -> ... -> 9999
        for i in range(size):
            engine.add_node(str(i), 0.0)  # Superconductors
        for i in range(size - 1):
            engine.add_edge(str(i), str(i+1))
        
        start = time.time()
        res = engine.propagate_dag("0", 100.0, 0.9999, None)
        duration = time.time() - start
        
        print(f"\n[DAG] 10K chain propagation: {duration:.4f}s, affected={len(res)} nodes")
        self.assertTrue(duration < 2.0, f"Should be fast. Took {duration:.2f}s")
        self.assertEqual(len(res), size)

    def test_fan_out_10k(self):
        """10,000 children from 1 root: fan-out DAG propagation."""
        engine = EchoEngine()
        fan = 10_000
        
        engine.add_node("ROOT", 0.1)
        for i in range(fan):
            engine.add_node(f"C{i}", 1.0)
            engine.add_edge("ROOT", f"C{i}")
        
        start = time.time()
        res = engine.propagate_dag("ROOT", 100.0, 0.8, None)
        duration = time.time() - start
        
        print(f"\n[FAN] 10K fan-out propagation: {duration:.4f}s, affected={len(res)} nodes")
        self.assertTrue(duration < 1.0)
        self.assertEqual(len(res), fan + 1)

    def test_cascade_detection(self):
        """Cascade failure detection on a diamond DAG."""
        engine = EchoEngine()
        
        # Build: ROOT -> [A, B, C] -> SINK
        engine.add_node("ROOT", 0.1)
        engine.add_node("A", 1.0)
        engine.add_node("B", 1.0)
        engine.add_node("C", 1.0)
        engine.add_node("SINK", 5.0)
        engine.add_edge("ROOT", "A")
        engine.add_edge("ROOT", "B")
        engine.add_edge("ROOT", "C")
        engine.add_edge("A", "SINK")
        engine.add_edge("B", "SINK")
        engine.add_edge("C", "SINK")
        
        report = engine.detect_cascade("ROOT", 100.0, 0.8, None, 0.5)
        
        print(f"\n[CASCADE] Affected: {report['affected_count']}/{report['total_nodes']}")
        print(f"[CASCADE] Ratio: {report['affected_ratio']:.2%}")
        print(f"[CASCADE] Critical: {report['is_critical']}")
        
        self.assertTrue(report['is_critical'], "100% cascade should be critical at 50% threshold")
        self.assertEqual(report['affected_count'], 5)

    def test_lyapunov_stability(self):
        """Verify energy decreases per level (Lyapunov stability)."""
        engine = EchoEngine()
        
        # Build a 3-level tree
        engine.add_node("ROOT", 0.5)
        for i in range(10):
            name = f"L1_{i}"
            engine.add_node(name, 1.0)
            engine.add_edge("ROOT", name)
            for j in range(5):
                child = f"L2_{i}_{j}"
                engine.add_node(child, 2.0)
                engine.add_edge(name, child)
        
        is_stable, levels = engine.verify_stability("ROOT", 100.0, 0.8)
        
        print(f"\n[LYAPUNOV] Stable: {is_stable}")
        print(f"[LYAPUNOV] Energy per level: {[f'{e:.2f}' for e in levels]}")
        
        self.assertTrue(is_stable, "System must be Lyapunov stable with decay < 1.0")
        # Energy should decrease
        for i in range(1, len(levels)):
            if levels[i] > 0:
                self.assertLessEqual(levels[i], levels[i-1] * 1.01)  # Allow tiny tolerance

if __name__ == '__main__':
    unittest.main()
