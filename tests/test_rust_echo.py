import unittest
import time
import sys
import os

# Ensure we can import flowlang
sys.path.insert(0, os.path.join(os.getcwd(), "flowlang"))

try:
    from flowlang_core import EchoEngine
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

class TestRustEcho(unittest.TestCase):
    def setUp(self):
        if not HAS_RUST:
            self.skipTest("Rust extension not available")
        self.engine = EchoEngine()

    def test_basic_graph_ops(self):
        """Test basic node/edge operations in Rust engine."""
        self.assertTrue(self.engine.add_node("A"))
        self.assertFalse(self.engine.add_node("A")) # Duplicate
        self.engine.add_node("B")
        self.assertTrue(self.engine.add_edge("A", "B"))
        
        self.assertTrue(self.engine.is_dag())
        self.assertEqual(self.engine.node_count(), 2)
        self.assertEqual(self.engine.edge_count(), 1)
        
        # Test Cycle
        self.assertFalse(self.engine.add_edge("B", "A")) # Should be rejected
        self.assertTrue(self.engine.is_dag())

    def test_propagation_correctness(self):
        """Test that Rust propagation matches expected decay logic."""
        # Chain: A -> B -> C -> D
        # Effect at A=1.0, decay=0.5
        # Expected: A=1.0, B=0.5, C=0.25, D=0.125
        order = ["A", "B", "C", "D"]
        res = self.engine.propagate(order, "A", 1.0, 0.5, None, True, False)
        
        self.assertAlmostEqual(res["A"], 1.0)
        self.assertAlmostEqual(res["B"], 0.5)
        self.assertAlmostEqual(res["C"], 0.25)
        self.assertAlmostEqual(res["D"], 0.125)

    def test_benchmark_large_chain(self):
        """Benchmark Rust speed on a large chain."""
        size = 10000
        order = [str(i) for i in range(size)]
        
        start = time.time()
        # Propagate from middle
        mid = str(size // 2)
        res = self.engine.propagate(order, mid, 100.0, 0.99, None, True, True)
        duration = time.time() - start
        
        print(f"\nRust propagation on {size} nodes took: {duration:.6f}s")
        self.assertTrue(duration < 0.1, f"Rust should be fast! Took {duration}s")
        self.assertEqual(len(res), size) # Should reach ends with 0.99 decay

if __name__ == '__main__':
    unittest.main()
