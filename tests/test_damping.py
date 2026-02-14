import unittest
import sys
import os
from flowlang.graph_engine import SystemTreeEngine
from flowlang.types import CriticalFeature, EchoSignature

# Ensure we can import flowlang
sys.path.insert(0, os.path.join(os.getcwd(), "flowlang"))

class TestDampingPhysics(unittest.TestCase):
    def setUp(self):
        self.engine = SystemTreeEngine()
        if not getattr(self.engine, 'rust_engine', None):
            self.skipTest("Rust engine not available for Deep Tech physics")

    def test_mass_difference(self):
        """Verify that LOW criticality nodes (High Mass) dampen more than CRITICAL nodes (Low Mass)."""
        # Chain A: CRITICAL -> CRITICAL -> CRITICAL
        # Chain B: LOW -> LOW -> LOW
        
        # Build Chain A (Superconductors - Mass 0.1)
        # Factor = 1/(1+0.1) = 0.909
        self.engine.graph.add_node("A1", data=CriticalFeature(name="A1", value=1, echo_signature=EchoSignature.CRITICAL))
        self.engine.rust_engine.add_node("A1", 0.1)
        self.engine.graph.add_node("A2", data=CriticalFeature(name="A2", value=1, echo_signature=EchoSignature.CRITICAL))
        self.engine.rust_engine.add_node("A2", 0.1)
        self.engine.graph.add_node("A3", data=CriticalFeature(name="A3", value=1, echo_signature=EchoSignature.CRITICAL))
        self.engine.rust_engine.add_node("A3", 0.1)
        
        # Build Chain B (Insulators - Mass 10.0)
        # Factor = 1/(1+10) = 0.0909
        self.engine.graph.add_node("B1", data=CriticalFeature(name="B1", value=1, echo_signature=EchoSignature.LOW))
        self.engine.rust_engine.add_node("B1", 10.0)
        self.engine.graph.add_node("B2", data=CriticalFeature(name="B2", value=1, echo_signature=EchoSignature.LOW))
        self.engine.rust_engine.add_node("B2", 10.0)
        self.engine.graph.add_node("B3", data=CriticalFeature(name="B3", value=1, echo_signature=EchoSignature.LOW))
        self.engine.rust_engine.add_node("B3", 10.0)
        
        # Propagate through Chain A
        order_a = ["A1", "A2", "A3"]
        res_a = self.engine.rust_engine.propagate(order_a, "A1", 100.0, 1.0, None, True, False)
        
        # Propagate through Chain B
        order_b = ["B1", "B2", "B3"]
        res_b = self.engine.rust_engine.propagate(order_b, "B1", 100.0, 1.0, None, True, False)
        
        if "A3" in res_a:
            print(f"\nCRITICAL Chain (Mass 0.1): {res_a['A3']:.4f}")
        if "B3" in res_b:
            print(f"LOW Chain (Mass 10.0): {res_b['B3']:.4f}")
        
        self.assertTrue(res_a['A3'] > 80.0, "Critical chain should transmit well (>80%)")
        self.assertTrue(res_b['B3'] < 10.0, "Low chain should dampen heavily (<10%)")
        
    def test_mixed_chain_physics(self):
        """Verify physics in a mixed chain."""
        # M1(Critical) -> M2(Low) -> M3(Critical)
        # M1=100 -> M2 (dampens heavily) -> M3 (transmits slightly less)
        
        self.engine.rust_engine.add_node("M1", 0.1)
        self.engine.rust_engine.add_node("M2", 10.0)
        self.engine.rust_engine.add_node("M3", 0.1)
        
        order = ["M1", "M2", "M3"]
        res = self.engine.rust_engine.propagate(order, "M1", 100.0, 1.0, None, True, False)
        
        # Theory:
        # M1 = 100
        # M2 = 100 * (1/(1+10)) = 9.09
        # M3 = 9.09 * (1/(1+0.1)) = 8.26
        
        print(f"Mixed Chain: {res['M1']} -> {res.get('M2',0):.2f} -> {res.get('M3',0):.2f}")
        
        self.assertAlmostEqual(res['M2'], 9.09, delta=1.0)
        self.assertAlmostEqual(res['M3'], 8.26, delta=1.0)

if __name__ == '__main__':
    unittest.main()
