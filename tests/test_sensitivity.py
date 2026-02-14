
import unittest
import math
from flowlang_core import EchoEngine

class TestSensitivityDamping(unittest.TestCase):
    def setUp(self):
        self.engine = EchoEngine()

    def test_pure_exponential_decay(self):
        """Verify I_d = I_0 * e^(-gamma * d) for a simple chain."""
        # A -> B -> C -> D
        nodes = ["A", "B", "C", "D"]
        for n in nodes:
            self.engine.add_node(n, 0.0) # Zero mass
        
        for i in range(len(nodes)-1):
            self.engine.add_edge(nodes[i], nodes[i+1])

        gamma = 0.5
        i0 = 100.0
        results = self.engine.sensitivity_damping("A", i0, gamma, None)

        print(f"\n[SENSITIVITY] Results: {results}")

        # Expected:
        # A (d=0): 100 * 1 = 100
        # B (d=1): 100 * e^-0.5
        # C (d=2): 100 * e^-1.0
        # D (d=3): 100 * e^-1.5

        self.assertAlmostEqual(results["A"], 100.0)
        self.assertAlmostEqual(results["B"], 100.0 * math.exp(-0.5 * 1))
        self.assertAlmostEqual(results["C"], 100.0 * math.exp(-0.5 * 2))
        self.assertAlmostEqual(results["D"], 100.0 * math.exp(-0.5 * 3))

    def test_mass_modulation(self):
        """Verify mass affects the damping as a multiplier: I_d * (1/(1+mass))."""
        # A(0.0) -> B(9.0) -> C(0.0)
        self.engine.add_node("A", 0.0)
        self.engine.add_node("B", 9.0) # Mass 9.0 -> Factor 0.1
        self.engine.add_node("C", 0.0)
        
        self.engine.add_edge("A", "B")
        self.engine.add_edge("B", "C")

        gamma = 0.5
        i0 = 100.0
        results = self.engine.sensitivity_damping("A", i0, gamma, None)

        # B (d=1, mass=9): 100 * e^-0.5 * 0.1
        expected_b = 100.0 * math.exp(-0.5) * 0.1
        self.assertAlmostEqual(results["B"], expected_b)

        # C (d=2, mass=0): 100 * e^-1.0 * 1.0
        # Note: In this protocol, mass at B doesn't affect C's "incoming" signal scalar, 
        # but C's own mass affects C's value. 
        # The protocol formula computes I_d based on depth and *local* mass.
        # So C is just depth 2 with mass 0.
        expected_c = 100.0 * math.exp(-1.0) * 1.0
        self.assertAlmostEqual(results["C"], expected_c)

if __name__ == '__main__':
    unittest.main()
