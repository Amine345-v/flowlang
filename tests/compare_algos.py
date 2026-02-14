
import unittest
from flowlang_core import EchoEngine

class TestAlgoComparison(unittest.TestCase):
    def test_compare_fan_in_amplification(self):
        """
        Compare DAG Propagation (Physics) vs Sensitivity Damping (Protocol)
        in a Fan-In scenario (Amplification Risk).
        """
        engine = EchoEngine()
        
        # Scenario: 10 upstream modules (U0..U9) all feed into 1 downstream module (D)
        # S (Source) triggers all U nodes.
        # This simulates a "Core Library Update" affecting 10 features, 
        # which all impact the "Checkout" service.
        
        # 0. Setup
        engine.add_node("Source", 0.0) # Critical update
        engine.add_node("Checkout", 1.0) # Normal service
        
        fan_size = 10
        for i in range(fan_size):
            u_name = f"Upstream_{i}"
            engine.add_node(u_name, 0.0)
            engine.add_edge("Source", u_name)
            engine.add_edge(u_name, "Checkout")
            
        print(f"\n--- Scenario: Fan-In ({fan_size} paths to Checkout) ---")
        
        # 1. Physics Simulation (propagate_dag)
        # Real-world: 10 changes hitting Checkout SHOULD cause high stress.
        # Energy determines risk.
        physics_res = engine.propagate_dag("Source", 100.0, 0.9, None)
        p_val = physics_res["Checkout"]
        
        # 2. Safety Protocol (sensitivity_damping)
        # Control: We want to limit pure reach, regardless of complexity.
        # Checkout is Depth 2. Strict decay.
        protocol_res = engine.sensitivity_damping("Source", 100.0, 0.5, None) # gamma 0.5 ~ decay 0.6
        # equivalent decay for comparison: e^-gamma = 0.9 => gamma = -ln(0.9) ~= 0.105
        # Let's use gamma that matches decay 0.9 approx -> 0.1
        safe_res = engine.sensitivity_damping("Source", 100.0, 0.1, None)
        s_val = safe_res["Checkout"]
        
        print(f"[Physics Engine] Checkout Stress: {p_val:.2f} (High Risk Accumulation)")
        print(f"[Safety Protocol] Checkout Limit:  {s_val:.2f} (Strict Depth Limit)")
        
        # Conclusion
        if p_val > s_val * 2:
            print(">> CONCLUSION: Physics shows DANGEROUS amplification.")
            print(">> ACTION: The Safety Protocol would cut this off to prevent collapse.")
        else:
            print(">> CONCLUSION: System is stable.")

    def test_compare_chain_decay(self):
        """
        Compare in a simple chain. They should be similar if gamma matches decay.
        """
        engine = EchoEngine()
        # A -> B -> C -> D
        nodes = ["A", "B", "C", "D"]
        for n in nodes: engine.add_node(n, 0.0)
        for i in range(len(nodes)-1): engine.add_edge(nodes[i], nodes[i+1])
        
        print(f"\n--- Scenario: Linear Chain (Depth 3) ---")
        
        # Decay 0.6
        p_res = engine.propagate_dag("A", 100.0, 0.6, None)
        # Gamma corresponding to 0.6: -ln(0.6) ~= 0.51
        s_res = engine.sensitivity_damping("A", 100.0, 0.5108, None)
        
        print(f"[Physics Engine] D value: {p_res['D']:.2f}")
        print(f"[Safety Protocol] D value: {s_res['D']:.2f}")
        
        self.assertAlmostEqual(p_res['D'], s_res['D'], delta=1.0)
        print(">> CONCLUSION: In simple chains, they agree perfectly.")

if __name__ == '__main__':
    unittest.main()
