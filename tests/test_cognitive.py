import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.getcwd(), "flowlang"))

from flowlang.runtime import Runtime, EvalContext
from flowlang.errors import RuntimeFlowError
from flowlang.types import CriticalFeature, Contract, ValueTag
from flowlang.parser import parse


class TestCognitiveStack(unittest.TestCase):

    def test_graph_dag_enforcement(self):
        """Cyclic chains must be rejected by DAG enforcement."""
        r = Runtime()
        if r.system_tree is None:
            print("SKIP: no graph engine")
            return
        # Two chains that form a cycle: A->B and B->A
        src = '''
chain C1 {
    nodes: [A, B];
    propagation: causal(decay=0.5);
}
chain C2 {
    nodes: [B, A];
    propagation: causal(decay=0.5);
}
'''
        r.tree = parse(src)
        try:
            r._build_structs()
            self.fail("Expected cycle detection error")
        except RuntimeFlowError as e:
            self.assertIn("Cycles", str(e))

    def test_formal_verification_pass(self):
        """Contracts that hold must return passing result."""
        r = Runtime()
        feat = CriticalFeature(
            name="Temperature",
            value=150,
            contracts=[
                Contract("Max temp", "value < 200", "hard"),
                Contract("Min temp", "value > 100", "soft"),
            ]
        )
        result = r._verify_contracts([feat], {})
        self.assertTrue(result.value)
        self.assertEqual(result.meta["confidence"], 1.0)

    def test_formal_verification_fail(self):
        """Broken hard contracts must return failing result."""
        r = Runtime()
        feat = CriticalFeature(
            name="Temperature",
            value=250,
            contracts=[
                Contract("Max temp", "value < 200", "hard"),
            ]
        )
        result = r._verify_contracts([feat], {})
        self.assertFalse(result.value)
        self.assertEqual(result.meta["confidence"], 0.0)

    def test_policy_team_linking(self):
        """Policies must be linked to Teams in SystemTree."""
        r = Runtime()
        if r.system_tree is None:
            print("SKIP: no graph engine")
            return
        src = '''
policy StrictSafety {
    rules: ["NoLoops"];
}
team SafetyTeam: Command<Judge> [
    size = 1,
    policy = StrictSafety
];
'''
        r.tree = parse(src)
        r._build_structs()
        g = r.system_tree.graph
        self.assertIn("Policy:StrictSafety", g.nodes)
        self.assertIn("Team:SafetyTeam", g.nodes)
        self.assertTrue(
            g.has_edge("Policy:StrictSafety", "Team:SafetyTeam")
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
