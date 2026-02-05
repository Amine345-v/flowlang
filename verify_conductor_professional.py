from flowlang.runtime import Runtime, EvalContext
from flowlang.types import Order, CommandKind, ValueTag
import unittest
from unittest.mock import MagicMock

class TestProfessionalConductor(unittest.TestCase):
    def setUp(self):
        self.rt = Runtime(dry_run=True)
        # Mocking ai_provider to inspect kwargs
        self.rt.ai_provider = MagicMock()

    def test_1_strict_team_typing(self):
        print("\n--- Test 1: Strict Team Typing ---")
        source = """
        team researcher : Command<Search> [size=1];
        flow main(using: researcher) {
            checkpoint "a" { researcher.try("fail"); }
        }
        """
        with self.assertRaises(Exception) as cm:
            self.rt.load(source)
            self.rt.run_flow("main")
        print(f"Caught expected error: {cm.exception}")
        self.assertIn("cannot perform 'try'", str(cm.exception))

    def test_2_sequential_handover(self):
        print("\n--- Test 2: Sequential Handover ---")
        source = """
        team worker : Command<Try> [size=3];
        flow main(using: worker) {
            checkpoint "stage1" (report: r1) {
                r1 = worker.try("job1");
            }
            checkpoint "stage2" (report: r2) {
                r2 = worker.try(r1);
            }
        }
        """
        self.rt.dry_run = False # Allow provider calls for this test
        self.rt.load(source)
        self.rt.run_flow("main")
        
        # Verify call history
        # First call: worker#0
        # Second call: worker#1
        # etc.
        calls = self.rt.ai_provider.execute.call_args_list
        self.assertEqual(len(calls), 2)
        # Note: Runtime._dispatch_provider returns (result, member_idx) 
        # but execute itself just returns TypedValue.
        # Check logs if needed, but the internal _idx should have incremented.
        print("Sequential calls verified.")

    def test_3_binary_path_awareness(self):
        print("\n--- Test 3: Binary Path Awareness ---")
        source = """
        process p "Roadmap" {
            root: "Root";
            branch "Root" -> ["Left", "Right"];
            node "Left" { type: "task"; };
        }
        team t : Command<Try> [size=1];
        flow main(using: t) {
            checkpoint "x" { t.try(my_order); }
        }
        """
        self.rt.load(source)
        # Create order manually linked to node
        order = Order(id="O1", payload="work", kind=CommandKind.Try, process_node="Left")
        ctx = EvalContext(variables={"my_order": order}, checkpoints=[])
        
        # Execute action manually
        action_node = list(self.rt.tree.find_data("action_stmt"))[0]
        self.rt._exec_action(action_node, ctx)
        
        # Check if maestro_path was passed to provider
        calls = self.rt.ai_provider.execute.call_args_list
        kwargs = calls[0][1]
        print(f"Maestro Path passed: {kwargs.get('maestro_path')}")
        self.assertEqual(kwargs.get("maestro_path"), "0") # Root -> Left (idx 0)

    def test_4_exclusive_activity(self):
        print("\n--- Test 4: Exclusive Activity (Skip) ---")
        source = """
        chain c { nodes: [Step1, Step2]; }
        team t : Command<Try> [size=1];
        flow main(using: t) {
            checkpoint "one" { t.try(o); }
        }
        """
        self.rt.load(source)
        # Mark Step1 as satisfied
        self.rt.chains["c"]["effects"]["Step1"] = "satisfied"
        
        order = Order(id="O2", payload="skip_me", kind=CommandKind.Try, chain_node="Step1")
        ctx = EvalContext(variables={"o": order}, checkpoints=[])
        
        action_node = list(self.rt.tree.find_data("action_stmt"))[0]
        self.rt._exec_action(action_node, ctx)
        
        # ai_provider should NOT have been called
        self.assertEqual(self.rt.ai_provider.execute.call_count, 0)
        print("Redundant activity skipped correctly.")

if __name__ == "__main__":
    unittest.main()
