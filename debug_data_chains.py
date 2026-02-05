from flowlang.runtime import Runtime, EvalContext
from flowlang.types import Order, CommandKind, ValueTag
import sys

def test_exclusive_activity():
    print("--- Test 1: Exclusive Activity (Manual Touch) ---")
    rt = Runtime(dry_run=True)
    source = """
    team t : Command<Try> [size=1];
    chain c { nodes: [n1, n2]; propagation: causal(decay=1.0, forward=true); }
    flow main(using: t) { 
        checkpoint "init" { 
            t.try(my_order); 
        } 
    }
    """
    rt.load(source)
    print(rt.tree.pretty())
    
    # 1. Manually mark n1 as satisfied
    rt.chains["c"]["effects"]["n1"] = "satisfied"
    
    order = Order(id="order1", payload="task1", kind=CommandKind.Try, chain_node="n1")
    ctx = EvalContext(variables={"my_order": [order]}, checkpoints=["init"])
    
    # Execute action
    action_node = list(rt.tree.find_data("action_stmt"))[0]
    rt._exec_action(action_node, ctx)
    
    # Check if it was skipped
    if any("Exclusive Activity: Skipping n1" in l for l in rt.console):
        print("RESULT: PASSED (Action skipped as expected)")
    else:
        print("RESULT: FAILED (Action was not skipped)")
        # print("Logs:", rt.console)

def test_causal_propagation():
    print("\n--- Test 2: Causal Propagation (Guiding Thread) ---")
    rt = Runtime(dry_run=True)
    source = """
    team t : Command<Try> [size=1];
    chain c { nodes: [n1, n2]; propagation: causal(decay=1.0, forward=true); }
    flow main(using: t) { 
        checkpoint "init" { 
            t.try(order1);
            t.try(order2);
        }
    }
    """
    rt.load(source)
    
    # Link orders to chain nodes
    o1 = Order(id="O1", payload="Task1", kind=CommandKind.Try, chain_node="n1")
    o2 = Order(id="O2", payload="Task2", kind=CommandKind.Try, chain_node="n2")
    
    ctx = EvalContext(variables={"order1": [o1], "order2": [o2]}, checkpoints=["init"])
    
    # The first action (O1) will run and implicitly touch n1
    # n1 touch will propagate "satisfied" to n2 (decay=1.0, fwd=true)
    # The second action (O2) should then skip n2
    
    rt.run_flow("main")
    
    logs = "\n".join(rt.console)
    if "[chain] c.propagate n1 effect=satisfied" in logs and "Exclusive Activity: Skipping n2" in logs:
        print("RESULT: PASSED (Causal propagation triggered skip on downstream node)")
    else:
        print("RESULT: FAILED")
        print("Logs:", logs)

if __name__ == "__main__":
    test_exclusive_activity()
    test_causal_propagation()
