from flowlang.runtime import Runtime, EvalContext
from flowlang.types import Order, CommandKind, ValueTag
from flowlang.errors import RuntimeFlowError, SemanticError
import sys

def test_strict_team_typing():
    print("--- Test 1: Strict Team Typing ---")
    rt = Runtime(dry_run=True)
    source = 'team t : Command<Search> [size=1]; flow main(using: t) { checkpoint "init" { t.ask("hello"); } }'
    try:
        rt.load(source)
        rt.run_flow("main")
        print("RESULT: FAILED (No error raised)")
    except (RuntimeFlowError, SemanticError) as e:
        print(f"RESULT: PASSED (Caught expected error: {e})")
    except Exception as e:
        print(f"RESULT: ERROR (Unexpected exception: {type(e).__name__}: {e})")

def test_order_batch_processing():
    print("\n--- Test 2: Order Batch Processing & Sequential Distribution ---")
    rt = Runtime(dry_run=True)
    source = 'team t1 : Command<Try> [size=2]; flow main(using: t1) { checkpoint "run" { t1.try(my_items); } }'
    try:
        rt.load(source)
        orders = [
            Order(id="1", payload="task1", kind=CommandKind.Try),
            Order(id="2", payload="task2", kind=CommandKind.Try)
        ]
        ctx = EvalContext(variables={"my_items": orders}, checkpoints=["run"])
        # Find the action_stmt node
        action_node = list(rt.tree.find_data("action_stmt"))[0]
        # Direct execution of _exec_action to verify internals
        rt._exec_action(action_node, ctx)
        
        m1 = orders[0].audit_trail[-1].get("team_member")
        m2 = orders[1].audit_trail[-1].get("team_member")
        
        print(f"Order 1 member: {m1}, Order 2 member: {m2}")
        if m1 != m2 and orders[0].state == "processing":
            print("RESULT: PASSED (Orders distributed sequentially)")
        else:
            print(f"RESULT: FAILED (m1={m1}, m2={m2}, state={orders[0].state})")
    except Exception as e:
        print(f"RESULT: ERROR ({type(e).__name__}: {e})")
        import traceback
        traceback.print_exc()

def test_checkpoint_report_handover():
    print("\n--- Test 3: Checkpoint Report Handover (Unload/Load) ---")
    source = 'team t1 : Command<Search> [size=1]; flow main(using: t1) { checkpoint "step1" (report: ["res"]) { res = "keep me"; temp = "hidden"; } }'
    rt = Runtime(dry_run=True)
    try:
        rt.load(source)
        rt.run_flow("main")
        
        logs = "\n".join(rt.console)
        if "Report handover: kept ['res'], pruned 1 keys" in logs:
            print("RESULT: PASSED (Context pruned correctly)")
        else:
            print("RESULT: FAILED (Pruning logs not found)")
            print("Full logs:", logs)
    except Exception as e:
        print(f"RESULT: ERROR ({type(e).__name__}: {e})")

if __name__ == "__main__":
    test_strict_team_typing()
    test_order_batch_processing()
    test_checkpoint_report_handover()
