import pytest
from flowlang.runtime import Runtime
from flowlang.types import Order, CommandKind, ValueTag
from flowlang.errors import RuntimeFlowError

def test_strict_team_typing():
    rt = Runtime(dry_run=True)
    source = """
    team t : Command<Search> [size=1];
    flow main(using: t) {
        checkpoint "init" {
            t.ask("hello"); 
        }
    }
    """
    rt.load(source)
    with pytest.raises(RuntimeFlowError, match="is specialized for Search, cannot perform 'ask'"):
        rt.run_flow("main")

def test_order_batch_processing():
    rt = Runtime(dry_run=True)
    source = """
    team t1 : Command<Try> [size=1];
    flow main(using: t1) {
        checkpoint "run" {
            v = t1.try(my_items);
        }
    }
    """
    rt.load(source)
    # Inject orders into variables manually for the test
    orders = [
        Order(id="1", payload="task1", kind=CommandKind.Try),
        Order(id="2", payload="task2", kind=CommandKind.Try)
    ]
    
    # We need a way to inject variables. Let's hijack _execute_flow to use our ctx.
    # Actually, we can just run a flow that sets them, or use a mock.
    # Let's just manually call _exec_action to test the logic.
    from flowlang.runtime import EvalContext
    ctx = EvalContext(variables={"my_items": orders}, checkpoints=["run"])
    # Get the action node from the tree
    action_node = list(rt.tree.find_data("action_stmt"))[0]
    rt._exec_action(action_node, ctx)
    
    # Check if orders were updated
    assert orders[0].state == "processing"
    assert len(orders[0].audit_trail) > 0
    assert isinstance(ctx.variables["_"].value, list)

def test_checkpoint_report_handover():
    source = """
    team t1 : Command<Search> [size=1];
    flow main(using: t1) {
        checkpoint "step1" (report: ["res"]) {
            temp = "hidden";
            res = "keep me";
        }
        checkpoint "step2" {
            // temp should be gone here
        }
    }
    """
    rt = Runtime(dry_run=True)
    rt.load(source)
    rt.run_flow("main")
    
    # Check console for pruning log
    assert any("pruned 1 keys" in l for l in rt.console)
    assert any("kept ['res']" in l for l in rt.console)

def test_orders_promotion():
    rt = Runtime(dry_run=True)
    rt.load("""
    team t1 : Command<Search> [size=1];
    flow main(using: t1) {
        checkpoint "start" {
            v = t1.search("query");
        }
    }
    """)
    rt.run_flow("main")
    v = rt.persistence.load_state(rt.console[-1].split(" ")[-1]).eval_context.get("v") 
    # Actually simpler: check rt.metrics or logs
    assert any("[set] v = Order(id='order_1'" in str(l) for l in rt.console)
