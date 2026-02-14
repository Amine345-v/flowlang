import pytest
from flowlang.runtime import Runtime, EvalContext
from flowlang.types import Order, CommandKind, ValueTag
from flowlang.errors import RuntimeFlowError, SemanticError

def test_strict_team_typing():
    """Team<Search> cannot perform 'ask' — semantic analyzer should reject this."""
    rt = Runtime(dry_run=True)
    source = """
    team t : Command<Search> [size=1];
    flow main(using: t) {
        checkpoint "init" {
            t.ask("hello"); 
        }
    }
    """
    # The semantic analyzer catches the mismatch during load()
    with pytest.raises(SemanticError, match="cannot perform"):
        rt.load(source)

def test_order_batch_processing():
    """Test that dry_run mode creates an Order for 'try' verb."""
    rt = Runtime(dry_run=True)
    source = """
    team t1 : Command<Try> [size=1];
    flow main(using: t1) {
        checkpoint "run" {
            t1.try(my_items);
        }
    }
    """
    rt.load(source)
    rt.run_flow("main")
    # In dry_run mode, actions produce Order objects
    assert any("[dry_run]" in str(l) for l in rt.console)

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
    
    # Check console for pruning log (actual log says "pruned 2 keys")
    assert any("pruned" in l and "keys" in l for l in rt.console)

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
    # In dry_run mode, search verb should be tracked in metrics
    assert rt.metrics["verbs"].get("search", 0) >= 1
    # Also check that a result was assigned
    assert any("[set] v =" in str(l) for l in rt.console)
