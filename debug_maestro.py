from flowlang.runtime import Runtime, EvalContext
from flowlang.types import Order, CommandKind, ValueTag
import sys

def test_maestro_search():
    print("--- Test 1: Maestro Hierarchical Search (Binary Path) ---")
    rt = Runtime(dry_run=True)
    source = """
    process p "Product Roadmap" {
        root: "Product";
        branch "Product" -> ["Core", "UI"];
        branch "Core" -> ["Engine", "Parser"];
        node "Engine" { type: "system"; };
    }
    team t : Command<Try> [size=1];
    flow main(using: t) { 
        checkpoint "init" { 
            t.try("hello");
        }
    }
    """
    rt.load(source)
    
    # Test 'find' operation
    ctx = EvalContext(variables={}, checkpoints=[])
    path = rt._process_call("p", "find", ["Engine"], {}, ctx)
    
    # Expected path for Engine: Product -> Core (0) -> Engine (0) => "00"
    print(f"Path found: {path}")
    if path == "00":
        print("RESULT: PASSED")
    else:
        print(f"RESULT: FAILED (Expected '00', got '{path}')")

def test_maestro_auditing():
    print("\n--- Test 2: Maestro Auditing (Order -> Process Node) ---")
    rt = Runtime(dry_run=True)
    source = """
    process p "Product Roadmap" {
        root: "Product";
        node "UI" { state: "pending"; };
    }
    team t : Command<Try> [size=1];
    flow main(using: t) { 
        checkpoint "init" { 
            t.try(my_order);
        }
    }
    """
    rt.load(source)
    
    # Create order linked to a process node
    order = Order(id="UI_Task", payload="Fix button", kind=CommandKind.Try, process_node="UI")
    ctx = EvalContext(variables={"my_order": [order]}, checkpoints=["init"])
    
    # Execute action
    action_node = list(rt.tree.find_data("action_stmt"))[0]
    rt._exec_action(action_node, ctx)
    
    # Check if process mark was updated
    mark = rt.processes["p"]["marks"].get("UI")
    print(f"Mark: {mark}")
    if mark and "Accomplished" in mark:
        print("RESULT: PASSED (Maestro audit synced successfully)")
    else:
        print("RESULT: FAILED")

if __name__ == "__main__":
    test_maestro_search()
    test_maestro_auditing()
