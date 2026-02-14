import sys
import os
import json
from flowlang.runtime import Runtime, EvalContext

def insight_demo():
    print("=== Jol Studio: Insight Dashboard & Causal Echo Demo ===")
    
    # Ensure we are in the right directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    rt = Runtime(dry_run=False)
    
    try:
        print("\n[Step 1] Loading and Initial Execution of Bridge Flow...")
        rt.load("examples/bridge_engineering.flow")
        rt.run_flow("build_bridge")
        
        # Simulate exporting the map for the IDE
        # We need to find the latest state to export
        latest_state_path = rt.persistence.get_latest_state("build_bridge")
        if latest_state_path:
            state = rt.persistence.load_state(latest_state_path)
            map_path = rt.persistence.export_map_json("build_bridge", state)
            print(f"[Map] System Tree exported to: {map_path}")
            
            # Read the map to simulate User Insight
            with open(map_path, 'r', encoding='utf-8') as f:
                system_map = json.load(f)
                print(f"[Insight] Project: {system_map['project']}")
                print(f"[Insight] Traces Found: {len(system_map['traces'])}")
                for tr in system_map['traces']:
                    print(f"  - {tr['name']}: {tr['value']} (At: {tr['origin']})")

        print("\n[Step 2] Simulating User Adjustment (The Causal Echo)...")
        print("[User] 'Wait, the soil is actually SOFT_CLAY, not STABLE_ROCK! Update the trace.'")
        
        # We need a context for the echo
        # In a real IDE, this would be triggered via API. 
        # Here we simulate by reaching into the runtime.
        ctx = EvalContext(variables={}, checkpoints=[]) # Minimal ctx for the echo
        
        # Trigger the echo from the 'Analysis' node
        # This should invalidate 'Design' and 'Audit' downstream
        rt.causal_echo("construction_line", "Analysis", 0.1, ctx)
        
        print("\n[Step 3] Re-running the Flow with the new 'Current'...")
        # The flow should now RE-EXECUTE Design and Audit because their states were cleared
        # but it might skip Research if we didn't touch it or if we specifically reset it.
        # rt.run_flow handles resuming/skipping internally based on chain effects.
        rt.run_flow("build_bridge")
        
        print("\n=== Final Analysis ===")
        execution_logs = [log for log in rt.console if "[causal_echo]" in log or "Invalidating downstream" in log]
        for log in execution_logs:
            print(f"  {log}")

        print("\n[SUCCESS] Causal Echo demo completed. The system reacted to the manual trace edit.")

    except Exception as e:
        print(f"[Error] Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    insight_demo()
