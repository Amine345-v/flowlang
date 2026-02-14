import sys
import os
import json
from flowlang.runtime import Runtime

def verify_bridge():
    print("=== Jol Studio: Bridge Engineering Certainty Loop Verification ===")
    
    # Ensure we are in the right directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    rt = Runtime(dry_run=False)
    
    try:
        print("[System] Loading 'examples/bridge_engineering.flow'...")
        rt.load("examples/bridge_engineering.flow")
        
        print("[System] Running flow 'build_bridge'...")
        # Running with surveyor, architect, evaluator as teams
        rt.run_flow("build_bridge")
        
        print("\n=== Audit Log ===")
        found_final = False
        for log in rt.console:
            if "Yaqeen (Certainty) Reached" in log:
                found_final = True
            if "[Connector] Executing" in log:
                print(f"  > {log[:120]}...")

        if found_final:
            print("\n[SUCCESS] Certainty Loop completed. All professional workers followed the Current.")
        else:
            print("\n[FAILED] Flow did not reach approval.")
            
    except Exception as e:
        print(f"[Error] Verification failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_bridge()
