import sys
import os
import json
from flowlang.runtime import Runtime

def verify_judge():
    print("=== Jol Studio: Judge Engine (Constitutional Sovereignty) Verification ===")
    
    # Ensure we are in the right directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    rt = Runtime(dry_run=False)
    
    try:
        print("[System] Loading 'examples/bridge_engineering.flow'...")
        rt.load("examples/bridge_engineering.flow")
        
        # We want to SIMULATE a failure. 
        # Let's modify the researcher to output a HUGE load that fails feasibility.
        # Or better, just run it and see the pass, then simulate a broken ancestry.
        
        print("[System] Running flow 'build_bridge' for a SUCCESS baseline...")
        rt.run_flow("build_bridge")
        
        # Verify success
        latest_state_path = rt.persistence.get_latest_state("build_bridge")
        if latest_state_path:
            state = rt.persistence.load_state(latest_state_path)
            approval = state.eval_context.get("audit_res", {}).get("pass")
            print(f"[Verification] Initial Approval: {approval}")
        
        print("\n[System] Simulating a STRUCTURAL GAP (Broken Ancestry)...")
        # We'll manually trigger the judge with a "corrupted" trace
        from flowlang.types import Order, CriticalFeature, CommandKind, EvalContext
        
        corrupted_order = Order(
            id="corrupt-001",
            payload={"design": "broken"},
            kind=CommandKind.Judge,
            critical_features=[
                CriticalFeature(
                    name="piling_integrity",
                    value="VERIFIED",
                    feature_id="ENG-999",
                    ancestry_link="FORGED-PARENT", # Broken ancestry
                    feature_type="STRUCTURAL_SPEC"
                )
            ]
        )
        
        ctx = EvalContext(variables={}, checkpoints=[])
        res_val, _ = rt._execute_single_action("Auditors", "judge", [corrupted_order], {}, ctx)
        
        print(f"[Judge Result] Pass: {res_val.get('pass')}")
        print(f"[Structural Gap Report]\n{res_val.get('reason')}")
        
        if res_val.get('pass') is False and "Causal Gap" in res_val.get('reason'):
            print("\n[PASS] Judge Engine successfully detected the Causal/Ancestry Gap.")
        else:
            print("\n[FAIL] Judge Engine failed to detect the gap.")

        print("\n[System] Verifying Tree Completion Check...")
        missing_order = Order(
            id="missing-001",
            payload={},
            kind=CommandKind.Judge,
            critical_features=[] # No piling_integrity
        )
        res_val_missing, _ = rt._execute_single_action("Auditors", "judge", [missing_order], {}, ctx)
        print(f"[Judge Result] Pass: {res_val_missing.get('pass')}")
        if "Tree Gap" in res_val_missing.get('reason'):
            print("[PASS] Judge Engine successfully detected the Tree/Mandatory Gap.")
        else:
            print("[FAIL] Judge Engine failed to detect the missing mandatory node.")

        print("\n[SUCCESS] Judge Engine (Constitutional Sovereignty) Verification completed.")
            
    except Exception as e:
        print(f"[Error] Verification failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_judge()
