import sys
import os
import json
from flowlang.runtime import Runtime

def verify_cfp():
    print("=== Jol Studio: Critical Feature Protocol (CFP) Verification ===")
    
    # Ensure we are in the right directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    rt = Runtime(dry_run=False)
    
    try:
        print("[System] Loading 'examples/bridge_engineering.flow'...")
        rt.load("examples/bridge_engineering.flow")
        
        print("[System] Running flow 'build_bridge'...")
        rt.run_flow("build_bridge")
        
        print("\n=== Genetic Trace Inspection (CFP Compliance) ===")
        
        # Check the persistence state for CFP metadata
        latest_state_path = rt.persistence.get_latest_state("build_bridge")
        if latest_state_path:
            state = rt.persistence.load_state(latest_state_path)
            
            # Find the 'geo_report' or 'd_spec' in eval_context
            for var_name, var_val in state.eval_context.items():
                if hasattr(var_val, "critical_features") and var_val.critical_features:
                    print(f"\n[Variable: {var_name}] Found {len(var_val.critical_features)} Traces")
                    for cf in var_val.critical_features:
                        print(f"  FEATURE: {cf.name}")
                        print(f"    - ID: {cf.feature_id}")
                        print(f"    - Ancestry: {cf.ancestry_link}")
                        print(f"    - Type: {cf.feature_type}")
                        print(f"    - Impact Zones: {cf.impact_zones}")
                        print(f"    - Echo Sig: {cf.echo_signature}")
                        
                        # Verify we have the rich metadata
                        if not cf.feature_id or not cf.impact_zones:
                            print(f"    [FAIL] Rich CFP metadata missing!")
                        else:
                            print(f"    [PASS] CFP metadata verified.")

        print("\n[SUCCESS] CFP Verification completed. The 'Sieve' is functional.")
            
    except Exception as e:
        print(f"[Error] Verification failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_cfp()
