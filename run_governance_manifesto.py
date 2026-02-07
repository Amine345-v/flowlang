from flowlang.runtime import Runtime
import json
import os

def run_governance_demo():
    print("=== FlowLang: Professional Governance Manifesto ===")
    print("[Manifesto] FlowLang is the Constitution. Workers are the Executive.")
    
    rt = Runtime(dry_run=False)
    
    try:
        print("\n[System] Loading 'governance_manifesto.flow'...")
        rt.load("examples/governance_manifesto.flow")
        
        print("[System] Executing 'show_governance'...")
        rt.run_flow("show_governance")
        
        print("\n=== Governance Audit (The Conductor's Verdict) ===")
        found_violation = False
        for log_entry in rt.console:
            if "[Governance]" in log_entry:
                print(f"!!! {log_entry}")
                found_violation = True
            elif "[legal_js" in log_entry:
                print(f" - Result from JS Worker: {log_entry}")

        if found_violation:
            print("\n[SUCCESS] FlowLang successfully enforced the 'Strict' Law on the JS worker.")
        else:
            print("\n[ERROR] Governance violation was not detected.")
            
    except Exception as e:
        print(f"\n[Error] Demo failed: {str(e)}")

if __name__ == "__main__":
    run_governance_demo()
