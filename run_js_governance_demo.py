from flowlang.runtime import Runtime
import json
import os

def run_js_demo():
    print("=== FlowLang: JavaScript Governance Demo ===")
    print("[Law] FlowLang governs the JS Legal Expert.")
    
    rt = Runtime(dry_run=False) # We need actual execution for the connector
    
    try:
        print("\n[System] Loading 'js_governance.flow'...")
        rt.load("examples/js_governance.flow")
        
        print("[System] Executing 'js_audit_flow'...")
        rt.run_flow("js_audit_flow")
        
        # Check results
        print("\n=== Result Inspection ===")
        # The result of the judge call should be in the log
        for log_entry in rt.console:
            if "[legal_js" in log_entry:
                print(f"[Audit Log] {log_entry}")

        print("\n[Success] JS Worker was successfully governed by FlowLang.")
        
    except Exception as e:
        print(f"\n[Error] Demo failed: {str(e)}")

if __name__ == "__main__":
    run_js_demo()
