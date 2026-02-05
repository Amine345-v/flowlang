from flowlang.runtime import Runtime
import json
import os

def run_case_study():
    print("=== FlowLang Case Study: Cybersecurity Compliance Auditor ===")
    
    # Initialize Runtime in dry_run mode to simulate the professional logic
    rt = Runtime(dry_run=True)
    
    # Load the professional flow
    try:
        print("[System] Loading 'security_audit.flow'...")
        rt.load("examples/security_audit.flow")
        
        print("[System] Executing 'compliance_audit' conductor...")
        rt.run_flow("compliance_audit")
        
        print("\n=== Professional State Inspection ===")
        
        # 1. Inspect The Maestro (Process Tree)
        print("\n[Maestro] Compliance Roadmap Status:")
        mfa_status = rt.processes["compliance_map"]["marks"].get("MFA")
        print(f" - Node 'MFA': {mfa_status}")
        
        # 2. Inspect The Guiding Thread (Data Chain)
        print("\n[Chain] Audit Pipeline Causal Hits:")
        pipeline_effects = rt.chains["audit_pipeline"]["effects"]
        for node, effect in pipeline_effects.items():
            print(f" - Stage {node}: {effect}")
            
        # 3. Inspect the Order Audit trail
        print("\n[Audit] Last Order Trace:")
        # Find the last completed order
        if rt.metrics["actions"] > 0:
            last_order_id = f"order_{rt.metrics['actions'] - 1}"
            # In a real run, we'd fetch the actual Order object. 
            # Here we just show the metric.
            print(f" - Total professional actions logged: {rt.metrics['actions']}")
            print(f" - Verbs used: {json.dumps(rt.metrics['verbs'], indent=2)}")

        print("\n[Success] Case study execution simulation completed.")
        
    except Exception as e:
        print(f"[Error] Execution failed: {str(e)}")

if __name__ == "__main__":
    run_case_study()
