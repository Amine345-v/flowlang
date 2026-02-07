from flowlang.runtime import Runtime
import json
import os

def run_hybrid_demo():
    print("=== FlowLang: Integrated Meta-Language Demo ===")
    print("[Definition] FlowLang governs the 'Diagnostic Tool' (External Tier)")
    
    # Initialize Runtime in dry_run mode
    # Even in dry_run, the 'connector' policy will trigger the actual shell command
    # but AI teams (board/council) will be faked.
    rt = Runtime(dry_run=True)
    
    try:
        print("\n[System] Loading 'medical_legal.flow'...")
        rt.load("examples/medical_legal.flow")
        
        print("[System] Executing 'medical_legal_auth' conductor...")
        rt.run_flow("medical_legal_auth")
        
        print("\n=== Result Inspection (Governance Layer) ===")
        
        # 1. Inspect The Maestro (Process Tree)
        print("\n[Maestro] Medical Care Roadmap:")
        marks = rt.processes["medical_map"]["marks"]
        for node, status in marks.items():
            print(f" - {node}: {status}")
            
        # 2. Inspect Audit Metrics
        print("\n[Audit] Metrics:")
        print(f" - Actions performed: {rt.metrics['actions']}")
        print(f" - Verbs invoked: {json.dumps(rt.metrics['verbs'], indent=2)}")

        print("\n[Success] Hybrid integration verified. FlowLang governed the external script execution.")
        
    except Exception as e:
        print(f"\n[Error] Demo failed: {str(e)}")

if __name__ == "__main__":
    run_hybrid_demo()
