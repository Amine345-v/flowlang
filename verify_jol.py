import sys
import os
import json
from flowlang.runtime import Runtime

def verify():
    print("=== Jol Studio: Trace Propagation Verification ===")
    
    # Ensure we are in the right directory to find the flow file
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    rt = Runtime(dry_run=False)
    
    try:
        print("[System] Loading 'verify_jol.flow'...")
        rt.load("verify_jol.flow")
        
        print("[System] Running flow 'verify_traces'...")
        rt.run_flow("verify_traces")
        
        # Check logs for traces
        print("\n=== Log Audit ===")
        found_trace_in_cmd = False
        for log in rt.console:
            if "[Connector] Executing" in log and "critical_features" in log:
                print(f"[FOUND] Commanding Trace detected in shell command: {log[:100]}...")
                found_trace_in_cmd = True
            if "mechanical_integrity" in log:
                print(f"[FOUND] Worker output contained Critical Features: {log}")

        if found_trace_in_cmd:
            print("\n[SUCCESS] Commanding Traces (الأثر الأمري) successfully propagated between workers.")
        else:
            print("\n[FAILED] Traces were not found in the connector calls.")
            
    except Exception as e:
        print(f"[Error] Verification failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
