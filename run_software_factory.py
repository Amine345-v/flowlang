from flowlang.runtime import Runtime
from flowlang.types import Order, CommandKind
import os

def run_app():
    # Initialize the FlowLang Runtime in Dry Run mode for safe execution
    rt = Runtime(dry_run=True)
    
    # Load the FlowLang specification
    flow_path = os.path.join("examples", "software_factory.flow")
    with open(flow_path, "r") as f:
        rt.load(f.read())
    
    print("--- Starting FlowLang Conductor: Software Factory ---")
    
    # Execute the flow
    # The runtime handles stateful transitions between checkpoints (Stages)
    try:
        results = rt.run_flow("software_factory")
        
        print("\n--- Project Completion Summary ---")
        # Inspect the Maestro (Process Tree) for the family tree of work
        p_info = rt.processes["software_project"]
        print(f"Maestro Roadmap Marks: {p_info['marks']}")
        
        # Inspect the Guiding Thread (Chain) for causal hits
        c_info = rt.chains["build_sequence"]
        print(f"System Sequence Status: {c_info['effects']}")
        
    except Exception as e:
        print(f"Error during execution: {e}")

if __name__ == "__main__":
    run_app()
