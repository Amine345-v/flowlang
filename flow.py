#!/usr/bin/env python
import sys
import argparse
from pathlib import Path
from flowlang import Runtime

def main():
    parser = argparse.ArgumentParser(description="FlowLang CLI - Programming for Professions")
    subparsers = parser.add_subparsers(dest="command")

    # 'run' command
    run_parser = subparsers.add_parser("run", help="Run a .flow file")
    run_parser.add_argument("name", help="Name or path of the flow file (e.g. software_factory or examples/test.flow)")
    run_parser.add_argument("--dry-run", action="store_true", help="Run in simulation mode (no actual AI calls)")

    args = parser.parse_args()

    if args.command == "run":
        target = args.name
        # Heuristic search for the file
        potential_paths = [
            Path(target),
            Path(f"{target}.flow"),
            Path("examples") / target,
            Path("examples") / f"{target}.flow"
        ]
        
        flow_file = None
        for p in potential_paths:
            if p.exists() and p.is_file():
                flow_file = p
                break
        
        if not flow_file:
            print(f"Error: Could not find flow file for '{target}'")
            print("Checked: " + ", ".join(str(p) for p in potential_paths))
            sys.exit(1)

        print(f"[CLI] Running: {flow_file}")
        try:
            rt = Runtime(dry_run=args.dry_run)
            rt.load(flow_file)
            
            # Extract flow name from filename if not specified, 
            # or just run the first one found in the file.
            # For simplicity, we run the flow that matches the name 
            # or the first flow decl found.
            rt.run_flow()
        except Exception as e:
            print(f"[Error] {str(e)}")
            sys.exit(1)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
