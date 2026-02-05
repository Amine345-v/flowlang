import sys
from pathlib import Path

# Allow running as script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flowlang import Runtime


def main():
    if "--lint" in sys.argv:
        # VS Code Linting mode
        # The extension might send the content via stdin
        content = sys.stdin.read()
        try:
            rt = Runtime(dry_run=True)
            # Load from string to perform semantic checks
            rt.load(content)
            # If load succeeds, no errors found
            sys.exit(0)
        except Exception as e:
            # Format: "line 5, col 10: Error message"
            # Lark exceptions usually have line and column
            line = getattr(e, 'line', 1)
            column = getattr(e, 'column', 1)
            print(f"line {line}, col {column}: {str(e)}")
            sys.exit(0) # Exit cleanly so stdout is captured

    if len(sys.argv) < 2:
        print("Usage: python scripts/run.py <program.flow> [flow_name]")
        sys.exit(1)
    
    src = Path(sys.argv[1])
    flow_name = sys.argv[2] if len(sys.argv) > 2 else None
    rt = Runtime()
    rt.load(src)
    rt.run_flow(flow_name)


if __name__ == "__main__":
    main()
