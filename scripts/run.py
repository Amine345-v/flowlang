import sys
from pathlib import Path

# Allow running as script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flowlang import Runtime


def main():
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
