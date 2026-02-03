from pathlib import Path
from flowlang import Runtime


def test_run_example_flow():
    rt = Runtime()
    src = Path(__file__).resolve().parents[1] / "examples" / "example1.flow"
    rt.load(src)
    rt.run_flow()
    # basic metrics presence
    assert rt.metrics["checkpoints"] >= 1
    assert isinstance(rt.metrics["checkpoint_ms"], dict)
