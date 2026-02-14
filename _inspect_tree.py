"""Quick script to inspect command_action parse tree structure."""
from flowlang.parser import parse

src = """
team T: Command<Search> [size=1];
flow F(using: T) {
    checkpoint "C1" {
        T.search("q");
        x = T.search("q2");
    }
}
"""
tree = parse(src)
print("=== Full tree ===")
print(tree.pretty())
print()

for act in tree.find_data("command_action"):
    print("=== command_action ===")
    print(f"  children count: {len(act.children)}")
    for i, c in enumerate(act.children):
        t = type(c).__name__
        if t == "Token":
            print(f"  [{i}] Token type={c.type} value={c!r}")
        else:
            print(f"  [{i}] Tree data={c.data} children={len(c.children)}")

for act in tree.find_data("action_stmt"):
    print("=== action_stmt ===")
    print(f"  children count: {len(act.children)}")
    for i, c in enumerate(act.children):
        t = type(c).__name__
        if t == "Token":
            print(f"  [{i}] Token type={c.type} value={c!r}")
        else:
            print(f"  [{i}] Tree data={c.data} children={len(c.children)}")
