from flowlang.parser import parse
source = 'team t : Command<Search> [size=1]; flow main(using: t) { checkpoint "init" { } t.ask("hello"); }'
print(f"Parsing source: {repr(source)}")
try:
    tree = parse(source)
    print("Parse successful")
    print(tree.pretty())
except Exception as e:
    print(f"Parse failed: {e}")
