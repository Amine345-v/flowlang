from lark import Lark
try:
    with open('minimal.lark', 'r', encoding='utf-8') as f:
        grammar = f.read()
    parser = Lark(grammar, start='start', parser='lalr')
    source = 'team t : Command<Search> [size=1]; flow main(using: t) { checkpoint "init" { } t.ask("hello"); }'
    tree = parser.parse(source)
    print("Minimal Parse successful")
    print(tree.pretty())
except Exception as e:
    print(f"Minimal Parse failed: {e}")
