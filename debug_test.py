import sys
import os
sys.path.insert(0, os.getcwd())
try:
    from flowlang.parser import parse
    print("Imports OK")
    
    # EXACT string from test
    stmt = 'context.update("overflow")'
    src = f"""
        flow Dummy(using: T) {{
            checkpoint "start" {{
                {stmt};
            }}
        }}
        """
    print(f"Parsing:\n{src}")
    tree = parse(src)
    print("Parse OK")
    print(tree.pretty())
except Exception as e:
    print(f"Error: {e}")
