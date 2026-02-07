def normalize_indent(content):
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        stripped = line.lstrip()
        if not stripped:
            new_lines.append("")
            continue
        
        indent_count = len(line) - len(stripped)
        # Heuristic: convert 5-space indents to 4-space if they are at multiples of 5-ish
        # Actually, let's just use a simpler heuristic for this specific file.
        # We know the base indent should be 4.
        # 4, 8, 12, 16, 20, 24, 28...
        # If it's 5, 10, 15, 20, 25... we map them.
        
        # New strategy: find the level by dividing by the most likely indent width.
        # If it's closer to 4*N, use 4*N. If it's closer to 5*N, it's probably meant to be 4*N.
        
        level = round(indent_count / 4)
        new_indent = " " * (level * 4)
        new_lines.append(new_indent + stripped)
        
    return '\n'.join(new_lines)

with open('flowlang/runtime.py', 'r', encoding='utf8') as f:
    content = f.read()

normalized = normalize_indent(content)

with open('flowlang/runtime.py', 'w', encoding='utf8') as f:
    f.write(normalized)

print("Indentation normalized.")
