from pathlib import Path
from lark import Lark, Tree
from .errors import ParseError

GRAMMAR_PATH = Path(__file__).with_name("grammar.lark")

_parser = None

def _load_parser() -> Lark:
    global _parser
    if _parser is None:
        grammar = GRAMMAR_PATH.read_text(encoding="utf-8")
        _parser = Lark(grammar, start="start", parser="lalr", maybe_placeholders=True)
    return _parser

def parse(source: str | Path) -> Tree:
    try:
        text = Path(source).read_text(encoding="utf-8") if isinstance(source, Path) or (isinstance(source, str) and Path(source).exists()) else str(source)
        parser = _load_parser()
        return parser.parse(text)
    except Exception as e:
        raise ParseError(str(e)) from e
