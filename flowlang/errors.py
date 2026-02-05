class FlowLangError(Exception):
    pass

class ParseError(FlowLangError):
    pass

class SemanticError(FlowLangError):
    pass

class RuntimeFlowError(FlowLangError):
    pass

class SchemaValidationError(FlowLangError):
    """Raised when AI response fails schema validation."""
    pass

class ContextOverflowError(FlowLangError):
    """Raised when context exceeds size limits."""
    pass

class HumanGateTimeoutError(FlowLangError):
    """Raised when human confirmation times out."""
    pass
