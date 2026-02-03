class FlowLangError(Exception):
    pass

class ParseError(FlowLangError):
    pass

class SemanticError(FlowLangError):
    pass

class RuntimeFlowError(FlowLangError):
    pass
