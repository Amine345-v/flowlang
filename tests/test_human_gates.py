import os
import sys
sys.path.insert(0, os.getcwd())
import pytest
from unittest.mock import patch, MagicMock
from flowlang.runtime import Runtime, EvalContext
from lark import Tree, Token

class TestHumanGates:
    
    def test_confirm_approved(self):
        rt = Runtime()
        ctx = EvalContext(variables={}, checkpoints=[])
        
        # confirm("Proceed?") -> ans
        # context: confirm ( prompt, ...kw, target )
        # children: [Token(STRING "Proceed?"), Token(IDENT ans)]
        # Or named args?
        # Grammar: confirm ( STRING ( named_arg )* ) -> IDENT
        # children: Prompt, [named_arg...], Target
        
        node = Tree("confirm_stmt", [
            Token("STRING", '"Proceed?"'), 
            Token("IDENT", "ans")
        ])
        
        # Mock input to return 'y'
        with patch('builtins.input', return_value='y'), patch('builtins.print'):
            rt._exec_confirm_stmt(node, ctx)
            
        assert ctx.variables["ans"] is True
        
    def test_confirm_rejected(self):
        rt = Runtime()
        ctx = EvalContext(variables={}, checkpoints=[])
        
        node = Tree("confirm_stmt", [
            Token("STRING", '"Proceed?"'), 
            Token("IDENT", "ans")
        ])
        
        with patch('builtins.input', return_value='n'), patch('builtins.print'):
            rt._exec_confirm_stmt(node, ctx)
            
        assert ctx.variables["ans"] is False

    def test_confirm_auto_approve(self):
        rt = Runtime()
        ctx = EvalContext(variables={}, checkpoints=[])
        node = Tree("confirm_stmt", [Token("STRING", '"?"'), Token("IDENT", "ans")])
        
        with patch.dict(os.environ, {"FLOWLANG_AUTO_APPROVE": "1"}):
            rt._exec_confirm_stmt(node, ctx)
            
        assert ctx.variables["ans"] is True

    def test_dry_run_action_skipped(self):
        rt = Runtime(dry_run=True)
        ctx = EvalContext(variables={}, checkpoints=[])
        
        # New grammar structure: action_stmt -> [IDENT, command_action -> [ask_action -> [arg_list?]]]
        ask_node = Tree("ask_action", [Tree("arg_list", [])])
        action_node = Tree("command_action", [ask_node])
        node = Tree("action_stmt", [Token("IDENT", "TeamA"), action_node])
        
        rt._exec_action(node, ctx)
        
        # In dry_run mode, result is an Order object
        res = ctx.variables.get("_")
        from flowlang.types import Order
        assert isinstance(res, Order)
        assert any("[dry_run]" in str(l) for l in rt.console)
