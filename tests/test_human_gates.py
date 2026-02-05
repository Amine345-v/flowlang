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
        
        # action: Team.Do(args)
        # Tree structure for _exec_action:
        # children: [Team, ActionTree]
        # ActionTree children: [Verb, ArgList]
        
        action_node = Tree("command_action", [Token("IDENT", "ask"), Tree("arg_list", [])])
        node = Tree("action_stmt", [Token("IDENT", "TeamA"), action_node])
        
        rt._exec_action(node, ctx)
        
        # Result should be Unknown/dry_run
        res = ctx.variables.get("_")
        assert res.tag.name == "Unknown"
        assert res.meta["text"] == "dry_run"
