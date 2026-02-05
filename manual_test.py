import sys
import os
import traceback
from unittest.mock import patch

# Setup path
sys.path.insert(0, os.getcwd())

from flowlang.runtime import Runtime, EvalContext
from flowlang.errors import ContextOverflowError
from lark import Tree

def test_max_context_size_limit():
    print("Running test_max_context_size_limit...", end="")
    rt = Runtime()
    ctx = EvalContext(variables={}, checkpoints=[])
    limit = 10
    with patch.dict(os.environ, {"FLOWLANG_MAX_CONTEXT_SIZE": str(limit)}):
        for i in range(limit + 1):
            ctx.variables[f"k{i}"] = i
        
        node = Tree("context_stmt", ["update"])
        try:
            rt._exec_context_stmt(node, ctx)
            print(" FAILED (did not raise ContextOverflowError)")
        except ContextOverflowError:
            print(" PASSED")
        except Exception as e:
            print(f" FAILED (raised {type(e).__name__}: {e})")

def test_prune_allowed_when_full():
    print("Running test_prune_allowed_when_full...", end="")
    rt = Runtime()
    ctx = EvalContext(variables={}, checkpoints=[])
    limit = 5
    with patch.dict(os.environ, {"FLOWLANG_MAX_CONTEXT_SIZE": str(limit)}):
        for i in range(limit + 5):
            ctx.variables[f"k{i}"] = i
        
        # prune("k0")
        node = Tree("context_stmt", ["prune", Tree("expr_list", [Tree("expr", ["k0"])])])
        
        with patch.object(rt, "_eval_expr", return_value=["k0"]): # prune logic iterates over list if list, or single value
            # In runtime.py:
            # val = self._eval_expr(e, ctx)
            # if isinstance(val, list): ... else: keep_keys.add(str(val))
            # My logic in runtime:
            # if isinstance(args_tree, Tree) and args_tree.data == "expr_list":
            #    for e in args_tree.children:
            #        val = self._eval_expr(e, ctx)
            
            # So I should mock return 'k0' for the expression
            with patch.object(rt, "_eval_expr", return_value="k0"):
                rt._exec_context_stmt(node, ctx)
            
        if len(ctx.variables) == 1 and "k0" in ctx.variables:
            print(" PASSED")
        else:
            print(f" FAILED (len={len(ctx.variables)}, keys={list(ctx.variables.keys())})")

def test_snapshot_allowed_when_full():
    print("Running test_snapshot_allowed_when_full...", end="")
    rt = Runtime()
    ctx = EvalContext(variables={"a": 1}, checkpoints=[])
    limit = 1
    with patch.dict(os.environ, {"FLOWLANG_MAX_CONTEXT_SIZE": str(limit)}):
        ctx.variables["b"] = 2
        
        node = Tree("context_stmt", ["snapshot", Tree("expr_list", [Tree("expr", ["backup"])])])
        
        with patch.object(rt, "_eval_expr", return_value="backup"):
            rt._exec_context_stmt(node, ctx)
            
    if "__snapshot_backup__" in ctx.variables:
        print(" PASSED")
    else:
        print(" FAILED (snapshot not found)")

def test_bounded_deep_merge_limits():
    print("Running test_bounded_deep_merge_limits...", end="")
    rt = Runtime()
    base = EvalContext(variables={"logs": [i for i in range(100)]}, checkpoints=[], merge_policy="bounded_deep")
    other = EvalContext(variables={"logs": [i for i in range(50)]}, checkpoints=[])
    
    rt._merge_contexts(base, other)
    
    # Check
    log_len = len(base.variables["logs"])
    last_item = base.variables["logs"][-1]
    
    if log_len == 100 and last_item == 49:
        print(" PASSED")
    else:
        print(f" FAILED (len={log_len}, last={last_item})")

def test_bounded_deep_merge_recursion():
     print("Running test_bounded_deep_merge_recursion...", end="")
     rt = Runtime()
     deep = {}
     curr = deep
     for i in range(60):
         curr["next"] = {}
         curr = curr["next"]
         
     a = {"root": deep}
     b = {"root": {"extra": 1}}
     
     try:
         rt._bounded_deep_merge(a, b)
         print(" PASSED")
     except RecursionError:
         print(" FAILED (RecursionError)")
     except Exception as e:
         print(f" FAILED ({e})")

if __name__ == "__main__":
    try:
        test_max_context_size_limit()
        test_prune_allowed_when_full()
        test_snapshot_allowed_when_full()
        test_bounded_deep_merge_limits()
        test_bounded_deep_merge_recursion()
    except Exception:
        traceback.print_exc()
