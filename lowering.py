from __future__ import annotations
from typing import Any, Dict, List, Optional
from lark import Tree, Token

from .ir import (
    ProgramIR, FlowIR, CheckpointIR, OpIR,
    ActionOp, ContextUpdateOp, ChainTouchOp, SystemCallOp, DeployOp, FlowControlOp,
    ParBlockOp, RaceBlockOp,
)

class Lowering:
    def __init__(self, tree: Tree):
        self.tree = tree

    def lower(self, flow_name: Optional[str] = None) -> ProgramIR:
        flows: List[FlowIR] = []
        for flow in self.tree.find_data("flow_decl"):
            name = str(flow.children[0])
            if flow_name and name != flow_name:
                continue
            flows.append(self._lower_flow(flow))
        return ProgramIR(flows=flows)

    def _lower_flow(self, flow: Tree) -> FlowIR:
        name = str(flow.children[0])
        teams: List[str] = []
        first_ident_list = next(flow.find_data("ident_list"))
        for ch in first_ident_list.children:
            if isinstance(ch, Token):
                teams.append(str(ch))
        # merge_policy
        merge_policy = "last_wins"
        for hdr in flow.find_data("flow_header"):
            seen_key = False
            for ch in hdr.children:
                if isinstance(ch, Token):
                    sval = str(ch)
                    if seen_key:
                        if sval in ("last_wins", "deep_merge", "crdt"):
                            merge_policy = sval
                            seen_key = False
                            break
                        seen_key = False
                    if sval == "merge_policy":
                        seen_key = True
        cps: List[CheckpointIR] = []
        for cp in flow.find_data("checkpoint"):
            cp_name = cp.children[0].value
            ops = self._lower_block(cp)
            cps.append(CheckpointIR(name=cp_name, ops=ops))
        return FlowIR(name=name, teams=teams, merge_policy=merge_policy, checkpoints=cps)

    def _lower_block(self, node: Tree) -> List[OpIR]:
        ops: List[OpIR] = []
        # iterate direct local_stmt under node
        for lst in node.find_data("local_stmt"):
            child = lst.children[0]
            if isinstance(child, Tree):
                dt = child.data
                if dt == "action_stmt":
                    ops.append(self._lower_action(child))
                elif dt == "context_stmt":
                    ops.append(self._lower_context(child))
                elif dt == "chain_touch_stmt":
                    ops.append(self._lower_chain_touch(child))
                elif dt == "system_call_stmt":
                    ops.append(self._lower_system_call(child))
                elif dt == "deploy_stmt":
                    ops.append(self._lower_deploy(child))
                elif dt == "flow_control":
                    ops.append(self._lower_flow_control(child))
                elif dt == "if_stmt":
                    # normalize: if -> par of then/else branches
                    then_ops = self._lower_block(child.children[1])
                    else_ops = self._lower_block(child.children[2]) if len(child.children) > 2 else []
                    branches = [then_ops]
                    if else_ops:
                        branches.append(else_ops)
                    ops.append(ParBlockOp(branches=branches))
                elif dt == "par_stmt":
                    ops.append(self._lower_par(child))
                elif dt == "race_stmt":
                    ops.append(self._lower_race(child))
                else:
                    # ignore var_assign at IR stage (evaluated at runtime context), could be lifted later
                    pass
        return ops

    def _lower_action(self, node: Tree) -> ActionOp:
        team = str(node.children[0])
        cmd = node.children[1]
        verb = str(cmd.children[0])
        args: List[Any] = []
        kwargs: Dict[str, Any] = {}
        if len(cmd.children) > 1:
            for item in cmd.children[1].children if isinstance(cmd.children[1], Tree) else []:
                if isinstance(item, Tree) and item.data == "named_arg":
                    if len(item.children) == 2 and isinstance(item.children[0], Token):
                        kwargs[str(item.children[0])] = self._expr_to_placeholder(item.children[1])
                    else:
                        args.append(self._expr_to_placeholder(item.children[0]))
        return ActionOp(team=team, verb=verb, args=args, kwargs=kwargs)

    def _lower_context(self, node: Tree) -> ContextUpdateOp:
        vals: List[Any] = []
        if len(node.children) and isinstance(node.children[1], Tree):
            for e in node.children[1].children:
                vals.append(self._expr_to_placeholder(e))
        return ContextUpdateOp(values=vals)

    def _lower_chain_touch(self, node: Tree) -> ChainTouchOp:
        chain = str(node.children[0])
        node_name = node.children[1].value
        # effect is named arg
        effect_expr = None
        for ch in node.find_data("named_arg"):
            if isinstance(ch.children[0], Token) and str(ch.children[0]) == "effect":
                effect_expr = ch.children[1]
                break
        return ChainTouchOp(chain=chain, node=node_name, effect=self._expr_to_placeholder(effect_expr))

    def _lower_system_call(self, node: Tree) -> SystemCallOp:
        target = str(node.children[0])
        op = str(node.children[1])
        args: List[Any] = []
        kwargs: Dict[str, Any] = {}
        if len(node.children) > 2 and isinstance(node.children[2], Tree) and node.children[2].data == "arg_list":
            for item in node.children[2].children:
                if isinstance(item, Tree) and item.data == "named_arg":
                    if len(item.children) == 2 and isinstance(item.children[0], Token):
                        kwargs[str(item.children[0])] = self._expr_to_placeholder(item.children[1])
                    else:
                        args.append(self._expr_to_placeholder(item.children[0]))
        return SystemCallOp(target=target, op=op, args=args, kwargs=kwargs)

    def _lower_deploy(self, node: Tree) -> DeployOp:
        return DeployOp(model=node.children[0].value, env=node.children[1].value)

    def _lower_flow_control(self, node: Tree) -> FlowControlOp:
        kind = str(node.children[0])
        if kind == "back_to":
            target = node.children[1].value
            return FlowControlOp(kind="back_to", target=target)
        return FlowControlOp(kind="end")

    def _lower_par(self, node: Tree) -> ParBlockOp:
        block = node.children[0]
        branches: List[List[OpIR]] = []
        # split child stmts into individual branches (simple heuristic)
        for lst in block.find_data("local_stmt"):
            branches.append(self._lower_block(lst))
        return ParBlockOp(branches=branches)

    def _lower_race(self, node: Tree) -> RaceBlockOp:
        block = node.children[0]
        branches: List[List[OpIR]] = []
        for lst in block.find_data("local_stmt"):
            branches.append(self._lower_block(lst))
        return RaceBlockOp(branches=branches)

    def _expr_to_placeholder(self, expr: Any) -> Any:
        # For now, return a string marker; later we can embed expression trees
        if expr is None:
            return None
        if isinstance(expr, Token):
            return str(expr)
        if isinstance(expr, Tree):
            # a simple pretty marker
            return f"<expr:{expr.data}>"
        return expr
