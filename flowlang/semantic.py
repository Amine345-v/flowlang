from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set, Any, Optional, Tuple
from lark import Tree, Token
from .errors import SemanticError
from .types import CommandKind

@dataclass
class TeamInfo:
    name: str
    kind: CommandKind

@dataclass
class ChainInfo:
    name: str
    nodes: Set[str]

@dataclass
class FlowInfo:
    name: str
    teams: List[str]
    checkpoints: List[str]

class SemanticAnalyzer:
    """Walks the Lark parse tree and performs semantic checks:
    - team action compatibility (team kind vs action verb)
    - flow.back_to refers to existing checkpoint in same flow
    - chain.touch refers to existing chain and node name
    """

    def __init__(self, tree: Tree):
        self.tree = tree
        self.teams: Dict[str, TeamInfo] = {}
        self.chains: Dict[str, ChainInfo] = {}
        self.flows: Dict[str, FlowInfo] = {}
        self.processes: Set[str] = set()
        # allowed fields for result types (populated from result_decl, fallback defaults)
        self.result_fields: Dict[str, Set[str]] = {}
        # var -> type name (string form)
        self.var_types: Dict[str, str] = {}

    def analyze(self) -> None:
        # first pass: collect declarations
        for node in self.tree.find_data("team_decl"):
            # structure: team IDENT : Command < COMMAND_KIND > [ opts ] ;
            name = None
            kind = None
            for ch in node.children:
                if isinstance(ch, Token) and ch.type == "IDENT" and name is None:
                    name = str(ch)
                if isinstance(ch, Token) and ch.type == "COMMAND_KIND" and kind is None:
                    kind = CommandKind(str(ch))
            if name is None or kind is None:
                raise SemanticError("Malformed team declaration (missing name or COMMAND_KIND)")
            self.teams[name] = TeamInfo(name, kind)

        for node in self.tree.find_data("chain_decl"):
            # chain IDENT { nodes: [...] }
            name = None
            for ch in node.children:
                if isinstance(ch, Token) and ch.type == "IDENT":
                    name = str(ch)
                    break
            if name is None:
                raise SemanticError("Malformed chain declaration (missing name)")
            nodes_block = next(node.find_data("ident_list"), None)
            nodes = set()
            if nodes_block:
                for ch in nodes_block.children:
                    if isinstance(ch, Token):
                        nodes.add(str(ch))
            self.chains[name] = ChainInfo(name, nodes)

        # collect declared result types
        any_result = False
        for res in self.tree.find_data("result_decl"):
            any_result = True
            rname = None
            fields: Set[str] = set()
            for ch in res.children:
                if isinstance(ch, Token) and ch.type == "IDENT" and rname is None:
                    rname = str(ch)
            for fld in res.find_data("result_field"):
                fname = str(fld.children[0])
                fields.add(fname)
            if rname:
                self.result_fields[rname] = fields
        if not any_result:
            # defaults if user didn't declare
            self.result_fields = {
                "JudgeResult": {"confidence", "score", "pass"},
                "TryResult": {"output", "metrics"},
                "SearchResult": {"hits"},
                "CommunicateResult": {"text"},
            }

        # collect process names
        for node in self.tree.find_data("process_decl"):
            # process IDENT STRING { ... }
            for ch in node.children:
                if isinstance(ch, Token) and ch.type == "IDENT":
                    self.processes.add(str(ch))
                    break

        for node in self.tree.find_data("flow_decl"):
            flow_name = None
            for ch in node.children:
                if isinstance(ch, Token) and ch.type == "IDENT":
                    flow_name = str(ch)
                    break
            if flow_name is None:
                raise SemanticError("Malformed flow declaration (missing name)")
            # params
            params = []
            for lst in node.find_data("ident_list"):
                # the first ident_list inside flow_params is teams list
                for ch in lst.children:
                    if isinstance(ch, Token):
                        params.append(str(ch))
                break
            # checkpoints
            cps: List[str] = []
            for cp in node.find_data("checkpoint"):
                cp_name = cp.children[0].value  # STRING token
                cps.append(cp_name)
            self.flows[flow_name] = FlowInfo(flow_name, params, cps)

        # second pass: validate actions and controls within flows
        for flow_node in self.tree.find_data("flow_decl"):
            flow_name = str(flow_node.children[0])
            flow_info = self.flows.get(flow_name)
            if not flow_info:
                continue

            # Ensure referenced teams exist
            for t in flow_info.teams:
                if t not in self.teams:
                    raise SemanticError(f"Flow '{flow_name}' references unknown team '{t}'")

            # Validate statements within flow
            # reset per-flow var types
            self.var_types = {}

            for act in flow_node.find_data("action_stmt"):
                team_ident = str(act.children[0])
                if team_ident not in self.teams:
                    raise SemanticError(f"Unknown team '{team_ident}' in flow '{flow_name}'")
                team_info = self.teams[team_ident]
                cmd_node = act.children[1]
                if isinstance(cmd_node, Tree) and cmd_node.data == "command_action":
                    verb_tok = cmd_node.children[0]  # Token for verb
                    verb = str(verb_tok)
                    self._check_team_action(team_info, verb, flow_name)

            # type inference for assignments: infer expression types and record
            for assign in flow_node.find_data("var_assign"):
                var_name = str(assign.children[0])
                expr = assign.children[1]
                inferred = self._infer_expr_type(expr)
                if inferred:
                    self.var_types[var_name] = inferred

            # field access validation: e.g., J3.confidence
            for prim in flow_node.find_data("primary"):
                if len(prim.children) >= 2 and isinstance(prim.children[0], Tree) and prim.children[0].data == "name":
                    base_name = str(prim.children[0].children[0])
                    # If pattern is just field access (not a method call)
                    # e.g., name . field
                    for i in range(1, len(prim.children)):
                        ch = prim.children[i]
                        if isinstance(ch, Tree) and ch.data == "field":
                            fld = str(ch.children[0])
                            # skip if this is actually a verb on a team followed by a call; handled by action checks
                            is_followed_by_call = (i + 1 < len(prim.children) and isinstance(prim.children[i+1], Tree) and prim.children[i+1].data == "call")
                            if is_followed_by_call:
                                break
                            # validate field when we know var type
                            vtype = self.var_types.get(base_name)
                            if vtype:
                                allowed = self.result_fields.get(vtype, set())
                                if fld not in allowed:
                                    raise SemanticError(
                                        f"Variable '{base_name}' of type {vtype} has no field '{fld}' (allowed: {sorted(allowed)})")
                            break

            # flow.back_to checks
            for ctl in flow_node.find_data("flow_control"):
                # forms: flow . back_to ( STRING ) | flow . end
                if len(ctl.children) >= 1 and isinstance(ctl.children[0], Token) and str(ctl.children[0]) == "back_to":
                    cp_str = ctl.children[1].value  # STRING
                    if cp_str not in flow_info.checkpoints:
                        raise SemanticError(
                            f"flow.back_to('{cp_str}') not found in flow '{flow_name}' checkpoints {flow_info.checkpoints}")

            # chain.touch checks
            for touch in flow_node.find_data("chain_touch_stmt"):
                chain_name = str(touch.children[0])
                if chain_name not in self.chains:
                    raise SemanticError(f"Unknown chain '{chain_name}' in flow '{flow_name}'")
                node_name = touch.children[1].value  # STRING
                if node_name not in self.chains[chain_name].nodes:
                    raise SemanticError(
                        f"Chain '{chain_name}' has no node '{node_name}' (known: {sorted(self.chains[chain_name].nodes)})")

            # system_call_stmt target validation (must be known chain or process)
            for call in flow_node.find_data("system_call_stmt"):
                target = str(call.children[0])
                if target not in self.chains and target not in self.processes:
                    raise SemanticError(
                        f"Unknown system target '{target}' in flow '{flow_name}' (expected chain or process name)")

            # audit_stmt must target a known process
            for aud in flow_node.find_data("audit_stmt"):
                proc = str(aud.children[0])
                if proc not in self.processes:
                    raise SemanticError(
                        f"Unknown process '{proc}' in audit_stmt within flow '{flow_name}'")

    def _check_team_action(self, team: TeamInfo, verb: str, flow_name: str) -> None:
        mapping = {
            CommandKind.Search: "search",
            CommandKind.Try: "try",
            CommandKind.Judge: "judge",
            CommandKind.Communicate: "ask",
        }
        expected = mapping.get(team.kind)
        # Standardize verb for comparison against expected keyword literal
        actual = str(verb).strip().lower()
        if expected != actual:
            raise SemanticError(
                f"Team '{team.name}' of kind {team.kind.value} cannot perform {repr(actual)}. Expected {repr(expected)}.")

    def _infer_result_type_from_expr(self, expr: Tree) -> Optional[str]:
        """Detect expressions of the form TeamName.verb(...) and map to a result type name."""
        # look for a primary node with base name, field (verb), call
        target = None
        for prim in expr.find_data("primary"):
            if len(prim.children) >= 3 and isinstance(prim.children[0], Tree) and prim.children[0].data == "name":
                team_name = str(prim.children[0].children[0])
                f1, c1 = prim.children[1], prim.children[2]
                if isinstance(f1, Tree) and f1.data == "field" and isinstance(c1, Tree) and c1.data == "call":
                    verb = str(f1.children[0])
                    # map via team kind
                    team_info = self.teams.get(team_name)
                    if not team_info:
                        continue
                    kind_to_result = {
                        CommandKind.Search: "SearchResult",
                        CommandKind.Try: "TryResult",
                        CommandKind.Judge: "JudgeResult",
                        CommandKind.Communicate: "CommunicateResult",
                    }
                    return kind_to_result.get(team_info.kind)
        return None

    # --------- Expression type inference (basic) ---------
    def _infer_expr_type(self, expr: Tree) -> Optional[str]:
        """Infer a coarse type name for an expression.
        Returns one of: number|string|boolean|list|dict|<ResultType>|unknown.
        Raises SemanticError for incompatible binary ops when both sides known.
        """
        # shortcut: team.verb() result types
        t = self._infer_result_type_from_expr(expr)
        if t:
            return t
        # walk down expression kinds
        if isinstance(expr, Tree):
            dt = expr.data
            if dt in ("expr", "or_expr", "and_expr", "cmp_expr", "add_expr", "mul_expr", "unary_expr"):
                # binary chains: infer lhs then validate pairs
                if len(expr.children) == 1:
                    return self._infer_expr_type(expr.children[0])
                # left-assoc: a op b op c ...
                lhs = self._infer_expr_type(expr.children[0])
                i = 1
                while i < len(expr.children):
                    op_tok = expr.children[i]
                    rhs_type = self._infer_expr_type(expr.children[i+1])
                    op = str(op_tok) if isinstance(op_tok, Token) else None
                    self._validate_binop_types(dt, op, lhs, rhs_type)
                    # result type of numeric/string ops stays same domain when possible
                    if dt in ("add_expr", "mul_expr") and lhs in ("number", "string") and lhs == rhs_type:
                        pass
                    elif dt in ("and_expr", "or_expr"):
                        lhs = "boolean"
                    elif dt == "cmp_expr":
                        lhs = "boolean"
                    else:
                        lhs = lhs or rhs_type
                    i += 2
                return lhs
            if dt == "number":
                return "number"
            if dt == "string":
                return "string"
            if dt == "boolean":
                return "boolean"
            if dt == "list_literal":
                return "list"
            if dt == "dict_literal":
                return "dict"
            if dt == "name":
                var = str(expr.children[0])
                return self.var_types.get(var)
            if dt == "primary":
                # try base, then member accesses
                base = self._infer_expr_type(expr.children[0]) if expr.children else None
                # field access on result types is allowed; dicts unknown here statically
                return base
        return None

    def _validate_binop_types(self, level: str, op: Optional[str], lhs: Optional[str], rhs: Optional[str]):
        # If either side unknown, skip
        if lhs is None or rhs is None or op is None:
            return
        # Logical ops require booleans
        if level == "and_expr" or (op in ("&&", "||")):
            if lhs != "boolean" or rhs != "boolean":
                raise SemanticError(f"Logical op requires boolean operands, got {lhs} {op} {rhs}")
            return
        # Comparison results to boolean; allow numbers/strings comparison when both equal
        if level == "cmp_expr" or op in ("==", "!=", "<", ">", "<=", ">="):
            if lhs != rhs or lhs not in ("number", "string"):
                # allow comparing result types conservatively only if equal type names
                if lhs != rhs:
                    raise SemanticError(f"Incompatible comparison {lhs} {op} {rhs}")
            return
        # Arithmetic
        if level in ("add_expr", "mul_expr") or op in ("+", "-", "*", "/"):
            if lhs != rhs or lhs not in ("number", "string"):
                raise SemanticError(f"Incompatible arithmetic {lhs} {op} {rhs}")
            return
