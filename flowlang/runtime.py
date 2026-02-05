from __future__ import annotations
from dataclasses import dataclass
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast
import os
import json
from lark import Tree, Token
import asyncio

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter
    trace.set_tracer_provider(TracerProvider())
    _otel_tracer = trace.get_tracer(__name__)
    trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
except Exception:  # pragma: no cover - optional dependency
    _otel_tracer = None

from .parser import parse
from .semantic import SemanticAnalyzer
from .errors import RuntimeFlowError
from .types import TypedValue, ValueTag, Order, CommandKind
from .ai_providers import select_provider
from .persistence import PersistenceManager, FlowState

# Optional OpenAI client import for AI-backed execution
try:  # pragma: no cover - optional dependency
    from openai import OpenAI as _OpenAIClient
except Exception:  # pragma: no cover - optional dependency
    _OpenAIClient = None

@dataclass
class EvalContext:
    variables: Dict[str, Any]
    checkpoints: List[str]
    checkpoint_index: int = 0
    current_stage: Optional[str] = None
    back_to_target: Optional[str] = None
    reports: List[Any] = None
    merge_policy: str = "last_wins"

    def __post_init__(self):
        if self.reports is None:
            self.reports = []

class Runtime:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.tree: Optional[Tree] = None
        self.console: List[str] = []
        # runtime structures
        self.chains: Dict[str, Dict[str, Any]] = {}
        self.processes: Dict[str, Dict[str, Any]] = {}
        self.teams: Dict[str, Dict[str, Any]] = {}
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.roles: Dict[str, Dict[str, Any]] = {}
        self.metrics: Dict[str, Any] = {"actions": 0, "checkpoints": 0, "back_to": 0, "verbs": {}, "checkpoint_ms": {}}
        self.tracer = _otel_tracer
        # AI execution wiring: enabled when OpenAI client available and API key set
        self.ai_client = _OpenAIClient() if (_OpenAIClient and os.getenv("OPENAI_API_KEY")) else None
        # Default model and per-verb mapping; can be overridden by env vars
        self.ai_default_model = os.getenv("FLOWLANG_AI_MODEL", "gpt-4o")
        # Allow a simple per-verb override via env, e.g. FLOWLANG_AI_MODEL_SEARCH
        self.ai_model_per_verb: Dict[str, str] = {
            "ask": os.getenv("FLOWLANG_AI_MODEL_ASK", self.ai_default_model),
            "search": os.getenv("FLOWLANG_AI_MODEL_SEARCH", self.ai_default_model),
            "try": os.getenv("FLOWLANG_AI_MODEL_TRY", self.ai_default_model),
            "judge": os.getenv("FLOWLANG_AI_MODEL_JUDGE", self.ai_default_model),
        }
        # Multi-provider selector (OpenAI → Anthropic → Gemini → Mistral → Cohere → Azure → OpenRouter → Ollama)
        self.ai_provider = select_provider()
        self.persistence = PersistenceManager()

    def log(self, msg: str):
        self.console.append(msg)
        print(msg)

    def load(self, source: str | Path) -> Tree:
        self.tree = parse(source)
        SemanticAnalyzer(self.tree).analyze()
        self._build_structs()
        return self.tree

    # ---------- Execution entry ----------
    def run_flow(self, flow_name: Optional[str] = None):
        if not self.tree:
            raise RuntimeFlowError("No program loaded")
        # find target flow
        for flow in self.tree.find_data("flow_decl"):
            name = str(flow.children[0])
            if flow_name is None or name == flow_name:
                if self.tracer:
                    with self.tracer.start_as_current_span(f"flow:{name}"):
                        self._execute_flow(flow)
                else:
                    self._execute_flow(flow)
                return
        raise RuntimeFlowError(f"Flow '{flow_name}' not found")

    # ---------- Flow execution ----------
    def _execute_flow(self, flow: Tree, resume_state: Optional[FlowState] = None):
        name = str(flow.children[0])
        # teams list
        teams: List[str] = []
        first_ident_list = next(flow.find_data("ident_list"))
        for ch in first_ident_list.children:
            if isinstance(ch, Token):
                teams.append(str(ch))
        # collect checkpoints in order (nodes themselves)
        checkpoints_nodes: List[Tree] = [cp for cp in flow.find_data("checkpoint")]
        checkpoints_names: List[str] = [cp.children[0].value for cp in checkpoints_nodes]
        # read merge_policy header if present
        merge_policy = "last_wins"
        for hdr in flow.find_data("flow_header"):
            # scan tokens for merge_policy key then the policy value
            seen_key = False
            for ch in hdr.children:
                if isinstance(ch, Token):
                    sval = str(ch)
                    if seen_key:
                        if sval in ("last_wins", "deep_merge", "crdt", "bounded_deep"):
                            merge_policy = sval
                            seen_key = False
                            break
                        seen_key = False
                    if sval == "merge_policy":
                        seen_key = True
        ctx_vars = {}
        if resume_state:
            ctx_vars = resume_state.eval_context
            
        ctx = EvalContext(variables=ctx_vars, checkpoints=checkpoints_names, merge_policy=merge_policy)
        self.log(f"[flow] Start '{name}' with checkpoints: {checkpoints_names}")
        pc = 0
        
        if resume_state:
             # Fast forward
             last_cp = resume_state.checkpoints[-1] if resume_state.checkpoints else None
             if last_cp:
                 if last_cp in checkpoints_names:
                     pc = checkpoints_names.index(last_cp) + 1
                     self.log(f"[flow] Resuming: Fast-forwarding past {last_cp} (pc={pc})")
                 else:
                     self.log(f"[flow] Warning: synced checkpoint {last_cp} not found in current flow version")
        while pc < len(checkpoints_nodes):
            cp_node = checkpoints_nodes[pc]
            cp_name = cp_node.children[0].value
            ctx.current_stage = cp_name
            self.log(f"[checkpoint] -> {cp_name}")
            self.metrics["checkpoints"] += 1
            t0 = time.perf_counter()
            # execute local statements inside checkpoint
            if self.tracer:
                with self.tracer.start_as_current_span(f"checkpoint:{cp_name}"):
                    self._exec_block(cp_node.find_data("local_stmt"), ctx)
            else:
                self._exec_block(cp_node.find_data("local_stmt"), ctx)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self.metrics["checkpoint_ms"][cp_name] = dt_ms
            self.log(f"[checkpoint.done] {cp_name} in {dt_ms:.2f} ms")
            
            # 6. Concept Alignment: Checkpoint Reports
            # If the checkpoint defined a report, we prune everything else (Handover)
            report_vars = set()
            # Look for expr_list child directly in cp_node
            for child in cp_node.children:
                if isinstance(child, Tree) and child.data == "expr_list":
                     for e in child.children:
                         # Evaluate to get the variable name/key to keep
                         val = self._eval_expr(e, ctx)
                         report_vars.add(str(val))
                     break
            
            if report_vars:
                 kept = {}
                 for k, v in ctx.variables.items():
                     if k.startswith("__") or k in report_vars:
                         # 6. Concept Alignment: Unload Orders into Reports
                         # If it's an Order or a list of Orders, mark them complete as they are now 'reports'
                         def _mark_complete(val):
                             if isinstance(val, Order):
                                 val.complete()
                             elif isinstance(val, list):
                                 for x in val: _mark_complete(x)
                             elif isinstance(val, TypedValue) and val.tag == ValueTag.List:
                                 for x in val.value: _mark_complete(x)
                         
                         _mark_complete(v)
                         kept[k] = v
                 
                 removed = len(ctx.variables) - len(kept)
                 ctx.variables = kept
                 self.log(f"[checkpoint] Report handover: kept {list(report_vars)}, pruned {removed} keys")

            # Auto-save state (Save the pruned/handover state)
            try:
                saved_path = self.persistence.save_state(
                    name=name,
                    ctx_vars=ctx.variables,
                    checkpoints=checkpoints_names[:pc+1]
                )
                self.log(f"[persistence] Saved state to {saved_path}")
            except Exception as e:
                self.log(f"[persistence] Failed to save state: {e}")

            if ctx.back_to_target is not None:
                target = ctx.back_to_target
                if target not in ctx.checkpoints:
                    raise RuntimeFlowError(f"back_to unknown checkpoint '{target}'")
                pc = ctx.checkpoints.index(target)
                self.log(f"[flow] back_to -> {target}")
                ctx.back_to_target = None
                self.metrics["back_to"] += 1
                continue
            pc += 1
        self.log(f"[flow] End '{name}'")
        self.log(f"[metrics] {self.metrics}")

    def resume(self, state_path: str):
        """Resume execution from a saved state file."""
        state = self.persistence.load_state(state_path)
        self.log(f"[resume] Loaded state '{state.flow_name}' from {state.timestamp}")
        
        if not self.tree:
             raise RuntimeFlowError("Cannot resume: no program loaded. Load source first.")

        target_flow = None
        for flow in self.tree.find_data("flow_decl"):
            if str(flow.children[0]) == state.flow_name:
                target_flow = flow
                break
        if not target_flow:
            raise RuntimeFlowError(f"Flow '{state.flow_name}' not found in loaded program")
        
        self._execute_flow(target_flow, resume_state=state)

    # ---------- Statement execution ----------
    def _exec_block(self, stmts_iter, ctx: EvalContext):
        for node in stmts_iter:
            # node is a Tree("local_stmt", [...])
            child = node.children[0]
            if isinstance(child, Tree):
                match child.data:
                    case "var_assign":
                        name = str(child.children[0])
                        value = self._eval_expr(child.children[1], ctx)
                        ctx.variables[name] = value
                        self.log(f"[set] {name} = {value}")
                    case "action_stmt":
                        self._exec_action(child, ctx)
                    case "if_stmt":
                        self._exec_if(child, ctx)
                    case "while_stmt":
                        self._exec_while(child, ctx)
                    case "for_stmt":
                        self._exec_for(child, ctx)
                    case "par_stmt":
                        self._exec_par(child, ctx)
                    case "race_stmt":
                        self._exec_race(child, ctx)
                    case "flow_control":
                        self._exec_flow_control(child, ctx)
                    case "context_stmt":
                        self._exec_context_stmt(child, ctx)
                    case "chain_touch_stmt":
                        self._exec_chain_touch(child, ctx)
                    case "deploy_stmt":
                        self._exec_deploy(child, ctx)
                    case "audit_stmt":
                        self._exec_audit(child, ctx)
                    case "system_call_stmt":
                        self._exec_system_call(child, ctx)
                    case "confirm_stmt":
                        self._exec_confirm_stmt(child, ctx)
                    case _:
                        raise RuntimeFlowError(f"Unsupported statement: {child.data}")
            else:
                raise RuntimeFlowError("Malformed local statement")

    def _exec_if(self, node: Tree, ctx: EvalContext):
        cond = self._eval_expr(node.children[0], ctx)
        then_block = node.children[1]
        else_block = node.children[2] if len(node.children) > 2 else None
        if self._truthy(cond):
            self._exec_block(then_block.find_data("local_stmt"), ctx)
        elif else_block:
            self._exec_block(else_block.find_data("local_stmt"), ctx)

    def _exec_while(self, node: Tree, ctx: EvalContext):
        cond_node = node.children[0]
        block = node.children[1]
        guard = 0
        MAX_ITER = 1000
        while self._truthy(self._eval_expr(cond_node, ctx)):
            self._exec_block(block.find_data("local_stmt"), ctx)
            guard += 1
            if guard > MAX_ITER:
                raise RuntimeFlowError("while loop exceeded iteration limit")

    def _exec_for(self, node: Tree, ctx: EvalContext):
        var_name = str(node.children[0])
        iter_name = str(node.children[1])
        block = node.children[2]
        iterable = ctx.variables.get(iter_name)
        if not isinstance(iterable, list):
            raise RuntimeFlowError(f"for loop expects list in '{iter_name}'")
        for item in iterable:
            ctx.variables[var_name] = item
            self._exec_block(block.find_data("local_stmt"), ctx)

    def _exec_par(self, node: Tree, ctx: EvalContext):
        # par block: execute local statements concurrently on cloned contexts, then merge
        block = node.children[0]
        stmts = list(block.find_data("local_stmt"))
        self.log("[par] begin")
        async def run_stmt(stmt):
            subctx = self._ctx_clone(ctx)
            await self._a_exec_block([stmt], subctx)
            return subctx
        async def run_all():
            return await asyncio.gather(*(run_stmt(s) for s in stmts))
        results = asyncio.run(run_all()) if stmts else []
        # merge contexts according to policy
        for sub in results:
            self._merge_contexts(ctx, sub)
        self.log("[par] end")

    def _exec_race(self, node: Tree, ctx: EvalContext):
        # race block: execute all branches concurrently, take first finished, ignore others, merge winning context
        block = node.children[0]
        stmts = list(block.find_data("local_stmt"))
        if not stmts:
            return
        self.log("[race] begin")
        async def run_stmt(stmt):
            subctx = self._ctx_clone(ctx)
            await self._a_exec_block([stmt], subctx)
            return subctx
        async def run_race():
            tasks = [asyncio.create_task(run_stmt(s)) for s in stmts]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for p in pending:
                p.cancel()
            return list(done)[0].result()
        winner = asyncio.run(run_race())
        self._merge_contexts(ctx, winner)
        self.log("[race] end -> winner merged")

    # ---------- Async helpers ----------
    async def _a_exec_block(self, stmts_list: List[Tree], ctx: EvalContext):
        # run sync executor in an async-friendly manner
        self._exec_block(stmts_list, ctx)

    def _ctx_clone(self, ctx: EvalContext) -> EvalContext:
        return EvalContext(variables=dict(ctx.variables), checkpoints=list(ctx.checkpoints), checkpoint_index=ctx.checkpoint_index, back_to_target=None, reports=list(ctx.reports), merge_policy=ctx.merge_policy)

    def _merge_contexts(self, base: EvalContext, other: EvalContext):
        policy = base.merge_policy
        if policy == "last_wins":
            base.variables.update(other.variables)
        elif policy == "deep_merge":
            base.variables = self._deep_merge(base.variables, other.variables)
        elif policy == "crdt":
            base.variables = self._crdt_merge(base.variables, other.variables)
        elif policy == "bounded_deep":
            base.variables = self._bounded_deep_merge(base.variables, other.variables)
        else:
            base.variables.update(other.variables)
        # reports always extend
        base.reports.extend(other.reports)

    def _deep_merge(self, a: Any, b: Any) -> Any:
        # recursively merge dicts, extend lists, otherwise b wins
        if isinstance(a, dict) and isinstance(b, dict):
            out = dict(a)
            for k, v in b.items():
                if k in out:
                    out[k] = self._deep_merge(out[k], v)
                else:
                    out[k] = v
            return out
        if isinstance(a, list) and isinstance(b, list):
            return a + [x for x in b]
        return b

    def _bounded_deep_merge(self, a: Any, b: Any, depth: int = 0) -> Any:
        MAX_DEPTH = 50
        MAX_LIST_LEN = 100
        
        if depth > MAX_DEPTH:
            return b  # Depth limit reached, replacement wins
            
        if isinstance(a, dict) and isinstance(b, dict):
            out = dict(a)
            for k, v in b.items():
                if k in out:
                    out[k] = self._bounded_deep_merge(out[k], v, depth + 1)
                else:
                    out[k] = v
            return out
            
        if isinstance(a, list) and isinstance(b, list):
            # Enforce max list length
            new_items = [x for x in b]
            merged = a + new_items
            if len(merged) > MAX_LIST_LEN:
                # Keep last N
                return merged[-MAX_LIST_LEN:]
            return merged
            
        return b

    def _crdt_merge(self, a: Any, b: Any) -> Any:
        # Simple CRDT-like merge: numbers -> max, strings -> last, lists -> union preserving order, dicts -> recursive
        if isinstance(a, dict) and isinstance(b, dict):
            out = dict(a)
            for k, v in b.items():
                if k in out:
                    out[k] = self._crdt_merge(out[k], v)
                else:
                    out[k] = v
            return out
        if isinstance(a, list) and isinstance(b, list):
            seen = set()
            out = []
            for x in a + b:
                key = repr(x)
                if key not in seen:
                    seen.add(key)
                    out.append(x)
            return out
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a if a >= b else b
        return b

    def _exec_flow_control(self, node: Tree, ctx: EvalContext):
        # flow . back_to ( STRING ) | flow . end
        if len(node.children) and isinstance(node.children[0], Token) and str(node.children[0]) == "back_to":
            target = node.children[1].value
            ctx.back_to_target = target
        else:
            # end: do nothing in prototype
            pass

    def _exec_context_stmt(self, node: Tree, ctx: EvalContext):
        op = str(node.children[0]) if isinstance(node.children[0], Token) else node.children[0]
        
        # Max context size limit (number of keys in variables)
        MAX_CONTEXT_SIZE = int(os.getenv("FLOWLANG_MAX_CONTEXT_SIZE", "1000"))
        # Check size before any operation
        if len(ctx.variables) > MAX_CONTEXT_SIZE:
             # Allow prune/snapshot even if full, but block update
             if op not in ("prune", "snapshot"):
                 from .errors import ContextOverflowError
                 raise ContextOverflowError(f"Context size {len(ctx.variables)} exceeds limit {MAX_CONTEXT_SIZE}")

        if op == "update":
            # update(args...)
            args_tree = node.children[1] if len(node.children) > 1 else None
            items = []
            if isinstance(args_tree, Tree) and args_tree.data == "expr_list":
                for e in args_tree.children:
                    items.append(self._eval_expr(e, ctx))
            ctx.reports.append(items)
            self.log(f"[context.update] +{len(items)} items")
            
        elif op == "snapshot":
            # snapshot("name"?)
            name = "snapshot"
            if len(node.children) > 1:
                args_tree = node.children[1]
                if isinstance(args_tree, Tree) and args_tree.data == "expr_list":
                    name = str(self._eval_expr(args_tree.children[0], ctx))
            
            snap = {k: v for k, v in ctx.variables.items()}
            # Store snapshot validation-safe (e.g. in a special key or externally)
            # For now, we store in variables with a prefix, but we must exclude it from counting or future snapshots?
            # Better: store just a log message or return it. The original code stored in __snapshot__
            ctx.variables[f"__snapshot_{name}__"] = snap
            self.log(f"[context.snapshot] saved as '__snapshot_{name}__'")
            
        elif op == "prune":
            # prune(keep=[keys...])
            keep_keys = set()
            if len(node.children) > 1:
                args_tree = node.children[1]
                if isinstance(args_tree, Tree) and args_tree.data == "expr_list":
                    # Evaluating arguments which should be strings
                    for e in args_tree.children:
                        val = self._eval_expr(e, ctx)
                        if isinstance(val, list):
                            keep_keys.update(str(k) for k in val)
                        else:
                            keep_keys.add(str(val))
            
            # Always keep system keys
            preserved = {k: v for k, v in ctx.variables.items() if k.startswith("__") or k in keep_keys}
            removed_count = len(ctx.variables) - len(preserved)
            ctx.variables = preserved
            self.log(f"[context.prune] removed {removed_count} keys, kept {list(keep_keys)}")
            
        else:
            raise RuntimeFlowError(f"Unknown context op {op}")

    def _exec_chain_touch(self, node: Tree, ctx: EvalContext):
        chain_name = str(node.children[0])
        node_name = node.children[1].value
        effect = None
        # find an expr child if present
        for ch in node.children:
            if isinstance(ch, Tree) and ch.data in ("expr", "or_expr", "and_expr", "cmp_expr", "add_expr", "mul_expr", "unary_expr", "primary"):
                effect = self._eval_expr(ch, ctx)
                break
        # Default effect if not provided
        if effect is None:
            effect = 1.0
        # Apply effect and propagate using the chain's configured propagation policy
        ch = self.chains.get(chain_name)
        if not ch:
            raise RuntimeFlowError(f"Unknown chain '{chain_name}'")
        if str(node_name) not in ch["nodes"]:
            raise RuntimeFlowError(f"Chain '{chain_name}' has no node '{node_name}'")
        ch["effects"][str(node_name)] = effect
        order = ch["order"]
        decay = float(ch["propagation"].get("decay", 0.6))
        do_fwd = bool(ch["propagation"].get("forward", True))
        do_bwd = bool(ch["propagation"].get("backprop", True))
        cap = ch["propagation"].get("cap")
        try:
            idx = order.index(str(node_name))
        except ValueError:
            idx = None
        if idx is not None:
            # forward diffusion
            if do_fwd:
                cur = effect
                j = idx + 1
                while j < len(order):
                    # numeric decay only if numeric
                    if isinstance(cur, (int, float)):
                        cur = float(cur) * decay
                        if cap is not None and isinstance(cap, (int, float)) and cur < float(cap):
                            break
                        ch["effects"][order[j]] = max(cur, ch["effects"].get(order[j], 0))
                    else:
                        ch["effects"][order[j]] = cur
                    j += 1
            # backward diffusion
            if do_bwd:
                cur = effect
                j = idx - 1
                while j >= 0:
                    if isinstance(cur, (int, float)):
                        cur = float(cur) * decay
                        if cap is not None and isinstance(cap, (int, float)) and cur < float(cap):
                            break
                        ch["effects"][order[j]] = max(cur, ch["effects"].get(order[j], 0))
                    else:
                        ch["effects"][order[j]] = cur
                    j -= 1
        self.log(f"[chain.touch] {chain_name}.{node_name} effect={effect} (decay={decay}, cap={cap}, fwd={do_fwd}, bwd={do_bwd})")

    def _exec_deploy(self, node: Tree, ctx: EvalContext):
        model = node.children[0].value
        env = node.children[1].value
        
        if self.dry_run:
            self.log(f"[dry_run] Skip deploy model={model} env={env}")
            return
            
        # Governance checks
        # 1) capability 'deploy' present on any team's role
        has_deploy_cap = False
        for tinfo in self.teams.values():
            r = tinfo.get("role")
            if r and r in self.roles and ("deploy" in self.roles[r]["capabilities"]):
                has_deploy_cap = True
                break
        if not has_deploy_cap and self.roles:
            raise RuntimeFlowError("Deploy blocked: no team role with 'deploy' capability")
        # 2) chain constraint require_eval implies Evaluation node received sufficient effect
        for cname, ch in self.chains.items():
            req = ch["constraints"].get("require_eval")
            if req:
                eval_eff = ch["effects"].get("Evaluation", 0.0)
                if isinstance(eval_eff, (int, float)) and eval_eff < 0.7:
                    raise RuntimeFlowError(
                        f"Deploy blocked by chain '{cname}': require_eval and Evaluation effect {eval_eff} < 0.7")
        self.log(f"[deploy] model={model} env={env}")

    def _exec_audit(self, node: Tree, ctx: EvalContext):
        # IDENT "." "audit" "(" ")"
        proc = str(node.children[0])
        self.log(f"[process.audit] {proc}")

    def _exec_system_call(self, node: Tree, ctx: EvalContext):
        # IDENT "." IDENT "(" arg_list? ")"
        target = str(node.children[0])
        op = str(node.children[1])
        
        if self.dry_run:
             self.log(f"[dry_run] Skip system call {target}.{op}")
             return

        args: List[Any] = []
        kwargs: Dict[str, Any] = {}
        if len(node.children) > 2 and isinstance(node.children[2], Tree) and node.children[2].data == "arg_list":
            for item in node.children[2].children:
                if isinstance(item, Tree) and item.data == "named_arg":
                    if len(item.children) == 2 and isinstance(item.children[0], Token):
                        k = str(item.children[0])
                        v = self._eval_expr(item.children[1], ctx)
                        kwargs[k] = v
                    else:
                        v = self._eval_expr(item.children[0], ctx)
                        args.append(v)
        # Dispatch to chain ops
        if target in self.chains:
            self._chain_call(target, op, args, kwargs, ctx)
            return
        # Dispatch to process ops
        if target in self.processes:
            self._process_call(target, op, args, kwargs, ctx)
            return
        # Tools/resources
        if target in self.tools:
            self._tool_call(target, op, args, kwargs, ctx)
            return
        self.log(f"[system] {target}.{op} args={args} kwargs={kwargs}")

    def _exec_confirm_stmt(self, node: Tree, ctx: EvalContext):
        # confirm ( STRING [, kwargs] ) -> IDENT
        prompt_node = node.children[0]
        prompt = prompt_node.value.strip('"') if isinstance(prompt_node, Token) else str(prompt_node)
        
        # kwargs parsing (timeout, etc)
        kwargs = {}
        target_var = None
        
        # Iterating children to find named args and target variable
        # Grammar: "confirm" "(" STRING ("," named_arg)* ")" "->" IDENT
        # children: [STRING, named_arg*, IDENT]
        for ch in node.children[1:]:
             if isinstance(ch, Tree) and ch.data == "named_arg":
                 k = str(ch.children[0])
                 v = self._eval_expr(ch.children[1], ctx)
                 kwargs[k] = v
             elif isinstance(ch, Token):
                 target_var = str(ch)
                 
        if not target_var:
             # Should be last child?
             last = node.children[-1]
             if isinstance(last, Token):
                 target_var = str(last)
             else:
                 raise RuntimeFlowError("confirm statement missing target variable")

        timeout = kwargs.get("timeout", float('inf'))
        self.log(f"[gate] Requesting confirmation: '{prompt}' (timeout={timeout}s)")
        
        # In a real production system, this would suspend execution or call a callback.
        # For now, we simulate with CLI input or env var auto-approve.
        if os.getenv("FLOWLANG_AUTO_APPROVE"):
            self.log("[gate] Auto-approved by env")
            approved = True
        else:
            try:
                # Use input() with simple printed prompt
                 print(f"!!! CONFIRM REQUEST: {prompt} [y/N] ", end="", flush=True)
                 # blocking wait
                 resp = input().strip().lower()
                 approved = resp in ("y", "yes")
            except EOFError:
                 approved = False
        
        ctx.variables[target_var] = approved
        self.log(f"[gate] Result: {approved}")

    # ---------- Build runtime structures from AST ----------
    def _build_structs(self):
        self.chains.clear()
        self.processes.clear()
        self.teams.clear()
        self.tools.clear()
        self.roles.clear()
        if not self.tree:
            return
        # roles
        for rl in self.tree.find_data("role_decl"):
            rname = None
            caps: List[str] = []
            for tok in rl.children:
                if isinstance(tok, Token) and tok.type == "IDENT" and rname is None:
                    rname = str(tok)
                    break
            cap_block = next(rl.find_data("role_body"), None)
            if cap_block:
                for caps_list in cap_block.find_data("ident_list"):
                    for c in caps_list.children:
                        if isinstance(c, Token):
                            caps.append(str(c))
                    break
            if rname:
                self.roles[rname] = {"capabilities": set(caps)}
        # teams
        for tm in self.tree.find_data("team_decl"):
            tname = None
            kind = None
            size = 1
            distribution = "round_robin"
            role = None
            policy = None
            # first child IDENT is name; COMMAND_KIND present as token
            for tok in tm.children:
                if isinstance(tok, Token) and tok.type == "IDENT" and tname is None:
                    tname = str(tok)
                if isinstance(tok, Token) and tok.type == "COMMAND_KIND" and kind is None:
                    kind = str(tok)
            # options
            for opt in tm.find_data("team_opt"):
                # opt children: key (=) value
                if len(opt.children) >= 1:
                    key = str(opt.children[0])
                    if key == "size":
                        size = int(float(opt.children[-1]))
                    elif key == "distribution":
                        distribution = str(opt.children[-1])
                    elif key == "role":
                        role = str(opt.children[-1])
                    elif key == "policy":
                        policy = str(opt.children[-1])
            if tname:
                self.teams[tname] = {
                    "kind": kind,
                    "size": size,
                    "distribution": distribution,
                    "role": role,
                    "policy": policy,
                }
        # chains
        for ch in self.tree.find_data("chain_decl"):
            name = None
            for tok in ch.children:
                if isinstance(tok, Token) and tok.type == "IDENT":
                    name = str(tok)
                    break
            if not name:
                continue
            nodes = []
            lst = next(ch.find_data("ident_list"), None)
            if lst:
                for t in lst.children:
                    if isinstance(t, Token):
                        nodes.append(str(t))
            labels: Dict[str, Any] = {}
            label_block = next(ch.find_data("label_kv_list"), None)
            if label_block:
                # pairs IDENT : STRING
                it = iter(label_block.children)
                for a, b, colon in zip(it, it, [None]*999):
                    if isinstance(a, Token) and isinstance(b, Token):
                        labels[str(a)] = b.value.strip('"')
            constraints: Dict[str, Any] = {}
            for cons in ch.find_data("constraint_item"):
                key = str(cons.children[0])
                val = cons.children[1]
                constraints[key] = self._eval_expr(val, EvalContext(variables={}, checkpoints=[]))
            # propagation args
            prop = {"decay": 0.6, "backprop": True, "forward": True, "cap": None}
            prop_args = next(ch.find_data("prop_args"), None)
            if prop_args:
                for p in prop_args.children:
                    if isinstance(p, Tree) and p.data == "prop_arg":
                        key = str(p.children[0])
                        val = p.children[1]
                        v = self._eval_expr(val, EvalContext(variables={}, checkpoints=[]))
                        prop[key] = v
            self.chains[name] = {
                "nodes": set(nodes),
                "order": list(nodes),
                "labels": labels,
                "constraints": constraints,
                "effects": {},
                "propagation": prop,
            }
        # processes
        for pr in self.tree.find_data("process_decl"):
            pname = None
            for tok in pr.children:
                if isinstance(tok, Token) and tok.type == "IDENT":
                    pname = str(tok)
                    break
            if not pname:
                continue
            nodes: Dict[str, Dict[str, Any]] = {}
            root = None
            branches: Dict[str, List[str]] = {}
            for child in pr.children:
                if isinstance(child, Token) and child.type == "STRING" and root is None:
                     # This matches the 'root : STRING' part of the grammar
                     # Wait, children[1] is the description STRING. 
                     # children[2]... might be root. 
                     pass
            
            # Use find_data to be more robust
            root_match = list(pr.find_data("root")) # There is no 'root' rule in grammar, it's literal
            # Grammar is: "process" IDENT STRING "{" "root" ":" STRING ";" ...
            # Let's find the STRING tokens.
            strings = [str(c.value) for c in pr.children if isinstance(c, Token) and c.type == "STRING"]
            if len(strings) >= 2:
                root = strings[1]
            
            for br in pr.find_data("process_branch"):
                parent = str(br.children[0].value)
                children = [str(c.value) for c in br.find_data("string_list") for c in c.children]
                branches[parent] = children

            for pn in pr.find_data("process_node"):
                nname = pn.children[0].value
                props: Dict[str, Any] = {}
                for pa in pn.find_data("prop_assign"):
                    k = str(pa.children[0])
                    v = self._eval_expr(pa.children[1], EvalContext(variables={},checkpoints=[]))
                    props[k] = v
                nodes[nname] = props
            policies: Dict[str, Any] = {}
            for pol in pr.find_data("process_policy_item"):
                k = str(pol.children[0])
                v = self._eval_expr(pol.children[1], EvalContext(variables={},checkpoints=[]))
                policies[k] = v
            self.processes[pname] = {
                "nodes": nodes,
                "root": root,
                "branches": branches,
                "policies": policies,
                "marks": {},
            }
        # resources/tools
        for rs in self.tree.find_data("resource_decl"):
            rname = None
            rkind = None
            props: Dict[str, Any] = {}
            for tok in rs.children:
                if isinstance(tok, Token) and tok.type == "IDENT" and rname is None:
                    rname = str(tok)
                # second IDENT is kind? grammar uses literal tokens, so skip
            for prop in rs.find_data("resource_prop"):
                k = str(prop.children[0])
                v = self._eval_expr(prop.children[1], EvalContext(variables={}, checkpoints=[]))
                props[k] = v
            if rname:
                self.tools[rname] = props

    # ---------- Chain/Process ops ----------
    def _chain_call(self, name: str, op: str, args: List[Any], kwargs: Dict[str, Any], ctx: EvalContext):
        ch = self.chains[name]
        if op == "set_label":
            key = args[0] if args else kwargs.get("key")
            val = args[1] if len(args) > 1 else kwargs.get("value")
            ch["labels"][str(key)] = val
            self.log(f"[chain] {name}.set_label {key}={val}")
            return
        if op == "get_label":
            key = args[0] if args else kwargs.get("key")
            val = ch["labels"].get(str(key))
            ctx.variables["_"] = val
            self.log(f"[chain] {name}.get_label {key} -> {val}")
            return
        if op == "set_constraint":
            key = args[0] if args else kwargs.get("key")
            val = args[1] if len(args) > 1 else kwargs.get("value")
            ch["constraints"][str(key)] = val
            self.log(f"[chain] {name}.set_constraint {key}={val}")
            return
        if op == "propagate":
            node = args[0] if args else kwargs.get("node")
            eff = args[1] if len(args) > 1 else kwargs.get("effect")
            if str(node) not in ch["nodes"]:
                raise RuntimeFlowError(f"Chain '{name}' has no node '{node}'")
            ch["effects"][str(node)] = eff
            # diffuse according to propagation
            order = ch["order"]
            decay = float(ch["propagation"].get("decay", 0.6))
            do_fwd = bool(ch["propagation"].get("forward", True))
            do_bwd = bool(ch["propagation"].get("backprop", True))
            cap = ch["propagation"].get("cap")
            try:
                idx = order.index(str(node))
            except ValueError:
                idx = None
            if idx is not None:
                # forward
                if do_fwd:
                    cur = eff
                    j = idx + 1
                    while j < len(order):
                        if isinstance(cur, (int, float)):
                            cur = float(cur) * decay
                            if cap is not None and isinstance(cap, (int, float)) and cur < float(cap):
                                break
                            ch["effects"][order[j]] = max(cur, ch["effects"].get(order[j], 0))
                        else:
                            ch["effects"][order[j]] = cur
                        j += 1
                # backward
                if do_bwd:
                    cur = eff
                    j = idx - 1
                    while j >= 0:
                        if isinstance(cur, (int, float)):
                            cur = float(cur) * decay
                            if cap is not None and isinstance(cap, (int, float)) and cur < float(cap):
                                break
                            ch["effects"][order[j]] = max(cur, ch["effects"].get(order[j], 0))
                        else:
                            ch["effects"][order[j]] = cur
                        j -= 1
            self.log(f"[chain] {name}.propagate {node} effect={eff} with decay={decay}")
            return
        self.log(f"[chain] {name}.{op} args={args} kwargs={kwargs}")

    def _process_call(self, name: str, op: str, args: List[Any], kwargs: Dict[str, Any], ctx: EvalContext):
        pr = self.processes[name]
        if op == "mark":
            node = args[0] if args else kwargs.get("node")
            status = args[1] if len(args) > 1 else kwargs.get("status")
            # Policies: require_reason (bool), allowed_status (STRING comma-separated)
            policies = pr.get("policies", {})
            if policies.get("require_reason"):
                reason = kwargs.get("reason")
                if not reason:
                    raise RuntimeFlowError(f"Process '{name}': mark requires 'reason' per policy")
            allowed = policies.get("allowed_status")
            if isinstance(allowed, str) and allowed:
                allowed_set = {s.strip() for s in allowed.split(",")}
                if str(status) not in allowed_set:
                    raise RuntimeFlowError(f"Process '{name}': status '{status}' not allowed (policy)")
            pr["marks"][str(node)] = status
            self.log(f"[process] {name}.mark {node}={status}")
            return
        if op == "expand":
            parent = args[0] if args else kwargs.get("parent")
            children = args[1] if len(args) > 1 else kwargs.get("children", [])
            if not isinstance(children, list):
                children = [children]
            # Policy: max_children (number)
            policies = pr.get("policies", {})
            max_children = policies.get("max_children")
            if isinstance(max_children, (int, float)):
                if len(children) > int(max_children):
                    raise RuntimeFlowError(f"Process '{name}': expand exceeds max_children policy ({max_children})")
            for c in children:
                pr["nodes"].setdefault(str(c), {})
            self.log(f"[process] {name}.expand {parent} -> {children}")
            return
        if op == "collapse":
            node = args[0] if args else kwargs.get("node")
            # Policies: audit_required (bool), protected_nodes (STRING comma-separated)
            policies = pr.get("policies", {})
            prot = policies.get("protected_nodes")
            if isinstance(prot, str) and prot:
                if str(node) in {s.strip() for s in prot.split(", ")}:
                    raise RuntimeFlowError(f"Process '{name}': node '{node}' is protected (policy)")
            if policies.get("audit_required"):
                if pr["marks"].get(str(node)) != "audited":
                    raise RuntimeFlowError(f"Process '{name}': collapse requires audited status (policy)")
            pr["nodes"].pop(str(node), None)
            self.log(f"[process] {name}.collapse {node}")
            return
        if op == "find":
            # Phase 9: Maestro Search
            target = args[0] if args else kwargs.get("target")
            path = self._get_binary_path(name, str(target))
            self.log(f"[process] {name}.find {target} @ path={path}")
            return path
        self.log(f"[process] {name}.{op} args={args} kwargs={kwargs}")

    def _get_binary_path(self, pname: str, nname: str) -> str:
        """Computes bit-string path (e.g. 0101) from root to node."""
        pr = self.processes.get(pname)
        if not pr: return ""
        root = pr.get("root")
        if not root: return ""
        if nname == root: return ""
        
        # BFS to find path
        queue = [(root, "")]
        visited = {root}
        while queue:
            curr, path = queue.pop(0)
            children = pr["branches"].get(curr, [])
            for i, child in enumerate(children):
                if child not in visited:
                    # Use binary convention: 0 for first child, 1 for second
                    new_path = path + str(i if len(children) <= 2 else bin(i)[2:])
                    if child == nname:
                        return new_path
                    visited.add(child)
                    queue.append((child, new_path))
        return ""

    def _tool_call(self, name: str, op: str, args: List[Any], kwargs: Dict[str, Any], ctx: EvalContext):
        tool = self.tools[name]
        if op == "run":
            task = args[0] if args else kwargs.get("task", "")
            # simulate external tool execution
            result = TypedValue(tag=ValueTag.TryResult, meta={"output": f"tool:{name}:{task}", "metrics": {"time": 0.5}})
            ctx.variables["_"] = result
            self.metrics["actions"] += 1
            self.metrics["verbs"]["tool.run"] = self.metrics["verbs"].get("tool.run", 0) + 1
            self.log(f"[tool] {name}.run task={task} -> {result}")
            return
        self.log(f"[tool] {name}.{op} args={args} kwargs={kwargs}")

    def _exec_action(self, node: Tree, ctx: EvalContext):
        # IDENT '.' command_action
        team = str(node.children[0])
        action = node.children[1]
        verb = str(action.children[0])
        
        # parse args (needed for logging even in dry run)
        args: List[Any] = []
        kwargs: Dict[str, Any] = {}
        if len(action.children) > 1 and isinstance(action.children[1], Tree) and action.children[1].data == "arg_list":
             for item in action.children[1].children:
                 if isinstance(item, Tree) and item.data == "named_arg":
                     k = str(item.children[0])
                     v = self._eval_expr(item.children[1], ctx)
                     kwargs[k] = v
                 else:
                     v = self._eval_expr(item.children[0], ctx)
                     args.append(v)

        # 6. Concept Alignment: Strict Team Typing
        # Map verb to kind
        verb_kind_map = {
            "ask": "Communicate",
            "search": "Search",
            "try": "Try",
            "judge": "Judge"
        }
        
        # Check team definition
        team_info = self.teams.get(team)
        # Only check strictness if team is registered (to support dynamic/legacy teams)
        if team_info:
             kind = team_info.get("kind")
             req_kind = verb_kind_map.get(verb)
             if kind and req_kind and kind != req_kind:
                 raise RuntimeFlowError(f"Team '{team}' is specialized for {kind}, cannot perform '{verb}' ({req_kind})")

        req_kind = verb_kind_map.get(verb)

        # 6. Concept Alignment: Variables as Orders (Batch Processing)
        # If the first argument is a List of Orders (or just a list), we iterate (The Zone)
        # Unless it's a specific verb that takes a list as a single arg (like 'judge' comparing list?)
        # For now, we assume implicit mapping if arg0 is list and strict typing is on.
        
        is_batch = False
        batch_items = []
        if args and isinstance(args[0], list):
             is_batch = True
             batch_items = args[0]
        elif args and isinstance(args[0], Order):
             # Handle single order as if it's the only item in a batch for logic reuse
             is_batch = True
             batch_items = [args[0]]
             args = args[1:] # Adjust args for the loop

        if is_batch:
            self.log(f"[{team}.{verb}] Entering Zone: Processing batch of {len(batch_items)} items")
            results = []
            for item in batch_items:
                 i_args = [item] + args[1:]
                 # If item is not an Order, try promotion
                 if not isinstance(item, Order):
                     item = Order(id=f"auto_{len(results)}", payload=item, kind=req_kind or CommandKind.Try)
                 
                 # 8. Data Chains: Exclusive Activity Check
                 skip_reason = None
                 if item.chain_node:
                      for cname, cinfo in self.chains.items():
                          if item.chain_node in cinfo["nodes"]:
                              eff = cinfo["effects"].get(item.chain_node)
                              if eff in ("satisfied", "skip", "fixed"):
                                  skip_reason = str(eff)
                                  break
                 
                 if skip_reason:
                      self.log(f"[{team}.{verb}] Exclusive Activity: Skipping '{item.chain_node}' (state={skip_reason})")
                      res_val = item.payload # Reuse payload as result
                      member_idx = -1
                 else:
                      # Maestro Path Awareness: pass binary path to AI
                      if item.process_node:
                          for pname, pinfo in self.processes.items():
                              if item.process_node in pinfo["nodes"]:
                                  kwargs["maestro_path"] = self._get_binary_path(pname, item.process_node)
                                  break
                      
                       item.log_activity(team, verb, "Processing...")
                       
                       if self.dry_run:
                           self.log(f"[dry_run] Skip {team}.{verb}")
                           res_val = TypedValue(ValueTag.Unknown, meta={"text": "dry_run"})
                           member_idx = self._select_team_member(team) # Still increment index for sequential handover
                       else:
                           res_val, member_idx = self._execute_single_action(team, verb, i_args, kwargs, ctx)
                       
                       item.log_activity(team, verb, res_val, member_idx=member_idx)
                      
                      # 8. Data Chains: Implicit Communication
                      if item.chain_node:
                          for cname, cinfo in self.chains.items():
                              if item.chain_node in cinfo["nodes"]:
                                  # Propagate completion through the System Sequence
                                  self._chain_call(cname, "propagate", [item.chain_node, "satisfied"], {}, ctx)
                                  break
                      
                      # 9. Process Trees: Maestro Auditing
                      if item.process_node:
                          for pname, pinfo in self.processes.items():
                              if item.process_node in pinfo["nodes"]:
                                  # Record accomplishment in the family tree
                                  pinfo["marks"][item.process_node] = f"Accomplished: {team}.{verb}"
                                  self.log(f"[maestro] Mapped Order to Process Branch: {pname}/{item.process_node}")
                                  break
                 
                 results.append(res_val)
            
            final_result = TypedValue(ValueTag.List, value=results)
        else:
            if self.dry_run:
                self.log(f"[dry_run] Skip {team}.{verb}")
                final_result = TypedValue(ValueTag.Unknown, meta={"text": "dry_run"})
                member_idx = self._select_team_member(team)
            else:
                final_result, member_idx = self._execute_single_action(team, verb, args, kwargs, ctx)
            
            # 6. Concept Alignment: Result Promotion to Order
            # If the result is not already an Order, wrap it so it has a lifecycle
            if not isinstance(final_result, Order) and verb in verb_kind_map:
                 final_result = Order(
                     id=f"order_{self.metrics['actions']}",
                     payload=final_result,
                     kind=req_kind
                 )
        
        ctx.variables["_"] = final_result
        # Update monologue history if it was updated in kwargs
        if "history" in kwargs:
             ctx.variables["__monologue_history__"] = kwargs["history"]

    def _execute_single_action(self, team: str, verb: str, args: List[Any], kwargs: Dict[str, Any], ctx: EvalContext) -> Tuple[Any, int]:
        member_idx = self._select_team_member(team)
        if verb == "ask":
             prev = ctx.variables.get("__monologue_history__", [])
             prompt = args[0]
             if hasattr(prompt, "payload"): 
                 prompt = prompt.payload
             prev.append(str(prompt))
             kwargs["history"] = prev

        if self.tracer:
             with self.tracer.start_as_current_span(f"action:{team}.{verb}"):
                 result = self._dispatch_provider(team, verb, args, kwargs, member_idx)
        else:
             result = self._dispatch_provider(team, verb, args, kwargs, member_idx)
        
        # Metrics and Logging
        self.metrics["actions"] += 1
        self.metrics["verbs"][verb] = self.metrics["verbs"].get(verb, 0) + 1
        self.log(f"[{team}#{member_idx or 0}.{verb}] -> {result}")
        return result, member_idx

    def _dispatch_provider(self, team: str, verb: str, args: List[Any], kwargs: Dict[str, Any], member_idx: int) -> Tuple[Any, int]:
        if self.ai_provider:
            return self.ai_provider.execute(team, verb, args, kwargs), member_idx
        elif self.ai_client:
            return self._ai_command(team, verb, args, kwargs, member_idx), member_idx
        else:
            return self._fake_command(team, verb, args, kwargs, member_idx), member_idx


    def _ai_command(self, team: str, verb: str, args: List[Any], kwargs: Dict[str, Any], member_idx: int | None = None) -> Any:
        """Execute a verb through an AI model. All teams route here; team is only contextual.

        This uses OpenAI chat.completions synchronously via the official client. If the model
        returns JSON and we can parse it, we map it into the expected TypedValue meta; otherwise,
        we wrap the raw text content.
        """
        if not self.ai_client:
            return self._fake_command(team, verb, args, kwargs, member_idx)

        # Choose model per verb with fallback
        model = self.ai_model_per_verb.get(verb, self.ai_default_model)

        # Build structured instruction to encourage JSON outputs compatible with our ValueTags
        system_prompts: Dict[str, str] = {
            "ask": (
                "You are a helpful assistant. Answer clearly. Respond with JSON: {\n"
                "  \"text\": string,\n  \"history\": array\n}"
            ),
            "search": (
                "You are an information retrieval agent. Return JSON: {\n"
                "  \"hits\": array of strings\n} with representative URIs or titles."
            ),
            "try": (
                "You execute tasks and report results. Respond with JSON: {\n"
                "  \"output\": string,\n  \"metrics\": {\"time\": number}\n}"
            ),
            "judge": (
                "You evaluate against criteria and return JSON: {\n"
                "  \"score\": number,\n  \"confidence\": number,\n  \"pass\": boolean\n}"
            ),
        }
        system_msg = system_prompts.get(verb, "You are an AI that executes the requested command. Prefer JSON outputs.")

        # Compose user message based on verb/args/kwargs
        # Preserve the existing 'ask' history flow if provided
        if verb == "ask":
            prompt = args[0] if args else kwargs.get("prompt", "")
            history = kwargs.get("history", [])
            user_content = {
                "verb": verb,
                "team": team,
                "prompt": str(prompt),
                "history": list(history) if isinstance(history, list) else [],
                "options": {k: v for k, v in kwargs.items() if k not in ("prompt", "history")},
            }
        elif verb == "search":
            query = args[0] if args else kwargs.get("query", "")
            user_content = {"verb": verb, "team": team, "query": str(query), "options": kwargs}
        elif verb == "try":
            task = args[0] if args else kwargs.get("task", "")
            user_content = {"verb": verb, "team": team, "task": str(task), "options": kwargs}
        elif verb == "judge":
            target = args[0] if args else kwargs.get("target", "")
            criteria = args[1] if len(args) > 1 else kwargs.get("criteria", "score")
            user_content = {"verb": verb, "team": team, "target": target, "criteria": criteria, "options": kwargs}
        else:
            user_content = {"verb": verb, "team": team, "args": args, "options": kwargs}

        # Call the model
        try:
            resp = self.ai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": json.dumps(user_content)},
                ],
                temperature=float(kwargs.get("temperature", 0.7)) if isinstance(kwargs.get("temperature", 0.7), (int, float)) else 0.7,
                max_tokens=int(kwargs.get("max_tokens", 1000)) if str(kwargs.get("max_tokens", "")).isdigit() else 1000,
            )
            content = resp.choices[0].message.content if resp and resp.choices else ""
        except Exception as e:  # pragma: no cover - external dependency runtime error path
            # On any API error, surface a typed error as Unknown
            return TypedValue(tag=ValueTag.Unknown, meta={"error": str(e), "verb": verb, "args": args, "kwargs": kwargs})

        # Try to parse JSON content
        parsed: Any
        try:
            parsed = json.loads(content) if content else {}
        except Exception:
            parsed = None

        # Map to typed values
        if verb == "ask":
            if isinstance(parsed, dict) and ("text" in parsed or "history" in parsed):
                meta = {"text": parsed.get("text", content or ""), "history": parsed.get("history", kwargs.get("history", []))}
            else:
                meta = {"text": content, "history": kwargs.get("history", [])}
            return TypedValue(tag=ValueTag.CommunicateResult, meta=meta)
        if verb == "search":
            if isinstance(parsed, dict) and "hits" in parsed:
                meta = {"hits": parsed.get("hits", [])}
            else:
                # Fallback: single string content as one hit
                meta = {"hits": [content] if content else []}
            return TypedValue(tag=ValueTag.SearchResult, meta=meta)
        if verb == "try":
            if isinstance(parsed, dict) and ("output" in parsed or "metrics" in parsed):
                meta = {"output": parsed.get("output", content or ""), "metrics": parsed.get("metrics", {})}
            else:
                meta = {"output": content, "metrics": {}}
            return TypedValue(tag=ValueTag.TryResult, meta=meta)
        if verb == "judge":
            if isinstance(parsed, dict) and ("score" in parsed or "confidence" in parsed or "pass" in parsed):
                score = parsed.get("score", 0.0)
                conf = parsed.get("confidence", 0.0)
                passed = parsed.get("pass", bool(score))
            else:
                score = 0.0
                conf = 0.0
                passed = False
            return TypedValue(tag=ValueTag.JudgeResult, meta={"score": score, "confidence": conf, "pass": passed})

        # Unknown verbs: return textual content wrapped
        return TypedValue(tag=ValueTag.Unknown, meta={"text": content, "args": args, "kwargs": kwargs})

    def _fake_command(self, team: str, verb: str, args: List[Any], kwargs: Dict[str, Any], member_idx: int | None = None) -> Any:
        # Prototype mock implementations
        if verb == "ask":
            prompt = args[0] if args else kwargs.get("prompt", "")
            # دعم سجل الحوار الذاتي
            history = kwargs.get("history")
            if not history:
                history = []
            history.append(str(prompt))
            return TypedValue(tag=ValueTag.CommunicateResult, meta={"text": str(prompt), "history": history})
        if verb == "search":
            query = args[0] if args else kwargs.get("query", "")
            return TypedValue(tag=ValueTag.SearchResult, meta={"hits": [f"doc://{i}:{query}" for i in range(3)]})
        if verb == "try":
            descr = args[0] if args else kwargs.get("task", "")
            return TypedValue(tag=ValueTag.TryResult, meta={"output": f"ran:{descr}", "metrics": {"time": 1.23}})
        if verb == "judge":
            target = args[0] if args else kwargs.get("target", None)
            crit = args[1] if len(args) > 1 else kwargs.get("criteria", "score")
            conf = 0.85 if target else 0.6
            passed = conf >= 0.7
            return TypedValue(tag=ValueTag.JudgeResult, meta={"score": 0.8 if passed else 0.5, "confidence": conf, "pass": passed})
        return TypedValue(tag=ValueTag.Unknown, meta={"args": args, "kwargs": kwargs})

    def _select_team_member(self, team: str) -> int:
        info = self.teams.get(team)
        if not info:
            return 0
        size = int(info.get("size", 1))
        if size <= 0:
            size = 1
        dist = info.get("distribution", "round_robin")
        if dist == "round_robin":
            idx = info.setdefault("_idx", 0)
            sel = idx % size
            info["_idx"] = idx + 1
            return sel
        if dist == "priority" or dist == "weighted":
            # simple placeholder: always pick #0
            return 0
        return 0

    # ---------- Expressions ----------
    def _eval_expr(self, node: Tree | Token, ctx: EvalContext):
        if isinstance(node, Token):
            # name tokens handled in primary/name rule
            return str(node)
        if isinstance(node, Tree):
            dt = node.data
            if dt in ("expr", "or_expr", "and_expr", "cmp_expr", "add_expr", "mul_expr", "unary_expr"):
                # child reduction
                if len(node.children) == 1:
                    return self._eval_expr(node.children[0], ctx)
                # handle binary ops
                if dt == "unary_expr" and len(node.children) == 2 and isinstance(node.children[0], Token):
                    op = str(node.children[0])
                    val = self._eval_expr(node.children[1], ctx)
                    return self._unary(op, val)
                # For add/mul/cmp/or/and, do left-assoc
                return self._eval_binary_chain(node, ctx)
            if dt == "number":
                return float(node.children[0]) if "." in str(node.children[0]) else int(node.children[0])
            if dt == "string":
                s = node.children[0]
                return s[1:-1]
            if dt == "boolean":
                return True if str(node.children[0]) == "true" else False
            if dt == "name":
                name = str(node.children[0])
                return ctx.variables.get(name)
            if dt == "list_literal":
                return [self._eval_expr(ch, ctx) for ch in node.find_data("expr")]
            if dt == "dict_literal":
                # construct from dict_kv
                result: Dict[str, Any] = {}
                for kv in node.find_data("dict_kv"):
                    key_node = kv.children[0]
                    key = key_node.value if isinstance(key_node, Token) else str(key_node)
                    val = self._eval_expr(kv.children[1], ctx)
                    result[str(key).strip('"')] = val
                return result
            if dt == "primary":
                # support team.method(...) in expressions
                base_node = node.children[0]
                base_val = self._eval_expr(base_node, ctx)
                accum = base_val
                # Detect pattern: NAME . VERB (args) where NAME is a known team
                if isinstance(base_node, Tree) and base_node.data == "name":
                    team_name = str(base_node.children[0])
                    # look ahead children to see field then call
                    if len(node.children) >= 3:
                        f1, c1 = node.children[1], node.children[2]
                        if isinstance(f1, Tree) and f1.data == "field" and isinstance(c1, Tree) and c1.data == "call":
                            verb = str(f1.children[0])
                            if team_name in self.teams:
                                # parse args from c1
                                args = []
                                if c1.children:
                                    arglist = c1.children[0]
                                    if isinstance(arglist, Tree) and arglist.data == "expr_list":
                                        for e in arglist.children:
                                            args.append(self._eval_expr(e, ctx))
                                res = self._fake_command(team_name, verb, args, {}, self._select_team_member(team_name))
                                self.metrics["actions"] += 1
                                self.metrics["verbs"][verb] = self.metrics["verbs"].get(verb, 0) + 1
                                return res
                for acc in node.children[1:]:
                    if isinstance(acc, Tree):
                        if acc.data == "field":
                            fld = str(acc.children[0])
                            # تعديل: إذا كان accum من نوع TypedValue، نصل للـ meta
                            if isinstance(accum, TypedValue):
                                if accum.meta and fld in accum.meta:
                                    accum = accum.meta[fld]
                                else:
                                    raise RuntimeFlowError(f"Field '{fld}' not found on TypedValue {accum}")
                            elif isinstance(accum, dict):
                                if fld in accum:
                                    accum = accum[fld]
                                else:
                                    raise RuntimeFlowError(f"Field '{fld}' not found on object {accum}")
                            else:
                                raise RuntimeFlowError(f"Cannot access field '{fld}' on non-object: {accum}")
                        elif acc.data == "call":
                            # simple call support: if callable function name present in env
                            args = []
                            if acc.children:
                                # children[0] may be expr_list
                                arglist = acc.children[0]
                                if isinstance(arglist, Tree) and arglist.data == "expr_list":
                                    for e in arglist.children:
                                        args.append(self._eval_expr(e, ctx))
                            if callable(accum):
                                accum = accum(*args)
                            else:
                                # not callable -> keep as is
                                pass
                return accum
        raise RuntimeFlowError(f"Unsupported expression node: {node}")

    def _eval_binary_chain(self, node: Tree, ctx: EvalContext):
        # evaluates left-associative binary expressions
        # children alternate: term (op term)*
        def eval_child(i):
            return self._eval_expr(node.children[i], ctx)
        # start with first child
        acc = eval_child(0)
        i = 1
        while i < len(node.children):
            op = str(node.children[i])
            rhs = eval_child(i + 1)
            acc = self._apply_bin_op(op, acc, rhs)
            i += 2
        return acc

    def _apply_bin_op(self, op: str, a: Any, b: Any):
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            return a / b
        if op == "==":
            return a == b
        if op == "!=":
            return a != b
        if op == "<":
            return a < b
        if op == ">":
            return a > b
        if op == "<=":
            return a <= b
        if op == ">=":
            return a >= b
        if op == "&&":
            return bool(a) and bool(b)
        if op == "||":
            return bool(a) or bool(b)
        raise RuntimeFlowError(f"Unknown operator {op}")

    def _unary(self, op: str, v: Any):
        if op == "-":
            return -v
        if op == "!":
            return not bool(v)
        raise RuntimeFlowError(f"Unknown unary {op}")

    def _truthy(self, v: Any) -> bool:
        return bool(v)
