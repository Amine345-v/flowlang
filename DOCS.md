# FlowLang (مسير) – Full Documentation

## Overview
FlowLang is a domain-specific language (DSL) for orchestrating professional workflows, blending management science and digital product development. It models flows, teams, chains, process trees, checkpoints, and context, with rich control logic and advanced system structures.

---

## Key Features & Properties

### 1. Results & Types
- **result**: Structured output type for team actions (e.g., `JudgeResult`, `SearchResult`, `TryResult`, `CommunicateResult`).
- **type Command<KIND>**: Declares a command type for a team (`Search`, `Try`, `Judge`, `Communicate`).
- **TypedValue**: All results are returned as `TypedValue` objects, with a `tag` (type) and `meta` (fields).
- **Custom Fields**: You can define custom fields for each result type.
- **Type System**: Supports basic types (`number`, `string`, `boolean`, `list`, `dict`), and composite types (`Option`, `Union`).

### 2. Teams
- **team**: Represents a group of agents with a command kind, size, distribution, role, and policy.
- **Distribution**: Supports `round_robin`, `weighted`, `priority` for team member selection.
- **Role/Policy**: Teams can be assigned roles and policies for governance and access control.

### 3. Chains
- **chain**: Models a sequence of nodes (workflow stages) with propagation settings, labels, and constraints.
- **Propagation**: Causal propagation with `decay`, `backprop`, `forward`, and optional `cap`.
- **Labels/Constraints**: Key-value metadata and numeric/boolean constraints.
- **System Calls**: Chains support system operations: `set_label`, `get_label`, `set_constraint`, `propagate`.
- **Effect Diffusion**: Effects can be propagated forward/backward through chain nodes with decay.

### 4. Process Trees
- **process**: Hierarchical structure of tasks, branches, nodes, and policies.
- **Branching**: Define branches and child nodes for complex workflows.
- **Node Properties**: Each node can have custom properties.
- **Policy/Audit**: Processes can have policies and audit capabilities.
- **System Calls**: Operations include `mark`, `expand`, `collapse`, `audit`.

### 5. Resources
- **resource**: External tools, codebases, datasets, or metrics stores accessible in flows.
- **Tool Integration**: Tools can be called via system calls (e.g., `CodeTool.run("task")`).

### 6. Flows & Checkpoints
- **flow**: Main workflow, composed of checkpoints (stages), using teams, and specifying merge policies.
- **Checkpoints**: Stages in the flow, each with its own local logic and context.
- **Context Retention**: Context can be retained at checkpoints for stateful execution.
- **Merge Policy**: Controls how contexts are merged in parallel/race blocks (`last_wins`, `deep_merge`, `crdt`).
- **Distribution Policy**: Controls how work is distributed among teams.

### 7. Programming Model
- **Action Statements**: `team.verb(args...)` (e.g., `DevJudges.judge(S1, "criteria")`).
- **Assignment**: `var = expr` (assign result to variable).
- **Context Update**: `context.update(var1, var2, ...)` (update context with new values).
- **Chain/Process Ops**: `Chain.set_label(key, value)`, `Process.mark(node, status)`, etc.
- **Deploy/Audit**: `deploy(Model=..., env=...)`, `ProductTree.audit()`.
- **Control Flow**: `if`, `while`, `for`, `par`, `race`, `flow.back_to`, `flow.end`.
- **Parallelism**: `par` for parallel execution, `race` for competitive execution (winner merges context).
- **Variable Types**: All variables can hold any result, including TypedValue, lists, dicts, numbers, etc.

### 8. Data Structures
- **Variables**: Store results, context, and intermediate values.
- **TypedValue**: Results from actions are always `TypedValue` (access fields via `.meta`).
- **Lists/Dicts**: Standard Python-like lists and dicts supported.
- **System Structures**: Teams, chains, process trees replace classic data structures for workflow modeling.

### 9. Advanced Features
- **Monologue/Dialogue**: `ask` actions maintain a history in context (`__monologue_history__`).
- **Field Access**: Access result fields via `.meta["field"]` (e.g., `J1.meta["confidence"]`).
- **Custom Merge Policies**: Extend merging logic for context variables as needed.
- **Governance**: Deployments are checked for team capabilities and chain constraints.
- **Semantic Analysis**: Ensures type and field correctness, team-action compatibility, and valid references.
- **Metrics**: Execution metrics (actions, checkpoints, timings) are recorded for analysis.
- **Error Handling**: Semantic and runtime errors are reported with context.

### 10. Extensibility
- **Add Command Kinds**: Extend `types.py` for new team actions.
- **Custom Result Types**: Define new result types and fields.
- **Grammar Extension**: Add new statements or control structures in `grammar.lark`.
- **Tool Integration**: Implement real integrations for resources/tools.
- **Merge Strategies**: Add more merge policies or context management strategies.

---

## Example Program
```flow
result JudgeResult { confidence: number; score: number; pass: boolean; };
result TryResult { output: string; metrics: dict; };
result SearchResult { hits: list; };
result CommunicateResult { text: string; };

team DevJudges: Command<Judge> [size=1, role=Developer, policy=Security];
chain ModelUpdateChain {
  nodes: [Design, Training, Evaluation, Deployment];
  propagation: causal(decay=0.6, backprop=true, forward=true);
  labels: { priority: "high", owner: "ml-team" };
  constraints: { max_runtime: 3600; require_eval: true; };
}
process ProductTree "NLP Probabilistic Model" {
  root: "Root";
  branch "Design" -> ["Model_Structure", "Data_Schema"];
  node "Implement" { owner: "devA" };
  policy: { risk: 0.2 };
  audit: enabled;
}
flow BuildProbModel(using: DevSearchers, DevExperimenters, DevJudges, DevMonologue) {
  merge_policy: deep_merge;
  checkpoint "Understand" {
    report = DevMonologue.ask("ما المطلوب؟");
    S1 = DevSearchers.search("نماذج احتمالية");
    J1 = DevJudges.judge(S1, "ملاءمة");
    context.update(report, S1, J1);
  }
  checkpoint "Design" {
    params = {"alpha": 0.1, features: ["unigram", "bigram"]};
    T1 = DevExperimenters.try("نموذج أولي", params);
    J2 = DevJudges.judge(T1, "استقرار");
    conf = J2.meta["confidence"];
    context.update(params, T1, J2, conf);
    ModelUpdateChain.touch("Design", effect=conf);
    ModelUpdateChain.propagate("Evaluation", conf);
    if (conf >= 0.7) {
      deploy(Model="ProbNLP", env="staging");
    } else {
      alt = DevExperimenters.try("تحسين الإعدادات", {"alpha": 0.2});
      J3 = DevJudges.judge(alt, "F1");
      context.update(alt, J3, J3.meta["confidence"]);
      ModelUpdateChain.touch("Training", effect=J3.meta["confidence"]);
    }
    tries = ["A", "B", "C"];
    i = 0;
    while (i < 2) {
      tmp = DevExperimenters.try("loop-try", {idx: i});
      i = i + 1;
    }
    for item in tries {
      _ = DevSearchers.search(item);
    }
    CodeTool.run("scan-codebase");
    ProductTree.mark("Implement", "in_progress");
    ProductTree.expand("Design", ["HyperParam_Tuning", "Schema_Review"]);
    ProductTree.audit();
  }
}
```

---

## Quickstart (README)

Follow these steps to run FlowLang locally and execute flows with AI-backed verbs.

### 1) Install

```bash
python -m venv .venv
# Windows PowerShell
. .venv/Scripts/Activate.ps1
pip install -U pip
pip install -e .
```

### 2) Configure AI

Set your OpenAI API key (required for real execution; without it, the runtime falls back to mock responses):

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

Optional model overrides (global or per verb):

```powershell
# Global default model
$env:FLOWLANG_AI_MODEL = "gpt-5.2"

# Per-verb overrides
$env:FLOWLANG_AI_MODEL_ASK = "gpt-5.2"
$env:FLOWLANG_AI_MODEL_SEARCH = "gpt-5.2"
$env:FLOWLANG_AI_MODEL_TRY = "gpt-5.2"
$env:FLOWLANG_AI_MODEL_JUDGE = "gpt-5.2"
```

#### Other AI Providers (PowerShell examples)

FlowLang routes all verbs through an AI executor. OpenAI is wired by default, but you can adapt to other providers. Below are concrete PowerShell examples for environment variables. You can also introduce `FLOWLANG_AI_PROVIDER` to force a provider selection in your adapter.

- Anthropic (Claude)

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
# Global default model
$env:FLOWLANG_ANTHROPIC_MODEL = "claude-sonnet-5"
# Per-verb overrides
$env:FLOWLANG_ANTHROPIC_MODEL_ASK = "claude-sonnet-5"
$env:FLOWLANG_ANTHROPIC_MODEL_SEARCH = "claude-sonnet-5"
$env:FLOWLANG_ANTHROPIC_MODEL_TRY = "claude-sonnet-5"
$env:FLOWLANG_ANTHROPIC_MODEL_JUDGE = "claude-sonnet-5"
```

- Google Gemini

```powershell
$env:GEMINI_API_KEY = "g-..."
# Global default model
$env:FLOWLANG_GEMINI_MODEL = "gemini-3-flash"
# Per-verb overrides
$env:FLOWLANG_GEMINI_MODEL_ASK = "gemini-3-flash"
$env:FLOWLANG_GEMINI_MODEL_SEARCH = "gemini-3-flash"
$env:FLOWLANG_GEMINI_MODEL_TRY = "gemini-3-flash"
$env:FLOWLANG_GEMINI_MODEL_JUDGE = "gemini-3-flash"
```

- Mistral

```powershell
$env:MISTRAL_API_KEY = "mis-..."
# Global default model
$env:FLOWLANG_MISTRAL_MODEL = "mistral-large-latest"
# Per-verb overrides
$env:FLOWLANG_MISTRAL_MODEL_ASK = "mistral-large-latest"
$env:FLOWLANG_MISTRAL_MODEL_SEARCH = "mistral-small-latest"
$env:FLOWLANG_MISTRAL_MODEL_TRY = "mistral-small-latest"
$env:FLOWLANG_MISTRAL_MODEL_JUDGE = "mistral-large-latest"
```

- Cohere

```powershell
$env:COHERE_API_KEY = "coh-..."
# Global default model
$env:FLOWLANG_COHERE_MODEL = "command-r-plus"
# Per-verb overrides
$env:FLOWLANG_COHERE_MODEL_ASK = "command-r-plus"
$env:FLOWLANG_COHERE_MODEL_SEARCH = "command-r"
$env:FLOWLANG_COHERE_MODEL_TRY = "command-r-plus"
$env:FLOWLANG_COHERE_MODEL_JUDGE = "command-r-plus"
```

- Azure OpenAI

```powershell
$env:AZURE_OPENAI_API_KEY = "aoai-..."
$env:AZURE_OPENAI_ENDPOINT = "https://your-azure-openai.openai.azure.com/"
$env:AZURE_OPENAI_DEPLOYMENT = "gpt-5.2-deployment"
# Optional global default
$env:FLOWLANG_AZURE_OPENAI_MODEL = "gpt-5.2"
# Per-verb overrides (optional)
$env:FLOWLANG_AZURE_OPENAI_MODEL_ASK = "gpt-5.2"
$env:FLOWLANG_AZURE_OPENAI_MODEL_SEARCH = "gpt-5.2"
$env:FLOWLANG_AZURE_OPENAI_MODEL_TRY = "gpt-5.2"
$env:FLOWLANG_AZURE_OPENAI_MODEL_JUDGE = "gpt-5.2"
```

- OpenRouter (multi-model broker)

```powershell
$env:OPENROUTER_API_KEY = "or-..."
# Global default routed model
$env:FLOWLANG_OPENROUTER_MODEL = "openrouter/anthropic/claude-sonnet-5"
# Per-verb overrides
$env:FLOWLANG_OPENROUTER_MODEL_ASK = "openrouter/openai/gpt-5.2"
$env:FLOWLANG_OPENROUTER_MODEL_SEARCH = "openrouter/google/gemini-3-flash"
$env:FLOWLANG_OPENROUTER_MODEL_TRY = "openrouter/mistral/mistral-large-latest"
$env:FLOWLANG_OPENROUTER_MODEL_JUDGE = "openrouter/anthropic/claude-sonnet-5"
```

- Local (Ollama)

```powershell
$env:OLLAMA_HOST = "http://localhost:11434"
# Global default model
$env:FLOWLANG_OLLAMA_MODEL = "llama3.1:8b"
# Per-verb overrides
$env:FLOWLANG_OLLAMA_MODEL_ASK = "llama3.1:8b"
$env:FLOWLANG_OLLAMA_MODEL_SEARCH = "llama3.1:8b"
$env:FLOWLANG_OLLAMA_MODEL_TRY = "phi3:3.8b"
$env:FLOWLANG_OLLAMA_MODEL_JUDGE = "llama3.1:70b"
```

Notes:
- Keep per-verb JSON contracts stable across providers: ask `{text, history}`, search `{hits}`, try `{output, metrics}`, judge `{score, confidence, pass}`.
- If you implement a provider selector, consider a `FLOWLANG_AI_PROVIDER` env to force a specific provider.


### 3) Run a Flow

Use the Python API to run a `.flow` program:

```python
from flowlang.runtime import Runtime

rt = Runtime()
rt.load("flowlang/examples/hospital.flow")
rt.run_flow()  # or rt.run_flow("YourFlowName")

# Console output is also stored here:
for line in rt.console:
    print(line)
```

Notes:
- All verbs (e.g., `ask`, `search`, `try`, `judge`, and any custom verb) are routed to the AI executor by default.
- When AI is enabled (OPENAI_API_KEY set), the runtime prompts the model to emit JSON matching typed result shapes.
- Without a key, the runtime returns mock `TypedValue` objects so you can test control-flow and structure.

### 4) Typed Results

Results are returned as `TypedValue` with a `ValueTag` and `meta` fields:

- `ask` -> `CommunicateResult` with `meta = { text, history }`
- `search` -> `SearchResult` with `meta = { hits }`
- `try` -> `TryResult` with `meta = { output, metrics }`
- `judge` -> `JudgeResult` with `meta = { score, confidence, pass }`
- Unknown verbs -> `Unknown` with textual content in `meta.text`

### 5) Concurrency and Merge Policies

- Use `par { ... }` for parallel branches and `race { ... }` for competitive execution.
- Choose a `merge_policy` in the flow header: `last_wins`, `deep_merge`, or `crdt`.

### 6) Troubleshooting

- If you see mock outputs, ensure `OPENAI_API_KEY` is set in the same shell before running Python.
- For large outputs, tune `temperature` and `max_tokens` via verb kwargs (e.g., `ask("q", temperature=0.2, max_tokens=500)`).
- Check `rt.metrics` for action counts and verb frequencies.

---

## Runtime & Execution
- **Runtime**: Loads a `.flow` file, builds all structures, and executes the flow step-by-step.
- **EvalContext**: Holds variables, checkpoints, reports, and merge policy for each flow.
- **Actions**: All team actions return `TypedValue` (access fields via `.meta`).
- **Chain/Process Ops**: Methods for updating labels, constraints, propagating effects, marking nodes, etc.
- **Parallel/Async**: `par` and `race` blocks use Python's `asyncio` for concurrency.
- **Metrics**: Execution metrics (actions, checkpoints, timings) are recorded.
- **Error Handling**: Semantic and runtime errors are reported with context.

---

## Best Practices
- Always define result types for each team action.
- Use merge policies to control context merging in parallel/race blocks.
- Access result fields via `.meta` on `TypedValue`.
- Use checkpoints to structure your workflow and maintain context.
- Leverage chains and process trees for advanced workflow modeling.
- Use roles and policies for governance and access control.
- Use context updates to maintain state across checkpoints.

---

## Troubleshooting
- **Unknown field access**: Use `.meta["field"]` for TypedValue results.
- **Deploy blocked**: Check team roles for `deploy` capability and chain constraints.
- **Type errors**: Ensure all assignments and field accesses match declared result types.
- **Semantic errors**: Check for valid team-action compatibility and references.

---

## Extending FlowLang
- Add new command kinds or result types in `types.py`.
- Extend grammar for new statements or control structures.
- Implement real integrations for resources/tools.
- Add more merge policies or context management strategies.
- Extend semantic analysis for custom rules.

---

## File Structure
- `flowlang/grammar.lark` – Language grammar
- `flowlang/runtime.py` – Main runtime engine
- `flowlang/types.py` – Type system and TypedValue
- `flowlang/semantic.py` – Semantic analysis
- `flowlang/examples/example1.flow` – Example flow program
- `flowlang/tests/` – Test suite
- `flowlang/ir.py`, `flowlang/lowering.py` – Intermediate representation (optional)
- `flowlang/README.md` – Quickstart and summary

---

## License
MIT
