# FlowLang Technical Specification v1.0

## 1. Philosophy: Occupational Programming

FlowLang reimagines programming constructs through the lens of **Management Science** and **Digital Product Development**. Instead of processing raw data, FlowLang orchestrates *intelligent agents* executing *commands* within a *system structure*.

| Traditional Construct | FlowLang Equivalent | Rationale |
|-----------------------|---------------------|-----------|
| `variable` | `Command` | Commands are mutable; they pass through processing stages that may alter their structure. |
| `array` / `table` | `team` | A homogeneous group of commands of the same type, enabling focused execution ("zone"). |
| `linked list` | `chain` | Causal propagation with bidirectional ripple effects; modifications echo through neighbors. |
| `tree` | `process` | Hierarchical audit structure; maps completed/remaining/extraneous work. |
| `for` / `while` | `flow` + `checkpoint` | Contextual iteration; checkpoints produce cumulative reports, offloading memory to the flow. |

---

## 2. Core Primitives

### 2.1 Command Types (Variables)

```flow
type Command<Search>;   // Information retrieval
type Command<Try>;      // Experimentation
type Command<Judge>;    // Evaluation/Decision
type Command<Communicate>; // Monologue (self-dialogue Q&A)
```

**Monologue (`Communicate`)**: A self-reflective dialog where the agent questions the task before executing Search, Try, or Judge.

### 2.2 Teams (Homogeneous Tables)

```flow
team DevSearchers: Command<Search> [size=2, distribution=round_robin];
```

- **size**: Number of parallel agents.
- **distribution**: `round_robin` | `weighted` | `priority`.
- **role / policy**: Governance constraints.

Teams enable "zone" focus—a single command type processed in a tight loop.

### 2.3 Chains (Linked Lists with Causal Propagation)

```flow
chain ModelUpdateChain {
  nodes: [Design, Training, Evaluation, Deployment];
  propagation: causal(decay=0.6, backprop=true, forward=true);
}
```

- **decay**: Effect attenuation per hop.
- **backprop / forward**: Bidirectional ripple.
- **cap**: Minimum threshold to halt propagation.

Chains reduce reliance on global memory; each node receives context from its neighbors.

### 2.4 Process Trees (Audit Maps)

```flow
process ProductTree "NLP Model" {
  root: "Root";
  branch "Design" -> ["Model_Structure", "Data_Schema"];
  node "Implement" { owner: "devA" };
  policy: { risk: 0.2 };
  audit: enabled;
}
```

Operations: `mark`, `expand`, `collapse`, `audit`.

Process trees answer: *What's done? What's left? What's extraneous?*

### 2.5 Flows & Checkpoints (Contextual Loops)

```flow
flow BuildModel(using: DevSearchers, DevJudges) {
  merge_policy: deep_merge;

  checkpoint "Understand" {
    report = DevMonologue.ask("What is required?");
    S1 = DevSearchers.search("probabilistic models");
    context.update(report, S1);
  }

  checkpoint "Design" {
    // Previous checkpoint's context is available here
    ...
  }
}
```

- **checkpoint**: A stage with local scope; produces a cumulative report.
- **context.update(...)**: Appends to the flow's memory.
- **merge_policy**: `last_wins` | `deep_merge` | `crdt`.

---

## 3. Control Flow

| Keyword | Description |
|---------|-------------|
| `if` / `else` | Conditional branching. |
| `while` | Conditional loop (max 1000 iterations). |
| `for item in list` | Deterministic iteration. |
| `par { ... }` | Parallel execution; contexts merge. |
| `race { ... }` | Competitive execution; first wins. |
| `flow.back_to("checkpoint")` | Jump to a previous checkpoint. |
| `flow.end` | Terminate the flow. |

---

## 4. Execution Model

1. **Parse** → Lark grammar produces AST.
2. **Semantic Analysis** → Type/reference validation.
3. **Build Structures** → Instantiate teams, chains, processes.
4. **Execute Flow** → Iterate checkpoints; dispatch commands to AI provider.
5. **Merge Context** → Apply `merge_policy` after `par`/`race`.
6. **Metrics** → Record action counts, checkpoint durations.

---

## 5. AI Provider Contract

All verbs are routed to an AI backend. Expected JSON schemas:

| Verb | Request Fields | Response Schema |
|------|----------------|-----------------|
| `ask` | `prompt`, `history` | `{ text: string, history: array }` |
| `search` | `query` | `{ hits: array }` |
| `try` | `task`, `options` | `{ output: string, metrics: {} }` |
| `judge` | `target`, `criteria` | `{ score: number, confidence: number, pass: boolean }` |

---

## 6. Governance

- **Roles**: Define capabilities (e.g., `deploy`).
- **Policies**: Rules enforced at runtime (e.g., `no_plain_secrets`).
- **Chain Constraints**: E.g., `require_eval: true` blocks deploy if Evaluation effect < 0.7.

---

## 7. Extensibility

- Add command kinds in `types.py`.
- Extend grammar in `grammar.lark`.
- Implement new merge policies in `runtime.py`.
- Add AI providers in `ai_providers.py`.
