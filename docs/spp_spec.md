# Specification: Standard Professional Protocol (SPP) v1.0

The Standard Professional Protocol (SPP) is a JSON-based interface that enables external execution environments (JS, C++, Rust, etc.) to integrate with the FlowLang **Governance Layer**.

## 1. Core Principles
- **Unidirectional Governance**: FlowLang (The Conductor) issues commands; External Tiers (The Workers) execute and report.
- **Stateful Handover**: Every request includes the current `Maestro Path` and `Context Report`.
- **Deterministic Commitment**: Workers must return a `TypedValue` that matches the FlowLang schema.

## 2. Protocol Schema (JSON-RPC 2.0 Style)

### A. Dispatch Command (Conductor -> Worker)
When FlowLang dispatches a verb (Search, Try, Judge, Ask) to an external SDK:
```json
{
  "jsonrpc": "2.0",
  "method": "execute_verb",
  "params": {
    "verb": "Judge",
    "args": ["Patient Report Data..."],
    "kwargs": { "criteria": "StrictCompliance" },
    "context": {
      "maestro_path": "0101",
      "stage": "Validation",
      "flow_id": "uuid-123"
    }
  },
  "id": 1
}
```

### B. Execution Result (Worker -> Conductor)
The worker must respond in a format the Conductor can "Prune" and "Audit":
```json
{
  "jsonrpc": "2.0",
  "result": {
    "tag": "REPORT",
    "content": { "pass": true, "reason": "No HIPAA violations found" },
    "meta": {
      "duration_ms": 150,
      "worker_lang": "javascript",
      "worker_version": "1.0.2"
    }
  },
  "id": 1
}
```

## 3. Worker Registration
Workers can register themselves as specialized "Professional Teams":
- **Capabilities**: A list of verbs the worker supports (e.g., `["Search", "Judge"]`).
- **Endpoint**: The transport layer (STDIO, HTTP, WebSockets, or Unix Socket).

## 4. Governance Enforcement
If a worker returns data that violates a FlowLang **Policy**:
1. The Conductor rejects the `Commit`.
2. The Conductor triggers a `flow.back_to` remediation.
3. The Audit Trail log marks the worker as "Inconsistent".

---

**SPP is the bridge that turns raw code into governed professional output.**
