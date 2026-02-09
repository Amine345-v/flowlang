# FlowLang JS SDK Prototype (`flowlang-js`)

This library allows JavaScript/TypeScript applications to act as workers or consumers within a FlowLang orchestrated environment.

## 1. Minimal API Design

```javascript
const { FlowWorker } = require('flowlang-js');

// 1. Create a Worker for a specific Professional Team
const worker = new FlowWorker({
    team: "legal_audit",
    capabilities: ["Judge"]
});

// 2. Implement the Professional Verb
worker.on("Judge", (args, context) => {
    const { content } = args;
    console.log(`[JS Worker] Governed by Flow: ${context.flow_id} at Path: ${context.maestro_path}`);
    
    // Deterministic logic
    const isViolated = content.includes("confidential");
    
    return {
        pass: !isViolated,
        reason: isViolated ? "Confidential data detected" : "Approved"
    };
});

// 3. Listen for Conductor commands (via STDIO or HTTP)
worker.start();
```

## 2. Integration with Frameworks (Node.js/Express)
You can wrap the FlowWorker inside an Express route to provide "Governance as a Service".

```javascript
app.post("/audit", async (req, res) => {
    const conductor = new FlowConductor("audit.flow");
    const result = await conductor.run("verify_compliance", { data: req.body });
    res.json(result);
});
```
