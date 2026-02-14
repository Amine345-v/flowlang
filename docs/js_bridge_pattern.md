# Example: The Bridge Pattern (JS <-> FlowLang)

This pattern demonstrates how FlowLang acts as a **Governance Layer** for a web application.

## 1. The Governance Layer (`audit.flow`)
FlowLang ensures that no AI advice reaches the user without a "Strict Judgment" checkpoint.

```flowlang
team legal_bot : Command<Judge> [size=1];
team research_bot : Command<Ask> [size=1];

flow verify_user_content(content: string) {
    checkpoint "analysis" (report: raw_analysis) {
        raw_analysis = research_bot.ask("Analyze for risks: " + content);
    }

    checkpoint "validation" (report: final_approved_result) {
        # Deterministic Gate: JS cannot bypass this.
        final_approved_result = legal_bot.judge(raw_analysis, policy="StrictCompliance");
    }
}
```

## 2. The Execution Layer (`app.js`)
JavaScript handles the UI and calls the "Orchestrator" (FlowLang) to get a certified result.

```javascript
import { FlowRunner } from 'flowlang-sdk';

async function handleSubmit(userInput) {
    console.log("Starting Professional Audit...");

    // 1. Call the Governance Layer
    // FlowLang handles the AI workforce, pruning, and state.
    const runner = new FlowRunner("audit.flow");
    const result = await runner.run("verify_user_content", { 
        content: userInput 
    });

    // 2. JS only acts on the 'Certified' output
    if (result.final_approved_result.status === "approved") {
        updateUI(result.final_approved_result.data);
    } else {
        showError("Content failed professional audit.");
    }
}
```

## 3. Why this works
- **Separation of Concerns**: JS doesn't "talk" to the LLM; it talks to the **Conductor**.
- **Auditability**: Every call generates a `.audit` file that developers can inspect.
- **Safety**: If the `analysis` stage hallucinations, the `validation` stage in the same flow acts as a deterministic barrier.
