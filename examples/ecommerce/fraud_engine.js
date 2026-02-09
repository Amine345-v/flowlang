const { FlowWorker, FlowResult } = require('../../sdk/js');

const worker = new FlowWorker({ team: "FraudGuard" });

worker.on("Judge", async (args, kwargs, ctx) => {
    // Simulate complex risk logic
    const amount = kwargs.amount || 0;
    const riskScore = kwargs.risk_score || 0;

    // Governance Rule: High Value + Med Risk = Manual Review (Fail for now)
    // In a real app, this might trigger a "Human Gate" in FlowLang

    if (amount > 1000 && riskScore > 50) {
        return FlowResult.judge(false, 0.9, "High Value Risk: Requires Manual Approval");
    }

    if (riskScore > 80) {
        return FlowResult.judge(false, 0.99, "Automatic Fraud Rejection");
    }

    // Log the approval path for audit
    // ctx.set("audit_log", `Approved by Node.js Worker at ${Date.now()}`);

    return FlowResult.judge(true, 1.0, "Transaction Clean");
});

worker.start();
