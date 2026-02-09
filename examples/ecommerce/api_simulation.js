const { FlowRunner } = require('flowlang-js');
const path = require('path');

// Configure the FlowLang Bridge
process.env.FLOWLANG_RUNNER = path.resolve(__dirname, '../../scripts/run.py');

// Simulate a Next.js API Handler
async function POST(req) {
    console.log(`[API] Received Order Request: ${JSON.stringify(req.body)}`);

    // 1. Initialize FlowRunner with the governance definition
    // Connects the "Policy" (Flow) to the "Mechanism" (Python Runner)
    const runner = new FlowRunner(path.resolve(__dirname, "order_lifecycle.flow"));

    // 2. Trigger the "Process Order" flow
    // The frontend blindly trusts the result because FlowLang guarantees the steps were followed.
    try {
        const result = await runner.run("manage_order_lifecycle", {
            order_id: req.body.orderId,
            amount: req.body.amount,
            user_risk_score: req.body.riskScore
        });

        if (result.status === "completed") {
            // Parse the stdout to find the final report (in a real app, use structured JSON output)
            console.log("[API] Governance Complete. Result:");
            console.log(result.raw_output);
            return { status: 200, message: "Order Processed" };
        } else {
            console.error(result.raw_output);
            return { status: 500, message: "Governance Failed" };
        }
    } catch (e) {
        console.error(e);
        return { status: 500, message: e.message };
    }
}

// --- Run Simulation ---
(async () => {
    // Scenario 1: Clean Order
    console.log("\n--- Scenario 1: Standard Order ---");
    await POST({ body: { orderId: "ORD-001", amount: 150.00, riskScore: 10 } });

    // Scenario 2: High Risk Fraud
    console.log("\n--- Scenario 2: Fraudulent Order ---");
    await POST({ body: { orderId: "ORD-999", amount: 5000.00, riskScore: 95 } });
})();
