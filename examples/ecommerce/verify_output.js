const { FlowRunner } = require('../../sdk/js');
const fs = require('fs');
const path = require('path');

async function runTest() {
    try {
        const flowPath = path.resolve(__dirname, 'order_lifecycle.flow');
        const runnerPath = path.resolve(__dirname, '../../scripts/run.py');

        process.env.FLOWLANG_RUNNER = runnerPath;

        const runner = new FlowRunner(flowPath);

        await runner.run("manage_order_lifecycle", {
            order_id: "ORD-001", amount: 150.00, user_risk_score: 10
        });

    } catch (e) {
        let msg = "ERROR:\n" + e.message + "\n";
        if (e.stderr) msg += "STDERR:\n" + e.stderr + "\n";
        msg += "STACK:\n" + e.stack;
        fs.writeFileSync('error.log', msg);
    }
}

runTest();
