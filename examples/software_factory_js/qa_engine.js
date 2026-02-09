const { FlowWorker, FlowResult } = require('../../sdk/js');
const { spawnSync } = require('child_process');
const path = require('path');

const worker = new FlowWorker({ team: "QA_Department" });

worker.on("Judge", async (args, kwargs, ctx) => {
    // Check if the implementation stage succeeded
    if (!args[0].success) {
        return FlowResult.judge(false, 1.0, "Architecture Implementation Failed");
    }

    console.log("[QA] Running automated ERP validation suite...");

    // We expect the architect to produce 'erp_core.test.js'
    const testFile = path.resolve(__dirname, 'dist/erp_core.test.js');

    if (!require('fs').existsSync(testFile)) {
        return FlowResult.judge(false, 1.0, "Critical: Test file missing!");
    }

    // Execute the generated test file
    const result = spawnSync('node', [testFile], { encoding: 'utf8' });

    if (result.status === 0) {
        console.log(result.stdout);
        return FlowResult.judge(true, 1.0, "ERP System Validated & Ready for Deployment");
    } else {
        console.error(result.stderr || result.stdout);
        return FlowResult.judge(false, 1.0, "ERP Validation Failed");
    }
});

worker.start();
