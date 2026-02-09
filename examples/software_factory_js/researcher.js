const { FlowWorker, FlowResult } = require('../../sdk/js');

const worker = new FlowWorker({ team: "Researcher" });

// State to simulate market change over time
let marketVersion = 1;

worker.on("Search", async (args, kwargs, ctx) => {
    const query = args[0];

    if (query === "check_drift") {
        console.log(`[Researcher] Checking for Market Drift (Current Version: ${marketVersion})`);

        // SIMULATION: If version is 1, we force a drift to version 2
        // If version is 2, we are stable.
        if (marketVersion === 1) {
            marketVersion = 2;
            console.log("[Researcher] DRIFT DETECTED! New requirements found.");
            return FlowResult.search([], 0).withMeta({ "drift_detected": true });
        } else {
            console.log("[Researcher] Market is stable.");
            return FlowResult.search([], 0).withMeta({ "drift_detected": false });
        }
    }

    // Default: Return concrete spec
    // If version 2, we change the spec!
    console.log("[Researcher] Analyzing Requirements...");

    if (marketVersion === 1) {
        return FlowResult.search([
            "Requirement: Core ERP v1",
            "Must support Basic HR"
        ], 1);
    } else {
        return FlowResult.search([
            "Requirement: Core ERP v2 (Pivot!)",
            "Must support AI-Driven HR",
            "Must support Blockchain Inventory"
        ], 2);
    }
});

// Helper to attach meta data (since FlowResult.search returns a plain object)
// In a real SDK this would be chainable. Here we hack it.
FlowResult.search = function (items, total) {
    const res = { tag: "REPORT", content: { items, total_count: total } };
    res.withMeta = function (meta) {
        Object.assign(this.content, meta);
        return this;
    }
    return res;
};

worker.start();
