const { FlowWorker } = require('./index');

// 1. Initialize Worker
const worker = new FlowWorker({ team: "LegalExperts" });

// 2. Implement professional logic
worker.on("Judge", async (args, kwargs, context) => {
    const content = args[0] || "";
    const criteria = kwargs.criteria || "Standard";

    const hasConfidential = content.includes("CONFIDENTIAL") || content.includes("SECRET");

    return {
        tag: "REPORT",
        content: {
            pass: !hasConfidential,
            reason: hasConfidential ? "Confidential leak detected" : "Audit passed",
            criteria_used: criteria,
            maestro_path: context.maestro_path
        },
        meta: {
            engine: "javascript_v8",
            timestamp: new Date().toISOString()
        }
    };
});

// 3. Command the worker to start
worker.start();
