const { FlowWorker, FlowResult } = require('../../sdk/js');

const worker = new FlowWorker({ team: "SeniorArchitect" });

worker.on("Try", async (args, kwargs, ctx) => {
    const action = args[0];

    if (action === "design") {
        console.log("[Architect] Analyzing Requirements & Creating Technical Spec...");
        return FlowResult.try(true, {
            "title": "ERP Core System Spec",
            "modules": {
                "HR": ["hire", "fire", "list"],
                "Inventory": ["add", "remove", "check"]
            },
            "architecture_style": "Modular Monolith",
            "database_schema": {
                "employees": ["id", "name", "role"],
                "stock": ["item", "qty"]
            }
        });
    }

    return FlowResult.try(false, {}, "Architect only does Design now.");
});

worker.start();
