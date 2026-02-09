const { FlowWorker, FlowResult } = require('../../sdk/js');

const worker = new FlowWorker({ team: "InventorySys" });

// Mock Database
const INVENTORY = {
    "SKU-101": 5, // In stock
    "SKU-999": 0  // Out of stock
};

worker.on("Try", async (args, kwargs, ctx) => {
    const action = args[0];
    const orderId = kwargs.order_id;

    // Simulate database latency
    await new Promise(r => setTimeout(r, 10));

    if (action === "reserve") {
        // In a real app, use sku from context or kwargs
        // For demo, we assume all orders are for SKU-101 unless specified
        const sku = "SKU-101";

        if (INVENTORY[sku] > 0) {
            INVENTORY[sku]--;
            return FlowResult.try(true, { "reserved": 1, "remaining": INVENTORY[sku] });
        } else {
            return FlowResult.try(false, {}, "Out of Stock");
        }
    }

    return FlowResult.try(false, {}, "Unknown Action");
});

worker.start();
