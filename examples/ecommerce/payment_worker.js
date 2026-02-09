const { FlowWorker, FlowResult } = require('../../sdk/js');

const worker = new FlowWorker({ team: "BankGateway" });

worker.on("Try", async (args, kwargs, ctx) => {
    const action = args[0]; // e.g. "charge"
    const amount = kwargs.amount;
    const orderId = kwargs.order_id;

    // Simulate payment gateway
    // 95% success rate
    const success = Math.random() > 0.05;

    if (success) {
        return FlowResult.try(true, { "txn_id": "TXN-" + Math.floor(Math.random() * 10000) });
    } else {
        return FlowResult.try(false, {}, "Insufficient Funds");
    }
});

worker.start();
