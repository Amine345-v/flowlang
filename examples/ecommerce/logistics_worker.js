const { FlowWorker, FlowResult } = require('../../sdk/js');

const worker = new FlowWorker({ team: "LogisticsProvider" });

worker.on("Search", async (args, kwargs, ctx) => {
    // "Search" implies finding the best carrier
    const destination = kwargs.destination;
    const weight = kwargs.weight;

    // Logic: Heavy -> UPS, Light -> FedEx

    const carriers = [
        { name: "FedEx Ground", cost: 12.50, eta: "3 days" },
        { name: "UPS Standard", cost: 14.00, eta: "2 days" }
    ];

    return FlowResult.search(carriers, carriers.length);
});

worker.start();
