const { FlowWorker, FlowResult } = require('../../sdk/js');
const fs = require('fs');
const path = require('path');

const worker = new FlowWorker({ team: "DevTeam" });

const OUTPUT_DIR = path.resolve(__dirname, 'dist');
if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR);

worker.on("Try", async (args, kwargs, ctx) => {
    const action = args[0];
    const spec = kwargs.spec;

    if (action === "implement") {
        console.log(`[Developer] Receiving Spec: ${spec.output.title}`);
        console.log("[Developer] Writing code in 'dist/'...");

        // Generate Code based on Spec
        // (Simplified logic: we just spit out the same ERP core, but purely triggered by the flow)

        const implCode = `
class ERP {
    constructor() {
        this.hr = { hire: () => 1, list: () => [{id:1}] }; // simplified based on spec
        this.inventory = { add: () => {}, check: () => 10 };
    }
}
module.exports = ERP;
`;
        fs.writeFileSync(path.join(OUTPUT_DIR, 'erp_core.js'), implCode);

        // Generate Test
        const testCode = `
const ERP = require('./erp_core');
const erp = new ERP();
if (!erp.hr.hire()) throw new Error("Hire failed");
console.log("Dev Tests Passed");
`;
        fs.writeFileSync(path.join(OUTPUT_DIR, 'erp_core.test.js'), testCode);

        return FlowResult.try(true, {
            "status": "Implementation Complete",
            "files": ["erp_core.js", "erp_core.test.js"]
        });
    }

    return FlowResult.try(false, {}, "Unknown Action");
});

worker.start();
