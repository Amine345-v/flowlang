try {
    console.log("Attempting to require ../../sdk/js");
    const SDK = require('../../sdk/js');
    console.log("Require successful. Exports:", Object.keys(SDK));

    if (!SDK.FlowRunner) {
        console.error("FlowRunner is missing from exports!");
    } else {
        console.log("FlowRunner found.");
        const runner = new SDK.FlowRunner("test.flow");
        console.log("FlowRunner instantiated.");
    }
} catch (e) {
    console.error("Require failed:", e);
}
