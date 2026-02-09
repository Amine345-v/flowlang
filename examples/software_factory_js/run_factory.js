const { FlowRunner } = require('flowlang-js');
const path = require('path');
const fs = require('fs');

async function buildSoftware(projectName) {
    console.log(`[PM] Requesting Software Build: ${projectName} (Team: Research -> Architect -> Dev -> QA)`);

    process.env.FLOWLANG_RUNNER = path.resolve(__dirname, '../../scripts/run.py');
    
    // IDE Integration: Set export path to JOL-IDE public folder
    const idePublicDir = path.resolve(__dirname, "../../jol-ide---لغة-برمجة-المهن/public");
    const ideStatePath = path.join(idePublicDir, "ide_state.json");
    
    process.env.FLOWLANG_IDE_EXPORT_PATH = ideStatePath;
    console.log(`[IDE] Live state export enabled: ${ideStatePath}`);
    
    // Initialize Governance
    const factory = new FlowRunner(path.resolve(__dirname, "factory.flow"));

    try {
        const result = await factory.run("build_feature", {
            feature_name: projectName
        });

        if (result.status === "completed") {
            console.log("=== FACTORY REPORT ===");
            console.log(result.raw_output);
            console.log("=== DONE ===");
        } else {
            console.error("Factory Halted:", result.raw_output);
        }

    } catch (e) {
        console.error("Governance Failure:", e.message);
    }
}

// Run the Simulation
buildSoftware("Enterprise-Resource-Planning-Core-V2");
