const { FlowRunner } = require('flowlang-js');
const path = require('path');
const fs = require('fs');

async function startProfessionalSimulation() {
    console.log(`[STUDIO] Launching Universal Professional Simulation...`);

    process.env.FLOWLANG_RUNNER = path.resolve(__dirname, '../../scripts/run.py');

    const idePublicDir = path.resolve(__dirname, "../../jol-ide---لغة-برمجة-المهن/public");
    const ideStatePath = path.join(idePublicDir, "ide_state.json");

    if (!fs.existsSync(idePublicDir)) {
        fs.mkdirSync(idePublicDir, { recursive: true });
    }

    process.env.FLOWLANG_IDE_EXPORT_PATH = ideStatePath;
    console.log(`[IDE] Live state export enabled: ${ideStatePath}`);

    const factory = new FlowRunner(path.resolve(__dirname, "cross_field.flow"));

    try {
        const result = await factory.run("professional_convergence", {});

        if (result.status === "completed") {
            console.log("=== PROFESSIONAL REPORT ===");
            console.log(result.raw_output);
            console.log("=== SIMULATION DONE ===");
        } else {
            console.error("Simulation Halted:", result.raw_output);
        }

    } catch (e) {
        console.error("Governance Failure:", e.message);
    }
}

startProfessionalSimulation();
