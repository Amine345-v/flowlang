const fs = require('fs');
const path = require('path');

async function handleCommand() {
    const input = JSON.parse(fs.readFileSync(0, 'utf-8'));
    const { verb, team, options } = input;

    if (verb === 'try' && team === 'MechanicalEngineer') {
        const material = options?.material || "Steel";
        const load = options?.load || 100;
        const logPath = path.resolve(__dirname, 'physics_sim.log');

        const yieldStrengths = {
            "Titanium": 900,
            "Steel": 250,
            "Aluminum": 70
        };

        const strength = yieldStrengths[material] || 100;
        const safetyFactor = strength / load;
        const pass = safetyFactor > 1.5;

        const logEntry = `[${new Date().toISOString()}] SIM: mat=${material} load=${load}kg result=${pass ? 'PASS' : 'FAIL'} factor=${safetyFactor.toFixed(2)}\n`;
        fs.appendFileSync(logPath, logEntry);

        const report = {
            pass: pass,
            safety_factor: safetyFactor.toFixed(2),
            critical_features: [
                {
                    name: "mechanical_integrity",
                    value: pass ? "SAFE" : "UNSAFE",
                    confidence: 0.95,
                    impact: "constraint"
                },
                {
                    name: "safety_factor",
                    value: safetyFactor.toFixed(2),
                    confidence: 1.0,
                    impact: "guidance"
                }
            ],
            resources: {
                mechanical: {
                    filename: "physics_sim.log",
                    origin_path: logPath,
                    last_sync: new Date().toISOString(),
                    integrity: pass ? "STRICT_PHYSICS_PASS" : "FAILURE_PREDICTED",
                    load_stats: {
                        material: material,
                        load: `${load}kg`,
                        safety_factor: safetyFactor.toFixed(2)
                    },
                    raw_content: fs.readFileSync(logPath, 'utf-8')
                }
            }
        };

        console.log(JSON.stringify(report));
    }
}

handleCommand().catch(e => {
    process.stderr.write(e.toString() + "\n");
    process.exit(1);
});
