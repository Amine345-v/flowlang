const fs = require('fs');
const path = require('path');

async function handleCommand() {
    const input = JSON.parse(fs.readFileSync(0, 'utf-8'));
    const { verb, team, target } = input;

    if (verb === 'judge' && team === 'CyberExpert') {
        const distDir = path.resolve(__dirname, 'dist');
        const logs = [`[SCAN] Inspecting origin: ${distDir}`];
        let vulnerabilities = [];

        if (fs.existsSync(distDir)) {
            const files = fs.readdirSync(distDir).filter(f => f.endsWith('.js'));
            files.forEach(file => {
                const filePath = path.join(distDir, file);
                const content = fs.readFileSync(filePath, 'utf-8');
                logs.push(`[SCAN] checking ${file} at ${filePath}...`);

                if (content.includes('eval(')) vulnerabilities.push(`Insecure use of eval() in ${file}`);
                if (content.includes('innerHTML')) vulnerabilities.push(`Potential XSS via innerHTML in ${file}`);
            });
        }

        const pass = vulnerabilities.length === 0;
        const report = {
            pass: pass,
            score: pass ? 1.0 : 0.4,
            resources: {
                cyber: {
                    target: "BUILD_ARTIFACTS",
                    origin_path: distDir,
                    last_sync: new Date().toISOString(),
                    level: pass ? "SECURE" : "VULNERABLE",
                    logs: logs,
                    vulnerabilities: vulnerabilities.join("; ") || "Transparent Security Integrity",
                    raw_content: logs.join('\n')
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
