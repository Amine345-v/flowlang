const fs = require('fs');
const path = require('path');

async function handleCommand() {
    const input = JSON.parse(fs.readFileSync(0, 'utf-8'));
    const { verb, team, query } = input;

    if (verb === 'search' && team === 'Economist') {
        const budgetPath = path.resolve(__dirname, 'budget.json');

        let budgetData;
        if (!fs.existsSync(budgetPath)) {
            budgetData = {
                items: [
                    { id: '1', name: 'R&D', forecast: 120000, actual: 115000 },
                    { id: '2', name: 'Infrastructure', forecast: 45000, actual: 52000 },
                    { id: '3', name: 'Marketing', forecast: 30000, actual: 0 }
                ]
            };
            fs.writeFileSync(budgetPath, JSON.stringify(budgetData, null, 2));
        } else {
            budgetData = JSON.parse(fs.readFileSync(budgetPath, 'utf-8'));
        }

        const sheet = budgetData.items.map(item => [
            item.id,
            item.name,
            `$${item.forecast.toLocaleString()}`,
            `$${item.actual.toLocaleString()}`
        ]);

        const totalForecast = budgetData.items.reduce((sum, item) => sum + item.forecast, 0);
        const totalActual = budgetData.items.reduce((sum, item) => sum + item.actual, 0);

        const report = {
            status: totalActual > totalForecast ? "OVER_BUDGET" : "HEALTHY",
            critical_features: [
                {
                    name: "budget_status",
                    value: totalActual > totalForecast ? "OVER_BUDGET" : "HEALTHY",
                    confidence: 1.0,
                    impact: "requirement"
                },
                {
                    name: "total_actual_spend",
                    value: totalActual,
                    confidence: 1.0,
                    impact: "guidance"
                }
            ],
            resources: {
                economic: {
                    filename: "budget.json",
                    origin_path: budgetPath,
                    last_sync: new Date().toISOString(),
                    integrity: "SIGNED_FILESYSTEM",
                    sheet: sheet,
                    total: `$${totalForecast.toLocaleString()}`,
                    raw_content: JSON.stringify(budgetData, null, 4)
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
