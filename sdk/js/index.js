const { spawnSync } = require('child_process');

/**
 * FlowWorker: Implement Professional Verbs in JavaScript
 */
class FlowWorker {
    constructor({ team }) {
        this.team = team;
        this.verbs = {};
    }

    on(verb, callback) {
        this.verbs[verb] = callback;
    }

    async start() {
        const argsJson = process.env.FLOW_ARGS || "[]";
        const kwargsJson = process.env.FLOW_KWARGS || "{}";
        const contextJson = process.env.FLOW_CONTEXT || "{}";

        try {
            const args = JSON.parse(argsJson);
            const kwargs = JSON.parse(kwargsJson);
            const context = JSON.parse(contextJson);
            const verb = process.env.FLOW_VERB || "Try";

            if (this.verbs[verb]) {
                const result = await this.verbs[verb](args, kwargs, context);
                console.log(JSON.stringify(result));
            } else {
                throw new Error(`Verb '${verb}' not implemented by worker for team ${this.team}`);
            }
        } catch (err) {
            process.stderr.write(`[REJECTED] ${err.message}\n`);
            process.exit(1);
        }
    }
}

/**
 * FlowRunner: Execute FlowLang flows from JavaScript
 */
class FlowRunner {
    constructor(flowFile) {
        this.flowFile = flowFile;
        this.runnerPath = process.env.FLOWLANG_RUNNER || 'scripts/run.py';
    }

    async run(flowName, params = {}) {
        const argsStr = JSON.stringify(params);

        const result = spawnSync('python', [this.runnerPath, this.flowFile, flowName], {
            encoding: 'utf8',
            env: { ...process.env, FLOW_PARAMS: argsStr }
        });

        if (result.error) {
            throw new Error(`Failed to execute FlowLang: ${result.error.message}`);
        }

        if (result.status !== 0) {
            throw new Error(`FlowLang execution failed: ${result.stderr}`);
        }

        return {
            raw_output: result.stdout,
            status: "completed"
        };
    }
}

module.exports = { FlowWorker, FlowRunner };
