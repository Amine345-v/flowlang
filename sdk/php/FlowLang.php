<?php

namespace FlowLang;

class FlowRunner {
    protected $flowFile;
    protected $runnerPath;

    public function __construct($flowFile) {
        $this->flowFile = $flowFile;
        $this->runnerPath = getenv('FLOWLANG_RUNNER') ?: 'scripts/run.py';
    }

    public function run($flowName, $params = []) {
        $paramsJson = json_encode($params);
        $cmd = "python " . escapeshellarg($this->runnerPath) . " " . escapeshellarg($this->flowFile) . " " . escapeshellarg($flowName);
        
        $descriptorSpec = [
            0 => ["pipe", "r"],
            1 => ["pipe", "w"],
            2 => ["pipe", "w"]
        ];

        $process = proc_open($cmd, $descriptorSpec, $pipes, null, [
            'FLOW_PARAMS' => $paramsJson
        ]);

        if (is_resource($process)) {
            $stdout = stream_get_contents($pipes[1]);
            $stderr = stream_get_contents($pipes[2]);
            fclose($pipes[0]);
            fclose($pipes[1]);
            fclose($pipes[2]);
            $returnCode = proc_close($process);

            if ($returnCode !== 0) {
                throw new \Exception("FlowLang execution failed: " . $stderr);
            }

            return [
                "raw_output" => $stdout,
                "status" => "completed"
            ];
        }
        throw new \Exception("Failed to start FlowLang process");
    }
}

class FlowWorker {
    protected $team;
    protected $verbs = [];

    public function __construct($team) {
        $this->team = $team;
    }

    public function on($verb, $callback) {
        $this->verbs[$verb] = $callback;
    }

    public function start() {
        $args = json_decode(getenv('FLOW_ARGS') ?: '[]', true);
        $kwargs = json_decode(getenv('FLOW_KWARGS') ?: '{}', true);
        $context = json_decode(getenv('FLOW_CONTEXT') ?: '{}', true);
        $verb = getenv('FLOW_VERB') ?: 'Try';

        try {
            if (isset($this->verbs[$verb])) {
                $result = call_user_func($this->verbs[$verb], $args, $kwargs, $context);
                echo json_encode($result);
            } else {
                throw new \Exception("Verb '{$verb}' not implemented");
            }
        } catch (\Exception $e) {
            fwrite(STDERR, "[REJECTED] " . $e->getMessage() . "\n");
            exit(1);
        }
    }
}
