package flowlang

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
)

type FlowRunner struct {
	FlowFile   string
	RunnerPath string
}

func NewFlowRunner(flowFile string) *FlowRunner {
	runnerPath := os.Getenv("FLOWLANG_RUNNER")
	if runnerPath == "" {
		runnerPath = "scripts/run.py"
	}
	return &FlowRunner{FlowFile: flowFile, RunnerPath: runnerPath}
}

func (r *FlowRunner) Run(flowName string, params map[string]interface{}) (map[string]interface{}, error) {
	paramsJson, _ := json.Marshal(params)
	cmd := exec.Command("python", r.RunnerPath, r.FlowFile, flowName)
	cmd.Env = append(os.Environ(), "FLOW_PARAMS="+string(paramsJson))

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()
	if err != nil {
		return nil, fmt.Errorf("FlowLang execution failed: %s", stderr.String())
	}

	return map[string]interface{}{
		"raw_output": stdout.String(),
		"status":     "completed",
	}, nil
}

type FlowWorker struct {
	Team  string
	Verbs map[string]func(args []interface{}, kwargs map[string]interface{}, context map[string]interface{}) interface{}
}

func NewFlowWorker(team string) *FlowWorker {
	return &FlowWorker{
		Team:  team,
		Verbs: make(map[string]func(args []interface{}, kwargs map[string]interface{}, context map[string]interface{}) interface{}),
	}
}

func (w *FlowWorker) On(verb string, callback func(args []interface{}, kwargs map[string]interface{}, context map[string]interface{}) interface{}) {
	w.Verbs[verb] = callback
}

func (w *FlowWorker) Start() {
	argsJson := os.Getenv("FLOW_ARGS")
	kwargsJson := os.Getenv("FLOW_KWARGS")
	ctxJson := os.Getenv("FLOW_CONTEXT")
	verb := os.Getenv("FLOW_VERB")

	var args []interface{}
	var kwargs map[string]interface{}
	var context map[string]interface{}

	json.Unmarshal([]byte(argsJson), &args)
	json.Unmarshal([]byte(kwargsJson), &kwargs)
	json.Unmarshal([]byte(ctxJson), &context)

	if verb == "" {
		verb = "Try"
	}

	if callback, ok := w.Verbs[verb]; ok {
		result := callback(args, kwargs, context)
		output, _ := json.Marshal(result)
		fmt.Println(string(output))
	} else {
		fmt.Fprintf(os.Stderr, "[REJECTED] Verb '%s' not implemented\n", verb)
		os.Exit(1)
	}
}
