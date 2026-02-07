import subprocess
import json
import os

class FlowRunner:
    """Execute FlowLang flows from Python."""
    def __init__(self, flow_file: str):
        self.flow_file = flow_file
        self.runner_path = os.environ.get("FLOWLANG_RUNNER", "scripts/run.py")

    def run(self, flow_name: str, params: dict = None):
        params = params or {}
        env = os.environ.copy()
        env["FLOW_PARAMS"] = json.dumps(params)
        
        result = subprocess.run(
            ["python", self.runner_path, self.flow_file, flow_name],
            capture_output=True,
            text=True,
            env=env
        )
        
        if result.returncode != 0:
            raise Exception(f"FlowLang execution failed: {result.stderr}")
            
        return {
            "raw_output": result.stdout,
            "status": "completed"
        }

class FlowWorker:
    """Implement Professional Verbs in Python (External SDK mode)."""
    def __init__(self, team: str):
        self.team = team
        self.verbs = {}

    def on(self, verb: str, callback):
        self.verbs[verb] = callback

    def start(self):
        args_json = os.environ.get("FLOW_ARGS", "[]")
        kwargs_json = os.environ.get("FLOW_KWARGS", "{}")
        context_json = os.environ.get("FLOW_CONTEXT", "{}")
        verb = os.environ.get("FLOW_VERB", "Try")

        try:
            args = json.loads(args_json)
            kwargs = json.loads(kwargs_json)
            context = json.loads(context_json)

            if verb in self.verbs:
                result = self.verbs[verb](args, kwargs, context)
                print(json.dumps(result))
            else:
                raise Exception(f"Verb '{verb}' not implemented")
        except Exception as e:
            import sys
            print(f"[REJECTED] {str(e)}", file=sys.stderr)
            sys.exit(1)
