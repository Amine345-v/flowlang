# Environment Variables Setup Guide

To run the FlowLang Software Factory with full capabilities (AI, IDE Integration, etc.), you need to set the following environment variables.

## 1. Create a `.env` file in `examples/software_factory_js/`

It is recommended to use `dotenv` or simply export these in your shell.

```bash
# === Core AI Configuration ===
# Required for actual AI execution. If not set, the system uses mock/fake responses.
export OPENAI_API_KEY="sk-..."
export FLOWLANG_AI_MODEL="gpt-4o"

# === FlowLang Runtime Configuration ===
# Path to the Python runtime bridge (Adjust path to your local repo)
# Windows Example:
export FLOWLANG_RUNNER="c:/Users/asusu/CascadeProjects/flowlang/flowlang/scripts/run.py"
# Linux/Mac Example:
# export FLOWLANG_RUNNER="/path/to/flowlang/scripts/run.py"

# === IDE Integration (Optional) ===
# Path where the runtime exports the live state JSON for the JOL-IDE to read.
# Point this to the 'public' folder of your JOL-IDE project.
export FLOWLANG_IDE_EXPORT_PATH="c:/Users/asusu/CascadeProjects/flowlang/jol-ide---لغة-برمجة-المهن/public/ide_state.json"

# === Automation (Optional) ===
# Set to 'true' to auto-approve "confirm" gates (useful for CI/CD or headless runs)
export FLOWLANG_AUTO_APPROVE="true"
```

## 2. Using with `run_factory.js`

The `run_factory.js` script already sets some defaults programmatically if not present, but for full control, set them in your environment before running:

```powershell
# Windows PowerShell Example
$env:OPENAI_API_KEY="your-key-here"
$env:FLOWLANG_IDE_EXPORT_PATH="c:/Users/asusu/CascadeProjects/flowlang/jol-ide---لغة-برمجة-المهن/public/ide_state.json"
npm start
```
