# FlowLang C++ Bridge (`flow_bridge.hpp`)

Designed for high-performance and industrial execution environments where FlowLang acts as the high-level orchestrator.

## 1. Minimal Header Design

```cpp
#include <flowlang/bridge.hpp>
#include <iostream>

int main() {
    // 1. Initialize the Bridge (handles SPP protocol via JSON over Pipe/Socket)
    flow::Bridge bridge("industrial_control");

    // 2. Register Callback for 'Search' (Worker Tier)
    bridge.register_verb("Search", [](const flow::Args& args, const flow::Context& ctx) {
        std::cout << "[C++ Worker] Executing low-level sensor scan..." << std::endl;
        std::cout << "[Maestro] Binary Path: " << ctx.maestro_path << std::endl;
        
        // Return a TypedValue
        return flow::Report({
            {"status", "stable"},
            {"temp", 42.5}
        });
    });

    // 3. Connect to Conductor and Enter Governance Loop
    bridge.run(); 
    return 0;
}
```

## 2. Competitive Advantage
- **Memory Safety**: FlowLang manages the "State" and "Memory pruning", while C++ handles the "Deterministic Calculation".
- **Maestro Awareness**: C++ code can query its position in the `Process Tree` via the `ctx.maestro_path` bit-string, enabling context-aware hardware control.
