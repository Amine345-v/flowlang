#include "flow_bridge.hpp"

int main() {
    flow::Bridge bridge("SecurityScanner");

    // Register high-performance scanning verb
    bridge.register_verb("Search", [](const std::string& args, const std::string& kwargs, const flow::Context& ctx) {
        // Simulate high-speed binary analysis
        return flow::Report("{\"vulnerabilities\": 0, \"status\": \"secure\", \"engine\": \"LLVM-Optimized\"}");
    });

    bridge.run();
    return 0;
}
