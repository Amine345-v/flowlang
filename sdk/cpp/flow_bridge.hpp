#ifndef FLOWLANG_BRIDGE_HPP
#define FLOWLANG_BRIDGE_HPP

#include <iostream>
#include <string>
#include <map>
#include <functional>
#include <vector>

namespace flow {

struct Context {
    std::string maestro_path;
    std::string flow_id;
};

// Simple Mock JSON implementation for the bridge demo
// In a real scenario, we would use nlohmann/json
struct TypedValue {
    std::string tag;
    std::string content_json;
    std::string meta_json;

    std::string to_json() const {
        return "{\"tag\":\"" + tag + "\",\"content\":" + content_json + ",\"meta\":" + meta_json + "}";
    }
};

class Bridge {
public:
    Bridge(const std::string& team) : team_name(team) {}

    typedef std::function<TypedValue(const std::string& args_json, const std::string& kwargs_json, const Context& ctx)> VerbCallback;

    void register_verb(const std::string& verb, VerbCallback cb) {
        verbs[verb] = cb;
    }

    void run() {
        // Read SPP from Environment (passed by Conductor)
        char* verb_ptr = std::getenv("FLOW_VERB");
        char* args_ptr = std::getenv("FLOW_ARGS");
        char* kwargs_ptr = std::getenv("FLOW_KWARGS");
        char* ctx_ptr = std::getenv("FLOW_CONTEXT");

        if (!verb_ptr || !args_ptr) {
            std::cerr << "[REJECTED] Missing SPP environment variables" << std::endl;
            std::exit(1);
        }

        std::string verb = verb_ptr;
        std::string args = args_ptr;
        std::string kwargs = kwargs_ptr ? kwargs_ptr : "{}";
        
        Context ctx;
        if (ctx_ptr) {
            // Rudimentary parsing for the prototype
            ctx.maestro_path = ""; // In real C++, use a JSON parser
        }

        if (verbs.count(verb)) {
            TypedValue result = verbs[verb](args, kwargs, ctx);
            std::cout << result.to_json() << std::endl;
        } else {
            std::cerr << "[REJECTED] Verb '" << verb << "' not implemented" << std::endl;
            std::exit(1);
        }
    }

private:
    std::string team_name;
    std::map<std::string, VerbCallback> verbs;
};

inline TypedValue Report(const std::string& content_json) {
    return {"REPORT", content_json, "{\"engine\":\"cpp_native\"}"};
}

} // namespace flow

#endif
