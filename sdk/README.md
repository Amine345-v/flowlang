# FlowLang Universal SDKs: The Professional Bridge

FlowLang is designed to be the **Architectural Governance Layer** for any software system. Whether you are building high-frequency trading algorithms, industrial control systems, or modern web applications, FlowLang acts as the "Conductor" that orchestrates your existing code bases (the "Musicians").

This repository contains the official SDKs (Bridges) that allow your code to speak the **Standard Professional Protocol (SPP)**.

## Installation

| Language | Path | Status |
|----------|------|--------|
| **JavaScript** | `sdk/js` | ✅ Alpha |
| **Python** | `sdk/python` | ✅ Alpha |
| **C++** | `sdk/cpp` | ✅ Alpha (Header-only) |
| **Go** | `sdk/go` | ✅ Alpha |
| **PHP** | `sdk/php` | ✅ Alpha |

---

## 1. Web & Legal Compliance (JavaScript / Node.js)
**Scenario**: A law firm wants an AI audit tool, but needs a "Human-in-the-Loop" guarantee and strict audit logs for liability reasons. The frontend is Next.js.

### The Flow (`audit.flow`)
```flowlang
team legal_bot : Command<Judge> [size=1, connector="node worker.js"];

flow document_review(doc: string) {
    checkpoint "compliance_check" (report: result) {
        result = legal_bot.judge(doc);
    }
}
```

### The Worker (`worker.js`)
```javascript
const { FlowWorker } = require('./sdk/js');

const worker = new FlowWorker({ team: "LegalExperts" });

worker.on("Judge", async (args) => {
    const doc = args[0];
    const isConfidential = doc.includes("CONFIDENTIAL");
    
    // JS Logic: Executing the "Work"
    return {
        tag: "REPORT",
        content: {
            pass: !isConfidential,
            reason: isConfidential ? "Data Leak Detected" : "Clean"
        }
    };
});

worker.start();
```

---

## 2. Quantitative Finance (Python)
**Scenario**: A hedge fund uses Python for heavy data analysis. FlowLang doesn't replace pandas/numpy; it *governs* the trading risks.

### The Flow (`trading.flow`)
```flowlang
team risk_engine : Command<Judge> [size=1, connector="python risk_engine.py"];

flow execute_trade(symbol: string, amount: number) {
    checkpoint "risk_gate" (report: approval) {
        approval = risk_engine.judge({"symbol": symbol, "volatility_cap": 0.05});
        if (approval.pass == false) {
            flow.end; // HARD BREAK: Trade cannot proceed.
        }
    }
}
```

### The Worker (`risk_engine.py`)
```python
from flowlang_sdk import FlowWorker
import pandas as pd # Specialized library

worker = FlowWorker("RiskManagement")

def calculate_risk(args, kwargs, context):
    symbol = kwargs.get("symbol")
    # Python is great at this data lifting
    df = pd.read_csv("market_data.csv")
    volatility = df[df['symbol'] == symbol]['volatility'].iloc[0]
    
    is_safe = volatility < kwargs['volatility_cap']
    
    return {
        "tag": "REPORT",
        "content": {"pass": bool(is_safe), "metric": float(volatility)}
    }

worker.on("Judge", calculate_risk)
worker.start()
```

---

## 3. Industrial Automation & IoT (C++)
**Scenario**: A manufacturing plant runs on embedded C++. Latency is critical. FlowLang sits on the server as the "Plant Manager", sending high-level directives to C++ controllers.

### The Flow (`factory.flow`)
```flowlang
team robotic_arm : Command<Try> [size=1, connector="./arm_controller"];

flow assembly_line() {
    checkpoint "weld_chassis" (report: status) {
        status = robotic_arm.try("weld", {"pressure": 1500});
    }
}
```

### The Worker (`arm_controller.cpp`)
```cpp
#include "flow_bridge.hpp"

int main() {
    flow::Bridge bridge("RoboticsDivision");

    bridge.register_verb("Try", [](auto args, auto kwargs, auto ctx) {
        // Direct hardware manipulation (simplified)
        // Hardware::SetPressure(1500);
        // Hardware::ActivateWelder();
        
        return flow::Report("{\"status\": \"welded\", \"integrity\": 99.8}");
    });

    bridge.run();
}
```

---

## 4. High-Throughput Logistics (Go)
**Scenario**: A shipping company routes thousands of packages per second. Go is used for its concurrency. FlowLang defines the "Routing Rules" (Business Logic).

### The Worker (`router.go`)
```go
package main
import "./sdk/go/flowlang"

func main() {
    worker := flowlang.NewFlowWorker("LogisticsFleet")

    worker.On("Search", func(args []interface{}, kwargs, ctx map[string]interface{}) interface{} {
        destination := args[0].(string)
        // Calculating optimal route using graph algorithms
        return map[string]interface{}{
            "tag": "RESULT",
            "content": map[string]string{
                "route_id": "R-9920",
                "estimation": "2 days",
            },
        }
    })

    worker.Start()
}
```

---

## 5. Enterprise CMS (PHP)
**Scenario**: A legacy news portal built on PHP/Laravel. FlowLang introduces an "Editorial Review" flow without rewriting the entire backend.

### The Worker (`editor.php`)
```php
<?php
require_once './sdk/php/FlowLang.php';

$worker = new \FlowLang\FlowWorker("EditorialBoard");

$worker->on("Judge", function($args, $kwargs, $context) {
    $articleId = $args[0];
    // Connect to legacy MySQL DB
    $db = new mysqli("localhost", "user", "pass", "cms");
    $res = $db->query("SELECT status FROM articles WHERE id = $articleId");
    
    $row = $res->fetch_assoc();
    $isReady = ($row['status'] === 'DRAFT_S2');

    return [
        "tag" => "REPORT",
        "content" => ["pass" => $isReady]
    ];
});

$worker->start();
```
