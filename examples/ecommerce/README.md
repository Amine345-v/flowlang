# E-Commerce Order Lifecycle: The "Software Factory" Approach

This example demonstrates how to apply **FlowLang's Governance Principles** (Maestro, Teams, Checkpoints) to a typical E-Commerce workflow.

Instead of treating an Order as a simple database row that gets updated by random API calls, we treat it as a **Engineering Project** that must pass through a rigorous "Factory Line" of validation and processing.

## 1. The Concept: "The Order Factory"

Just as a software feature goes through `Design -> Impl -> QA -> Release`, an Order goes through:
`Validation -> Financial -> Logistics`.

We define this lifecycle explicitly using the **Maestro Process Tree**:

```flowlang
process order_tracker "Order Lifecycle" {
    root: "Order";
    branch "Order" -> ["Validation", "Financial", "Logistics"];
    
    node "Fraud-Check" { status: "pending"; };
    node "Inventory-Reserve" { status: "pending"; };
    node "Payment-Capture" { status: "pending"; };
    node "Shipping-Label" { status: "pending"; };
}
```

This ensures that **no step is skipped**. You cannot ship an order that hasn't passed the "Financial" branch.

## 2. The Teams (Professional Workers)

We employ 4 specialized "Professional Workers" implemented in **JavaScript (Node.js)** using the `flowlang-js` SDK. Each has a strict definition of what they can do.

| Team | Role | Verb | File |
|------|------|------|------|
| **fraud_guard** | Security Judge | `Judge` (Pass/Fail) | `fraud_engine.js` |
| **inventory_sys** | Warehouse Mgr | `Try` (Action) | `inventory_worker.js` |
| **bank_gateway** | Finance Officer | `Try` (Action) | `payment_worker.js` |
| **logistics** | Shipping Agent | `Search` (Find Option) | `logistics_worker.js` |

## 3. The Flow (`order_lifecycle.flow`)

The flow acts as the **Governor**. It does not "do" the work; it **orchestrates** the workers and ensures they follow the process.

### Phase 1: Validation
- **Fraud Check**: The `fraud_guard` judges the transaction risk. If rejected, the flow ends immediately.
- **Inventory**: The `inventory_sys` attempts to reserve stock. If out of stock, the flow ends.

### Phase 2: Financial
- **Payment**: Only if Phase 1 passes, the `bank_gateway` attempts to charge the card.
- **Audit**: The Maestro marks `Payment-Capture` as "Paid" only if the bank returns success.

### Phase 3: Logistics
- **Shipping**: The `logistics` team searches for the best carrier (FedEx/UPS) based on weight/destination.
- **Completion**: The final label is generated, and the Order is marked "Completed".

## 4. Developer Experience (DX)

You can build this app like a standard Node.js project.

### Project Structure
```text
/ecommerce
  ├── package.json          # Dependencies (flowlang-js)
  ├── order_lifecycle.flow  # The Governance Logic
  ├── api_simulation.js     # The "Frontend" (Next.js API Route)
  ├── fraud_engine.js       # Worker 1
  ├── inventory_worker.js   # Worker 2
  ├── payment_worker.js     # Worker 3
  └── logistics_worker.js   # Worker 4
```

### running the Simulation
```bash
npm install
node api_simulation.js
```

This will run two scenarios:
1.  **Standard Order**: Passes all gates -> Shipped.
2.  **Fraud Order**: Fails at Step 1 -> Rejected.
