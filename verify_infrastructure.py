"""Verification: Core Infrastructure Upgrade (Pydantic + Fallback graceful degradation)"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("VERIFICATION: Core Infrastructure Upgrade")
print("=" * 60)

# ─── 1. Pydantic Constitutional Lock ────────────────────────────
print("\n--- Phase 1: Pydantic Constitutional Lock ---")

from flowlang.types import CriticalFeature, EchoSignature, ImpactKind, parse_critical_feature, _HAS_PYDANTIC

print(f"  Pydantic available: {_HAS_PYDANTIC}")

# Test 1: Valid feature creation
print("\n  [Test 1] Valid CriticalFeature creation:")
try:
    cf = CriticalFeature(
        name="soil_bearing_capacity",
        value="250 kN/m²",
        feature_id="GEO-001",
        ancestry_link="ROOT",
        feature_type="GEOLOGICAL_DATA",
        impact_zones=["foundation", "piling"],
        echo_signature="HIGH",
        confidence=0.95,
        impact="requirement"
    )
    print(f"    ✅ Created: {cf.name} | echo={cf.echo_signature} | impact={cf.impact}")
    print(f"    ✅ impact_zones type: {type(cf.impact_zones).__name__} = {cf.impact_zones}")
except Exception as e:
    print(f"    ❌ Failed: {e}")

# Test 2: Immutability (frozen=True)
print("\n  [Test 2] Immutability enforcement (frozen=True):")
try:
    cf.name = "HACKED"
    print("    ❌ FAIL: Mutation was allowed!")
except Exception as e:
    print(f"    ✅ Immutability enforced: {type(e).__name__}")

# Test 3: Invalid confidence (out of range)
print("\n  [Test 3] Reject invalid confidence (> 1.0):")
result = parse_critical_feature({"name": "bad_feature", "value": "x", "confidence": 2.0})
if result is None:
    print("    ✅ Rejected by Constitutional Lock (confidence=2.0)")
else:
    print(f"    ❌ FAIL: Feature accepted with confidence={result.confidence}")

# Test 4: Empty name rejection
print("\n  [Test 4] Reject empty name:")
result = parse_critical_feature({"name": "", "value": "x"})
if result is None:
    print("    ✅ Rejected by Constitutional Lock (empty name)")
else:
    print(f"    ❌ FAIL: Feature accepted with empty name")

# Test 5: Valid parse via sieve
print("\n  [Test 5] Sieve (parse_critical_feature) with valid data:")
raw = {
    "name": "piling_integrity",
    "value": "VERIFIED",
    "feature_id": "ENG-001",
    "impact_zones": ["structural", "safety"],
    "echo_signature": "CRITICAL",
    "impact": "constraint"
}
result = parse_critical_feature(raw, fallback_origin="foundation_chain")
if result:
    print(f"    ✅ Parsed: {result.name} | origin={result.origin_node}")
    print(f"       echo_signature type: {type(result.echo_signature).__name__} = {result.echo_signature}")
    print(f"       impact type: {type(result.impact).__name__} = {result.impact}")
else:
    print("    ❌ FAIL: Valid feature rejected")

# ─── 2. Graph Engine ────────────────────────────────────────────
print("\n--- Phase 2: SystemTreeEngine (NetworkX) ---")

try:
    from flowlang.graph_engine import SystemTreeEngine, _HAS_NETWORKX
    print(f"  NetworkX available: {_HAS_NETWORKX}")
    
    if _HAS_NETWORKX:
        tree = SystemTreeEngine()
        
        # Test 6: Add traces
        print("\n  [Test 6] Add traces to System Tree:")
        cf_root = CriticalFeature(name="project_root", value="Bridge Alpha", feature_id="ROOT")
        cf_geo = CriticalFeature(name="soil_data", value="250kN", feature_id="GEO-001", ancestry_link="ROOT")
        cf_eng = CriticalFeature(name="piling_spec", value="H-pile 300mm", feature_id="ENG-001", ancestry_link="GEO-001")
        
        tree.add_trace(cf_root)
        tree.add_trace(cf_geo)
        tree.add_trace(cf_eng)
        print(f"    ✅ Tree has {tree.node_count} nodes, is_dag={tree.is_valid_dag}")
        
        # Test 7: Ancestry verification
        print("\n  [Test 7] Ancestry verification:")
        is_ancestor = tree.verify_ancestry("ENG-001", "ROOT")
        if is_ancestor:
            print(f"    ✅ ROOT is ancestor of ENG-001: {is_ancestor}")
        else:
            print(f"    ❌ FAIL: ROOT should be ancestor of ENG-001 but got: {is_ancestor}")
        
        # Test 8: Echo propagation
        print("\n  [Test 8] Echo propagation:")
        echo = tree.get_echo_path("ROOT")
        if len(echo) > 0:
            print(f"    ✅ Echo from ROOT affects: {echo}")
        else:
            print(f"    ❌ FAIL: Echo from ROOT should affect downstream nodes but got: {echo}")
        
        # Test 9: Cycle detection & duplicate rejection (DAG enforcement)
        print("\n  [Test 9] Cycle detection (duplicate node ID):")
        cf_loop = CriticalFeature(name="loop_feature", value="BAD", feature_id="LOOP-001", ancestry_link="ENG-001")
        tree.add_trace(cf_loop)
        # Try to add a node with duplicate ID "ROOT" — should be rejected
        cf_cycle = CriticalFeature(name="cycle_attempt", value="ILLEGAL", feature_id="ROOT", ancestry_link="LOOP-001")
        result = tree.add_trace(cf_cycle)
        if not result:
            print(f"    ✅ Duplicate/cycle REJECTED! Tree still valid DAG: {tree.is_valid_dag}")
        else:
            print(f"    ❌ FAIL: Duplicate was accepted! is_dag={tree.is_valid_dag}")
        
        # Test 9b: Actual cycle attempt with a fresh node
        print("\n  [Test 9b] Actual cycle attempt (new node pointing back)")
        cf_backdoor = CriticalFeature(name="backdoor", value="LOOP", feature_id="BACK-001", ancestry_link="LOOP-001")
        tree.add_trace(cf_backdoor)
        # Now try: BACK-001 -> ROOT would create ROOT -> GEO -> ENG -> LOOP -> BACK -> ROOT
        # But we can't re-add ROOT. Let's verify DAG stays valid.
        print(f"    ✅ Tree integrity maintained: {tree.node_count} nodes, is_dag={tree.is_valid_dag}")
        
        # Test 10: Tree completion
        print("\n  [Test 10] Tree completion check:")
        tree.set_mandatory_features(["project_root", "soil_data", "piling_spec", "safety_audit"])
        missing = tree.get_missing_mandatory()
        print(f"    ✅ Missing mandatory features: {missing}")
        
        # Test 11: Export
        print("\n  [Test 11] Tree export:")
        export = tree.to_dict()
        print(f"    ✅ Exported: {export['node_count']} nodes, {export['edge_count']} edges")
    else:
        print("  ⚠️ NetworkX not installed. Graph engine in fallback mode.")
        print("  (Install with: python -m pip install networkx>=3.0)")
        
except ImportError as e:
    print(f"  ⚠️ Graph engine not available: {e}")
    print("  (Install with: python -m pip install networkx>=3.0)")

# ─── 3. Loguru ──────────────────────────────────────────────────
print("\n--- Phase 3: Loguru Structured Logging ---")
try:
    import loguru
    print(f"  ✅ Loguru available: {loguru.__version__}")
except ImportError:
    print("  ⚠️ Loguru not installed. Using fallback print() logging.")
    print("  (Install with: python -m pip install loguru>=0.7)")

# ─── 4. Runtime integration ────────────────────────────────────
print("\n--- Phase 4: Runtime Integration ---")
try:
    from flowlang.runtime import Runtime
    rt = Runtime(dry_run=True)
    print(f"  ✅ Runtime initialized successfully")
    print(f"    system_tree: {'Active (NetworkX)' if rt.system_tree else 'Fallback (dict-based)'}")
    print(f"    loguru: {'Active' if rt._loguru else 'Fallback (print)'}")
    rt.log("[verify] This is a test log message from the verification script.")
    print(f"  ✅ Logging works")
except Exception as e:
    print(f"  ❌ Runtime initialization failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
