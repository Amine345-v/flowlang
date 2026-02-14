"""
Performance benchmarks for FlowLang.
Tests parsing speed and runtime execution performance.
"""
import time
import statistics

import pytest

from flowlang.parser import parse
from flowlang.runtime import Runtime


# Number of repetitions for benchmarks
BENCHMARK_ITERATIONS = 50


class TestPerformance:
    """Performance test cases."""
    
    def test_parsing_performance(self):
        """Test the performance of parsing a flow definition."""
        flow_src = """
        team dev : Command<Search> [size=2];
        team qa : Command<Judge> [size=1];
        team ops : Command<Try> [size=1];
        flow benchmark_flow(using: dev, qa, ops) {
            checkpoint "develop" {
                dev.search("requirements");
                qa.judge("code", "quality");
            }
            checkpoint "test" {
                ops.try("fix");
                qa.judge("result", "pass");
            }
            checkpoint "deploy" {
                ops.try("deploy to staging");
            }
        }
        """
        
        times = []
        for _ in range(BENCHMARK_ITERATIONS):
            start_time = time.perf_counter()
            parse(flow_src)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        mean = statistics.mean(times)
        assert mean < 0.1, f"Parsing too slow: {mean*1000:.2f}ms mean"
    
    def test_runtime_load_performance(self):
        """Test the performance of loading (parse + semantic analysis)."""
        flow_src = """
        team dev : Command<Search> [size=2];
        team qa : Command<Judge> [size=1];
        flow load_test(using: dev, qa) {
            checkpoint "step1" {
                dev.search("query");
                qa.judge("result", "criteria");
            }
            checkpoint "step2" {
                dev.search("follow_up");
            }
        }
        """
        
        times = []
        for _ in range(BENCHMARK_ITERATIONS):
            rt = Runtime(dry_run=True)
            start_time = time.perf_counter()
            rt.load(flow_src)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        mean = statistics.mean(times)
        assert mean < 0.2, f"Loading too slow: {mean*1000:.2f}ms mean"
    
    def test_dry_run_execution_performance(self):
        """Test the performance of dry-run flow execution."""
        flow_src = """
        team dev : Command<Search> [size=1];
        flow exec_test(using: dev) {
            checkpoint "run" {
                dev.search("query");
            }
        }
        """
        
        times = []
        for _ in range(BENCHMARK_ITERATIONS):
            rt = Runtime(dry_run=True)
            rt.load(flow_src)
            start_time = time.perf_counter()
            rt.run_flow("exec_test")
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        mean = statistics.mean(times)
        assert mean < 0.5, f"Dry-run execution too slow: {mean*1000:.2f}ms mean"
    
    def test_deep_merge_performance(self):
        """Test the performance of deep_merge operations."""
        rt = Runtime()
        
        # Create large dicts to merge
        dict_a = {f"key_{i}": [j for j in range(10)] for i in range(100)}
        dict_b = {f"key_{i}": [j + 10 for j in range(10)] for i in range(100)}
        
        times = []
        for _ in range(BENCHMARK_ITERATIONS):
            start_time = time.perf_counter()
            rt._deep_merge(dict_a.copy(), dict_b.copy())
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        mean = statistics.mean(times)
        assert mean < 0.01, f"Deep merge too slow: {mean*1000:.2f}ms mean"
