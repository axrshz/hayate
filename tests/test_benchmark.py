import argparse
import unittest

from benchmarks.cli import parse_int_list, parse_workloads
from benchmarks.common import percentile
from benchmarks.serving import combine_workload_runs, poisson_schedule


class BenchmarkMetricTests(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertEqual(percentile([1.0, 2.0, 3.0], 50), 2.0)
        self.assertAlmostEqual(percentile([1.0, 2.0], 95), 1.95)

    def test_matrix_parsers(self):
        self.assertEqual(parse_int_list("1,4,8"), (1, 4, 8))
        self.assertEqual(parse_workloads("128:64,512:128"), ((128, 64), (512, 128)))
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_workloads("128")

    def test_poisson_schedule_is_deterministic_and_monotonic(self):
        first = poisson_schedule(8, 2.0, 7)
        second = poisson_schedule(8, 2.0, 7)
        self.assertEqual(first, second)
        self.assertEqual(first[0], 0.0)
        self.assertTrue(all(left < right for left, right in zip(first, first[1:])))

    def test_workload_aggregation_weights_by_elapsed_time(self):
        template = {
            "requests": 2,
            "input_tokens": 10,
            "output_tokens": 5,
            "ttft_ms": [10.0, 20.0],
            "tpot_ms": [2.0, 4.0],
            "e2e_ms": [20.0, 40.0],
            "good_requests": 2,
            "peak_vram_gb": 9.0,
            "max_active_requests": 2,
            "scheduled_intervals": 0,
            "schedule_span_seconds": 0.0,
        }
        runs = [
            {**template, "elapsed_seconds": 1.0},
            {**template, "elapsed_seconds": 3.0},
        ]
        result = combine_workload_runs(runs)
        self.assertEqual(result["request_throughput"], 1.0)
        self.assertEqual(result["output_tokens_per_second"], 5.0)
        self.assertEqual(result["goodput_percent"], 100.0)


if __name__ == "__main__":
    unittest.main()
