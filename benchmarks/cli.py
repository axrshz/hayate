"""Command-line orchestration for the Hayate benchmark suites."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from hayate.engine.engine import Engine

from .micro import run_microbenchmarks
from .reporting import print_results
from .serving import run_offline_benchmarks, run_online_benchmarks


MODEL_NAME = "Qwen/Qwen3-4B"
DEFAULT_BATCH_SIZES = (1, 4, 8)
DEFAULT_CONTEXT_LENGTHS = (128, 512, 2048)
DEFAULT_WORKLOADS = ((128, 128), (512, 128), (2048, 256))
DEFAULT_ARRIVAL_RATES = (0.5, 1.0, 2.0, 4.0)


def parse_int_list(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected comma-separated positive integers"
        ) from exc
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def parse_float_list(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected comma-separated positive numbers"
        ) from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive numbers")
    return values


def parse_workloads(value: str) -> tuple[tuple[int, int], ...]:
    workloads = []
    try:
        for item in value.split(","):
            input_tokens, output_tokens = item.strip().split(":", maxsplit=1)
            pair = (int(input_tokens), int(output_tokens))
            if pair[0] < 1 or pair[1] < 1:
                raise ValueError
            workloads.append(pair)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected input:output pairs, for example 128:128,512:128"
        ) from exc
    if not workloads:
        raise argparse.ArgumentTypeError("at least one workload is required")
    return tuple(workloads)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Hayate prefill, decode, offline throughput, and online serving."
    )
    parser.add_argument(
        "--suite", choices=("micro", "offline", "online", "all"), default="micro"
    )
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--batch-sizes", type=parse_int_list, default=DEFAULT_BATCH_SIZES
    )
    parser.add_argument(
        "--context-lengths", type=parse_int_list, default=DEFAULT_CONTEXT_LENGTHS
    )
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--workloads", type=parse_workloads, default=DEFAULT_WORKLOADS)
    parser.add_argument("--requests", type=int, default=12)
    parser.add_argument(
        "--arrival-rates", type=parse_float_list, default=DEFAULT_ARRIVAL_RATES
    )
    parser.add_argument("--online-input-tokens", type=int, default=512)
    parser.add_argument("--online-output-tokens", type=int, default=128)
    parser.add_argument("--ttft-slo-ms", type=float, default=500.0)
    parser.add_argument("--tpot-slo-ms", type=float, default=50.0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    positive = (
        args.repetitions,
        args.decode_steps,
        args.requests,
        args.online_input_tokens,
        args.online_output_tokens,
    )
    if any(value < 1 for value in positive):
        parser.error("repetitions, steps, requests, and token lengths must be positive")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the benchmark requires an NVIDIA CUDA GPU")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    load_started = time.perf_counter()
    engine = Engine(args.model, compile=args.compile, compile_mode=args.compile_mode)
    model_load_seconds = time.perf_counter() - load_started
    device = torch.cuda.get_device_name(0)
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__} (CUDA {torch.version.cuda})")
    print(f"Model load: {model_load_seconds:.2f}s")
    print(f"Compile: {args.compile} ({args.compile_mode})")
    print("Output lengths are fixed; tokenization and model loading are outside timed inference.")

    rows = []
    if args.suite in ("micro", "all"):
        rows.extend(run_microbenchmarks(engine, args))
    if args.suite in ("offline", "all"):
        rows.extend(run_offline_benchmarks(engine, args))
    if args.suite in ("online", "all"):
        rows.extend(run_online_benchmarks(engine, args))
    print_results(rows)

    if args.json:
        payload = {
            "metadata": {
                "model": args.model,
                "device": device,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "compile": args.compile,
                "compile_mode": args.compile_mode,
                "seed": args.seed,
                "repetitions": args.repetitions,
                "model_load_seconds": model_load_seconds,
                "ttft_slo_ms": args.ttft_slo_ms,
                "tpot_slo_ms": args.tpot_slo_ms,
            },
            "results": rows,
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.json}")
