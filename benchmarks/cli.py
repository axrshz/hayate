"""Minimal prefill and decode benchmark for Hayate."""

import argparse
import gc
import json
import statistics
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from hayate.engine.constants import MAX_BATCH_SIZE
from hayate.engine.engine import Engine
from hayate.engine.request import Request


MODEL_NAME = "Qwen/Qwen3-4B"


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Hayate prefill and decode performance."
    )
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--context-tokens", type=positive_int, default=512)
    parser.add_argument("--decode-steps", type=positive_int, default=64)
    parser.add_argument("--repetitions", type=positive_int, default=3)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--json", type=Path)
    return parser.parse_args()


def exact_tokens(tokenizer, length: int, offset: int) -> list[int]:
    seed = tokenizer.encode(
        "A deterministic inference benchmark measures model execution. ",
        add_special_tokens=False,
    )
    if not seed:
        raise RuntimeError("benchmark seed text produced no tokens")
    shift = offset % len(seed)
    seed = seed[shift:] + seed[:shift]
    return (seed * ((length + len(seed) - 1) // len(seed)))[:length]


def make_requests(
    engine: Engine, batch_size: int, context_tokens: int, output_tokens: int
) -> list[Request]:
    requests = [
        Request(
            id=index,
            prompt_tokens=exact_tokens(engine.tokenizer, context_tokens, index),
            max_tokens=output_tokens,
        )
        for index in range(batch_size)
    ]
    for request in requests:
        engine._prepare_request(request)
    return requests


@contextmanager
def fixed_generation(engine: Engine):
    stop_ids = engine.sampler.stop_token_ids
    engine.sampler.stop_token_ids = set()
    try:
        yield
    finally:
        engine.sampler.stop_token_ids = stop_ids


def release(engine: Engine, requests: list[Request]) -> None:
    engine.scheduler.current_batch.clear()
    while not engine.scheduler.pool.empty():
        engine.scheduler.pool.get()
    for request in requests:
        if request.kv_cache is not None:
            request.kv_cache.reset()
    gc.collect()


def measure(operation) -> tuple[float, float]:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    operation()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    return elapsed, peak_vram_gb


def prefill_once(engine: Engine, args) -> tuple[float, float]:
    requests = make_requests(engine, args.batch_size, args.context_tokens, 1)
    result = measure(lambda: engine.prefill_batch(requests))
    release(engine, requests)
    return result


def decode_once(engine: Engine, args) -> tuple[float, float]:
    requests = make_requests(
        engine, args.batch_size, args.context_tokens, args.decode_steps + 1
    )
    engine.prefill_batch(requests)

    def decode():
        for _ in range(args.decode_steps):
            engine.decode_batch(requests)

    result = measure(decode)
    release(engine, requests)
    return result


def median(samples: list[tuple[float, float]]) -> tuple[float, float]:
    return (
        statistics.median(sample[0] for sample in samples),
        statistics.median(sample[1] for sample in samples),
    )


def run(engine: Engine, args) -> list[dict]:
    if args.batch_size > MAX_BATCH_SIZE:
        raise ValueError(f"batch size exceeds engine limit of {MAX_BATCH_SIZE}")
    if args.context_tokens + args.decode_steps > engine.max_position_embeddings:
        raise ValueError(
            f"context and decode steps exceed the {engine.max_position_embeddings}-token window"
        )

    with fixed_generation(engine):
        prefill_once(engine, args)
        prefill_seconds, prefill_vram = median(
            [prefill_once(engine, args) for _ in range(args.repetitions)]
        )

        decode_once(engine, args)
        decode_seconds, decode_vram = median(
            [decode_once(engine, args) for _ in range(args.repetitions)]
        )

    prompt_tokens = args.batch_size * args.context_tokens
    output_tokens = args.batch_size * args.decode_steps
    return [
        {
            "phase": "prefill",
            "latency_ms": prefill_seconds * 1000,
            "tokens_per_second": prompt_tokens / prefill_seconds,
            "peak_vram_gb": prefill_vram,
        },
        {
            "phase": "decode",
            "latency_ms_per_step": decode_seconds * 1000 / args.decode_steps,
            "tokens_per_second": output_tokens / decode_seconds,
            "peak_vram_gb": decode_vram,
        },
    ]


def print_results(results: list[dict]) -> None:
    prefill, decode = results
    print(
        f"\nprefill: {prefill['latency_ms']:.2f} ms, "
        f"{prefill['tokens_per_second']:.2f} tok/s, "
        f"{prefill['peak_vram_gb']:.2f} gb peak vram"
    )
    print(
        f"decode: {decode['latency_ms_per_step']:.2f} ms/step, "
        f"{decode['tokens_per_second']:.2f} tok/s, "
        f"{decode['peak_vram_gb']:.2f} gb peak vram"
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the benchmark requires an NVIDIA CUDA GPU")

    load_started = time.perf_counter()
    engine = Engine(args.model, compile=args.compile, compile_mode=args.compile_mode)
    model_load_seconds = time.perf_counter() - load_started

    print(f"model: {args.model}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(
        f"workload: batch={args.batch_size}, context={args.context_tokens}, "
        f"decode={args.decode_steps}, repetitions={args.repetitions}"
    )
    results = run(engine, args)
    print_results(results)

    if args.json:
        payload = {
            "config": {
                "model": args.model,
                "device": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "compile": args.compile,
                "compile_mode": args.compile_mode,
                "batch_size": args.batch_size,
                "context_tokens": args.context_tokens,
                "decode_steps": args.decode_steps,
                "repetitions": args.repetitions,
                "model_load_seconds": model_load_seconds,
            },
            "results": results,
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
