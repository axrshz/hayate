"""Controlled prefill and decode microbenchmarks."""

from __future__ import annotations

from hayate.engine.engine import Engine
from hayate.engine.request import Request

from .common import (
    configuration_skip_reason,
    fixed_generation,
    measure_cuda,
    median_measurements,
    new_request,
    release_requests,
    synchronize,
)


def _prepare_requests(
    engine: Engine,
    batch_size: int,
    context_length: int,
    output_tokens: int,
) -> list[Request]:
    requests = [
        new_request(engine, context_length, output_tokens, index, offset=index)
        for index in range(batch_size)
    ]
    for request in requests:
        engine._prepare_request(request)
    return requests


def _one_prefill(
    engine: Engine, batch_size: int, context_length: int
) -> tuple[dict, list[Request]]:
    requests = _prepare_requests(engine, batch_size, context_length, output_tokens=1)
    measurement = measure_cuda(lambda: engine.prefill_batch(requests))
    return measurement, requests


def _one_decode(
    engine: Engine,
    batch_size: int,
    context_length: int,
    decode_steps: int,
) -> tuple[dict, list[Request]]:
    requests = _prepare_requests(
        engine, batch_size, context_length, output_tokens=decode_steps + 1
    )
    engine.prefill_batch(requests)
    synchronize()

    def decode() -> None:
        for _ in range(decode_steps):
            engine.decode_batch(requests)

    return measure_cuda(decode), requests


def run_microbenchmarks(engine: Engine, args) -> list[dict]:
    rows = []
    with fixed_generation(engine):
        for context_length in args.context_lengths:
            for batch_size in args.batch_sizes:
                skip_reason = configuration_skip_reason(engine, batch_size, context_length)
                if skip_reason:
                    rows.append(
                        {
                            "suite": "prefill",
                            "batch_size": batch_size,
                            "context_tokens": context_length,
                            "status": skip_reason,
                        }
                    )
                    continue

                warmup, warmup_requests = _one_prefill(
                    engine, batch_size, context_length
                )
                release_requests(engine, warmup_requests)
                samples = []
                for _ in range(args.repetitions):
                    sample, requests = _one_prefill(
                        engine, batch_size, context_length
                    )
                    samples.append(sample)
                    release_requests(engine, requests)
                result = median_measurements(samples)
                prompt_tokens = batch_size * context_length
                rows.append(
                    {
                        "suite": "prefill",
                        "batch_size": batch_size,
                        "context_tokens": context_length,
                        "wall_ms": result["wall_seconds"] * 1000,
                        "gpu_ms": result["gpu_seconds"] * 1000,
                        "prompt_tokens_per_second": prompt_tokens / result["wall_seconds"],
                        "gpu_prompt_tokens_per_second": prompt_tokens / result["gpu_seconds"],
                        "peak_vram_gb": result["peak_vram_gb"],
                        "warmup_seconds": warmup["wall_seconds"],
                        "status": "ok",
                    }
                )

        for context_length in args.context_lengths:
            for batch_size in args.batch_sizes:
                final_length = context_length + args.decode_steps
                skip_reason = configuration_skip_reason(engine, batch_size, final_length)
                if skip_reason:
                    rows.append(
                        {
                            "suite": "decode",
                            "batch_size": batch_size,
                            "context_tokens": context_length,
                            "decode_steps": args.decode_steps,
                            "status": skip_reason,
                        }
                    )
                    continue

                warmup, warmup_requests = _one_decode(
                    engine, batch_size, context_length, args.decode_steps
                )
                release_requests(engine, warmup_requests)
                samples = []
                for _ in range(args.repetitions):
                    sample, requests = _one_decode(
                        engine, batch_size, context_length, args.decode_steps
                    )
                    samples.append(sample)
                    release_requests(engine, requests)
                result = median_measurements(samples)
                output_tokens = batch_size * args.decode_steps
                rows.append(
                    {
                        "suite": "decode",
                        "batch_size": batch_size,
                        "context_tokens": context_length,
                        "decode_steps": args.decode_steps,
                        "wall_ms_per_step": result["wall_seconds"] * 1000 / args.decode_steps,
                        "gpu_ms_per_step": result["gpu_seconds"] * 1000 / args.decode_steps,
                        "output_tokens_per_second": output_tokens / result["wall_seconds"],
                        "gpu_output_tokens_per_second": output_tokens / result["gpu_seconds"],
                        "peak_vram_gb": result["peak_vram_gb"],
                        "warmup_seconds": warmup["wall_seconds"],
                        "status": "ok",
                    }
                )
    return rows
