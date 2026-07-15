"""Offline saturation and Poisson-arrival serving benchmarks."""

from __future__ import annotations

import random
import time

import torch

from hayate.engine.constants import MAX_BATCH_SIZE
from hayate.engine.engine import Engine
from hayate.engine.request import Request

from .common import (
    configuration_skip_reason,
    fixed_generation,
    new_request,
    percentile,
    release_requests,
    synchronize,
)


def poisson_schedule(
    request_count: int, requests_per_second: float, seed: int
) -> list[float]:
    rng = random.Random(seed)
    schedule = [0.0]
    for _ in range(1, request_count):
        schedule.append(schedule[-1] + rng.expovariate(requests_per_second))
    return schedule


def run_scheduled_workload(
    engine: Engine,
    input_tokens: int,
    output_tokens: int,
    schedule: list[float],
    ttft_slo_ms: float,
    tpot_slo_ms: float,
    offset: int = 0,
) -> dict:
    requests = [
        new_request(engine, input_tokens, output_tokens, index, offset=offset + index)
        for index in range(len(schedule))
    ]
    submitted: list[Request] = []
    first_token_at: dict[int, float] = {}
    completed_at: dict[int, float] = {}
    next_request = 0
    max_active = 0

    synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    while (
        next_request < len(requests)
        or engine.scheduler.current_batch
        or not engine.scheduler.pool.empty()
    ):
        elapsed = time.perf_counter() - started
        while next_request < len(requests) and schedule[next_request] <= elapsed:
            request = requests[next_request]
            engine.add_request(request)
            submitted.append(request)
            next_request += 1

        progressed = engine.generate()
        observed_at = time.perf_counter()
        max_active = max(max_active, len(engine.scheduler.current_batch))
        for request in submitted:
            if request.tokens and request.id not in first_token_at:
                first_token_at[request.id] = observed_at
            if request.is_completed and request.id not in completed_at:
                completed_at[request.id] = observed_at

        if not progressed and next_request < len(requests):
            delay = schedule[next_request] - (time.perf_counter() - started)
            if delay > 0:
                time.sleep(delay)

    synchronize()
    elapsed = time.perf_counter() - started
    ttft_ms = []
    tpot_ms = []
    e2e_ms = []
    good = 0
    for index, request in enumerate(requests):
        arrival = started + schedule[index]
        first = first_token_at[request.id]
        completed = completed_at[request.id]
        request_ttft = (first - arrival) * 1000
        request_e2e = (completed - arrival) * 1000
        request_tpot = (
            (completed - first) * 1000 / (len(request.tokens) - 1)
            if len(request.tokens) > 1
            else 0.0
        )
        ttft_ms.append(request_ttft)
        tpot_ms.append(request_tpot)
        e2e_ms.append(request_e2e)
        if request_ttft <= ttft_slo_ms and request_tpot <= tpot_slo_ms:
            good += 1

    result = {
        "elapsed_seconds": elapsed,
        "requests": len(requests),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "request_throughput": len(requests) / elapsed,
        "prompt_tokens_per_second": len(requests) * input_tokens / elapsed,
        "output_tokens_per_second": len(requests) * output_tokens / elapsed,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "e2e_ms": e2e_ms,
        "good_requests": good,
        "goodput_requests_per_second": good / elapsed,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        "max_active_requests": max_active,
        "scheduled_intervals": max(len(schedule) - 1, 0),
        "schedule_span_seconds": schedule[-1] if schedule else 0.0,
    }
    release_requests(engine, requests)
    return result


def combine_workload_runs(runs: list[dict]) -> dict:
    elapsed = sum(run["elapsed_seconds"] for run in runs)
    requests = sum(run["requests"] for run in runs)
    input_tokens = runs[0]["input_tokens"]
    output_tokens = runs[0]["output_tokens"]
    ttft = [value for run in runs for value in run["ttft_ms"]]
    tpot = [value for run in runs for value in run["tpot_ms"]]
    e2e = [value for run in runs for value in run["e2e_ms"]]
    good = sum(run["good_requests"] for run in runs)
    scheduled_intervals = sum(run["scheduled_intervals"] for run in runs)
    schedule_span = sum(run["schedule_span_seconds"] for run in runs)
    return {
        "requests": requests,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "request_throughput": requests / elapsed,
        "prompt_tokens_per_second": requests * input_tokens / elapsed,
        "output_tokens_per_second": requests * output_tokens / elapsed,
        "ttft_p50_ms": percentile(ttft, 50),
        "ttft_p95_ms": percentile(ttft, 95),
        "ttft_p99_ms": percentile(ttft, 99),
        "tpot_p50_ms": percentile(tpot, 50),
        "tpot_p95_ms": percentile(tpot, 95),
        "tpot_p99_ms": percentile(tpot, 99),
        "e2e_p50_ms": percentile(e2e, 50),
        "e2e_p95_ms": percentile(e2e, 95),
        "e2e_p99_ms": percentile(e2e, 99),
        "goodput_requests_per_second": good / elapsed,
        "goodput_percent": good / requests * 100,
        "peak_vram_gb": max(run["peak_vram_gb"] for run in runs),
        "max_active_requests": max(run["max_active_requests"] for run in runs),
        "realized_offered_requests_per_second": (
            scheduled_intervals / schedule_span if schedule_span > 0 else None
        ),
    }


def _warm_serving_path(
    engine: Engine, input_tokens: int, output_tokens: int, batch_size: int
) -> dict:
    return run_scheduled_workload(
        engine,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        schedule=[0.0] * min(batch_size, MAX_BATCH_SIZE),
        ttft_slo_ms=float("inf"),
        tpot_slo_ms=float("inf"),
    )


def run_offline_benchmarks(engine: Engine, args) -> list[dict]:
    rows = []
    with fixed_generation(engine):
        for input_tokens, output_tokens in args.workloads:
            for batch_size in args.batch_sizes:
                skip_reason = configuration_skip_reason(
                    engine, batch_size, input_tokens + output_tokens - 1
                )
                if skip_reason:
                    rows.append(
                        {
                            "suite": "offline",
                            "batch_size": batch_size,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "status": skip_reason,
                        }
                    )
                    continue

                warmup = _warm_serving_path(
                    engine, input_tokens, output_tokens, batch_size
                )
                runs = [
                    run_scheduled_workload(
                        engine,
                        input_tokens,
                        output_tokens,
                        [0.0] * batch_size,
                        args.ttft_slo_ms,
                        args.tpot_slo_ms,
                        offset=rep * batch_size,
                    )
                    for rep in range(args.repetitions)
                ]
                row = combine_workload_runs(runs)
                row.update(
                    {
                        "suite": "offline",
                        "batch_size": batch_size,
                        "warmup_seconds": warmup["elapsed_seconds"],
                        "status": "ok",
                    }
                )
                rows.append(row)
    return rows


def run_online_benchmarks(engine: Engine, args) -> list[dict]:
    with fixed_generation(engine):
        active_limit = min(args.requests, MAX_BATCH_SIZE)
        skip_reason = configuration_skip_reason(
            engine,
            active_limit,
            args.online_input_tokens + args.online_output_tokens - 1,
        )
        if skip_reason:
            return [
                {
                    "suite": "online",
                    "requests": args.requests,
                    "input_tokens": args.online_input_tokens,
                    "output_tokens": args.online_output_tokens,
                    "status": skip_reason,
                }
            ]

        warmup = _warm_serving_path(
            engine,
            args.online_input_tokens,
            args.online_output_tokens,
            active_limit,
        )
        rows = []
        for rate_index, rate in enumerate(args.arrival_rates):
            schedules = [
                poisson_schedule(args.requests, rate, args.seed + rate_index * 1000 + rep)
                for rep in range(args.repetitions)
            ]
            runs = [
                run_scheduled_workload(
                    engine,
                    args.online_input_tokens,
                    args.online_output_tokens,
                    schedule,
                    args.ttft_slo_ms,
                    args.tpot_slo_ms,
                    offset=rep * args.requests,
                )
                for rep, schedule in enumerate(schedules)
            ]
            row = combine_workload_runs(runs)
            row.update(
                {
                    "suite": "online",
                    "offered_requests_per_second": rate,
                    "warmup_seconds": warmup["elapsed_seconds"],
                    "status": "ok",
                }
            )
            rows.append(row)
        return rows
