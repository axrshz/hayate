"""Shared token, timing, memory, and statistics helpers."""

from __future__ import annotations

import gc
import math
import statistics
import time
from contextlib import contextmanager
from typing import Callable, Iterable

import torch

from hayate.engine.constants import MAX_BATCH_SIZE
from hayate.engine.engine import Engine
from hayate.engine.request import Request


def synchronize() -> None:
    torch.cuda.synchronize()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def exact_token_sequence(tokenizer, length: int, offset: int = 0) -> list[int]:
    """Build deterministic model inputs with exactly ``length`` tokens."""
    base = tokenizer.encode(
        "A deterministic inference benchmark measures model execution without tokenization. ",
        add_special_tokens=False,
    )
    if not base:
        raise RuntimeError("benchmark seed text unexpectedly produced no tokens")
    shift = offset % len(base)
    rotated = base[shift:] + base[:shift]
    repeats = (length + len(rotated) - 1) // len(rotated)
    return (rotated * repeats)[:length]


def new_request(
    engine: Engine,
    input_tokens: int,
    output_tokens: int,
    request_id: int,
    offset: int = 0,
) -> Request:
    return Request(
        id=request_id,
        prompt="",
        prompt_tokens=exact_token_sequence(engine.tokenizer, input_tokens, offset),
        max_tokens=output_tokens,
    )


@contextmanager
def fixed_generation(engine: Engine):
    """Disable EOS termination so every request produces the requested length."""
    stop_ids = engine.sampler.stop_token_ids
    engine.sampler.stop_token_ids = set()
    try:
        yield
    finally:
        engine.sampler.stop_token_ids = stop_ids


def release_requests(engine: Engine, requests: Iterable[Request]) -> None:
    engine.scheduler.current_batch.clear()
    while not engine.scheduler.pool.empty():
        engine.scheduler.pool.get()
    for request in requests:
        if request.kv_cache is not None:
            request.kv_cache.reset()
    gc.collect()


def measure_cuda(operation: Callable[[], None]) -> dict[str, float]:
    """Measure end-to-end wall time and kernel-stream GPU time separately."""
    synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.reset_peak_memory_stats()
    wall_start = time.perf_counter()
    start_event.record()
    operation()
    end_event.record()
    end_event.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    return {
        "wall_seconds": wall_seconds,
        "gpu_seconds": start_event.elapsed_time(end_event) / 1000.0,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


def median_measurements(measurements: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: statistics.median(measurement[key] for measurement in measurements)
        for key in measurements[0]
    }


def estimated_kv_bytes(engine: Engine, batch_size: int, sequence_length: int) -> int:
    # BF16 K and V: layers * batch * heads * positions * head_dim * 2 tensors * 2 bytes.
    return (
        engine.num_layers
        * batch_size
        * engine.num_kv_groups
        * sequence_length
        * engine.head_dim
        * 2
        * 2
    )


def configuration_skip_reason(
    engine: Engine, batch_size: int, sequence_length: int
) -> str | None:
    if batch_size > MAX_BATCH_SIZE:
        return "skipped_batch_limit"
    if sequence_length > engine.max_position_embeddings:
        return "skipped_context_window"
    total = torch.cuda.get_device_properties(0).total_memory
    model = torch.cuda.memory_allocated()
    if model + estimated_kv_bytes(engine, batch_size, sequence_length) >= total * 0.85:
        return "skipped_memory"
    return None
