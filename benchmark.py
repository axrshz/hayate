import argparse
import random
import string
import time
import torch
from transformers import AutoTokenizer

from hayate.engine.engine import Engine


MODEL_NAME = "Qwen/Qwen3-4B"
DEFAULT_SEED = 1337
DEFAULT_MAX_TOKENS = 256
DEFAULT_REPETITIONS = 5
DEFAULT_PROMPT_TOKENS = 224


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize_device():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def random_fragment(rng, min_length=3, max_length=10):
    return "".join(rng.choices(string.ascii_lowercase, k=rng.randint(min_length, max_length)))


def build_prompt(tokenizer, rng, target_tokens):
    fragments = []
    prompt_tokens = 0

    while prompt_tokens < target_tokens:
        fragments.append(random_fragment(rng))
        prompt = " ".join(fragments)
        prompt_tokens = len(tokenizer.encode(prompt))

    return prompt, prompt_tokens


def generate_requests(
    tokenizer,
    n_requests,
    max_tokens,
    seed,
    prompt_tokens=DEFAULT_PROMPT_TOKENS,
):
    rng = random.Random(seed)
    prompt, actual_prompt_tokens = build_prompt(tokenizer, rng, prompt_tokens)

    return [
        {
            "prompt": prompt,
            "prompt_tokens": actual_prompt_tokens,
            "max_tokens": max_tokens,
        }
        for _ in range(n_requests)
    ]


def print_header(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def percentile(values, pct):
    if not values:
        return 0.0

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    idx = (len(ordered) - 1) * (pct / 100.0)
    lower = int(idx)
    upper = min(lower + 1, len(ordered) - 1)
    weight = idx - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def cleanup_engine(engine):
    del engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_request_set_summary(requests, repetitions, seed, device_name, vram, compile_model):
    prompt_tokens = [req["prompt_tokens"] for req in requests]
    print_header("Workload")
    print(f"  model:                {MODEL_NAME}")
    print(f"  device:               {device_name} ({vram})")
    print(f"  torch.compile:        {'on' if compile_model else 'off'}")
    if compile_model:
        print("  compile note:         first compiled runs may include noticeable upfront overhead")
    print(f"  seed:                 {seed}")
    print(f"  requests:             {len(requests)}")
    print(f"  repetitions:          {repetitions}")
    print("  workload shape:       fixed")
    print(f"  prompt tokens/req:    {prompt_tokens[0]}")
    print(f"  max_tokens/request:   {requests[0]['max_tokens']}")


def print_compact_summary(benchmark_rows):
    print_header("Benchmark Summary")
    print(f"  {'mode':<20} {'mean':>10} {'p50':>10} {'p95':>10} {'total tok/s':>12}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")

    for title, result, _ in benchmark_rows:
        print(
            f"  {title:<20} "
            f"{sum(result['latencies']) / len(result['latencies']):>9.3f}s "
            f"{percentile(result['latencies'], 50):>9.3f}s "
            f"{percentile(result['latencies'], 95):>9.3f}s "
            f"{result['total_tokens'] / result['elapsed']:>11.2f}"
        )


def print_verbose_summary(benchmark_rows):
    print_header("Verbose Summary")
    print(f"  {'mode':<20} {'mean':>10} {'p50':>10} {'p95':>10} {'gen tok/s':>12} {'total tok/s':>12}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*12}")

    for title, result, note in benchmark_rows:
        print(
            f"  {title:<20} "
            f"{sum(result['latencies']) / len(result['latencies']):>9.3f}s "
            f"{percentile(result['latencies'], 50):>9.3f}s "
            f"{percentile(result['latencies'], 95):>9.3f}s "
            f"{result['generated_tokens'] / result['elapsed']:>11.2f} "
            f"{result['total_tokens'] / result['elapsed']:>11.2f}"
        )
        print(f"  note: {note}")


def print_benchmark_summary(title, latencies, elapsed, generated_tokens, total_tokens, latency_unit, wall_time_note=None):
    print_header(title)
    print(f"  latency unit:    {latency_unit}")
    print(f"  samples:         {len(latencies)}")
    print(f"  mean latency:    {sum(latencies) / len(latencies):.3f}s")
    print(f"  p50 latency:     {percentile(latencies, 50):.3f}s")
    print(f"  p95 latency:     {percentile(latencies, 95):.3f}s")
    print(f"  wall time:       {elapsed:.3f}s")
    if wall_time_note:
        print(f"  wall time note:  {wall_time_note}")
    print(f"  generated tokens:{generated_tokens:>11}")
    print(f"  total tokens:    {total_tokens:>11}")
    print(f"  generated_tok/s: {generated_tokens / elapsed:>11.2f}")
    print(f"  total_tok/s:     {total_tokens / elapsed:>11.2f}")


def warmup_single_request(engine, request):
    engine.generate_text(request["prompt"], max_tokens=request["max_tokens"])
    synchronize_device()


def benchmark_single_request(engine, requests, repetitions):
    warmup_single_request(engine, requests[0])

    latencies = []
    generated_tokens = 0
    total_tokens = 0
    total_elapsed = 0.0

    for _ in range(repetitions):
        for request in requests:
            synchronize_device()
            start = time.perf_counter()
            result = engine.generate_text(request["prompt"], max_tokens=request["max_tokens"])
            synchronize_device()
            elapsed = time.perf_counter() - start

            generated = len(result.tokens)
            latencies.append(elapsed)
            generated_tokens += generated
            total_tokens += request["prompt_tokens"] + generated
            total_elapsed += elapsed

    return {
        "latencies": latencies,
        "elapsed": total_elapsed,
        "generated_tokens": generated_tokens,
        "total_tokens": total_tokens,
    }


def warmup_static_batch(engine, requests):
    warmup_prompts = [req["prompt"] for req in requests]
    engine.generate_text(warmup_prompts, max_tokens=requests[0]["max_tokens"])
    synchronize_device()


def benchmark_static_batch(engine, requests, repetitions):
    warmup_static_batch(engine, requests)

    prompts = [req["prompt"] for req in requests]
    prompt_tokens = sum(req["prompt_tokens"] for req in requests)

    latencies = []
    generated_tokens = 0
    total_tokens = 0
    total_elapsed = 0.0

    for _ in range(repetitions):
        synchronize_device()
        start = time.perf_counter()
        results = engine.generate_text(prompts, max_tokens=requests[0]["max_tokens"])
        synchronize_device()
        elapsed = time.perf_counter() - start

        if not isinstance(results, list):
            results = [results]

        generated = sum(len(result.tokens) for result in results)
        latencies.append(elapsed)
        generated_tokens += generated
        total_tokens += prompt_tokens + generated
        total_elapsed += elapsed

    return {
        "latencies": latencies,
        "elapsed": total_elapsed,
        "generated_tokens": generated_tokens,
        "total_tokens": total_tokens,
    }


def run_benchmark(
    n_requests=10,
    repetitions=DEFAULT_REPETITIONS,
    max_tokens=DEFAULT_MAX_TOKENS,
    seed=DEFAULT_SEED,
    prompt_tokens=DEFAULT_PROMPT_TOKENS,
    verbose=False,
    compile_model=False,
):
    if n_requests < 1:
        raise ValueError("n_requests must be at least 1")
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    if prompt_tokens < 1:
        raise ValueError("prompt_tokens must be at least 1")

    set_seed(seed)
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB" if torch.cuda.is_available() else "N/A"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    requests = generate_requests(
        tokenizer=tokenizer,
        n_requests=n_requests,
        max_tokens=max_tokens,
        seed=seed,
        prompt_tokens=prompt_tokens,
    )

    print_request_set_summary(requests, repetitions, seed, device_name, vram, compile_model)

    engine = Engine(MODEL_NAME, compile_model=compile_model)
    single_result = benchmark_single_request(engine, requests, repetitions)
    cleanup_engine(engine)

    engine = Engine(MODEL_NAME, compile_model=compile_model)
    static_batch_result = benchmark_static_batch(engine, requests, repetitions)
    cleanup_engine(engine)

    benchmark_rows = [
        (
            "single request",
            single_result,
            "Latency is measured per request.",
        ),
        (
            "fixed batch",
            static_batch_result,
            "Latency is measured per run with a fixed-size batch of identical-shape prompts.",
        ),
    ]

    print_compact_summary(benchmark_rows)

    if verbose:
        print_verbose_summary(benchmark_rows)
        print_benchmark_summary(
            "Single Request",
            single_result["latencies"],
            single_result["elapsed"],
            single_result["generated_tokens"],
            single_result["total_tokens"],
            "per request",
            "Only one request is active at a time.",
        )
        print_benchmark_summary(
            "Fixed Batch",
            static_batch_result["latencies"],
            static_batch_result["elapsed"],
            static_batch_result["generated_tokens"],
            static_batch_result["total_tokens"],
            "per run (fixed-size batch)",
            "All requests share the same prompt shape and are submitted together.",
        )
    print()


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark hayate with a fixed-shape workload.")
    parser.add_argument("n_requests", nargs="?", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--prompt-tokens", type=int, default=DEFAULT_PROMPT_TOKENS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--verbose", action="store_true", help="Print detailed per-mode benchmark breakdowns.")
    parser.add_argument("--compile", action="store_true", dest="compile_model", help="Compile the model with torch.compile before benchmarking.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        n_requests=args.n_requests,
        repetitions=args.repetitions,
        max_tokens=args.max_tokens,
        prompt_tokens=args.prompt_tokens,
        seed=args.seed,
        verbose=args.verbose,
        compile_model=args.compile_model,
    )
