import time
import random
import string
import sys
import torch
from hayate.engine.engine import Engine


MODEL_NAME = "Qwen/Qwen3-4B"


def random_text(min_words=1, max_words=10):
    word_count = random.randint(min_words, max_words)
    words = [
        "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 10)))
        for _ in range(word_count)
    ]
    return " ".join(words)


def generate_requests(n_requests):
    requests = []
    for _ in range(n_requests):
        prompt = random_text(min_words=100, max_words=200)
        max_tokens = random.randint(100, 500)
        requests.append((prompt, max_tokens))
    return requests


def print_header(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def benchmark_sequential(engine, requests):
    print_header("Sequential (one request at a time)")

    total_tokens = 0
    total_time = 0.0

    for i, (prompt, max_tokens) in enumerate(requests):
        start = time.perf_counter()
        result = engine.generate_text(prompt, max_tokens=max_tokens)
        elapsed = time.perf_counter() - start

        actual_tokens = len(result.tokens)
        total_tokens += actual_tokens
        total_time += elapsed

        print(f"  req {i+1:3d} | {actual_tokens:4d} tokens | {elapsed:7.3f}s | {actual_tokens/elapsed:6.1f} tok/s")

    print(f"\n  total tokens: {total_tokens}")
    print(f"  total time:   {total_time:.3f}s")
    print(f"  throughput:   {total_tokens / total_time:.2f} tok/s")
    return total_tokens, total_time


def benchmark_batched(engine, requests):
    print_header("Batched (continuous batching)")

    prompts = [p for p, _ in requests]
    avg_max_tokens = int(sum(m for _, m in requests) / len(requests))

    start = time.perf_counter()
    results = engine.generate_text(prompts, max_tokens=avg_max_tokens)
    elapsed = time.perf_counter() - start

    if not isinstance(results, list):
        results = [results]
    total_tokens = sum(len(r.tokens) for r in results)

    print(f"  batch size:   {len(prompts)}")
    print(f"  avg max_tokens: {avg_max_tokens}")
    print(f"  total tokens: {total_tokens}")
    print(f"  total time:   {elapsed:.3f}s")
    print(f"  throughput:   {total_tokens / elapsed:.2f} tok/s")
    return total_tokens, elapsed


def run_benchmark(n_requests=10):
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB" if torch.cuda.is_available() else "N/A"

    print(f"\n  hayate benchmark")
    print(f"  model:  {MODEL_NAME}")
    print(f"  device: {device_name} ({vram})")
    print(f"  requests: {n_requests}")

    requests = generate_requests(n_requests)

    engine = Engine(MODEL_NAME)
    seq_tokens, seq_time = benchmark_sequential(engine, requests)

    del engine
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    engine = Engine(MODEL_NAME)
    batch_tokens, batch_time = benchmark_batched(engine, requests)

    print_header("Summary")
    print(f"  {'method':<35} {'tokens':>8} {'time':>10} {'tok/s':>10}")
    print(f"  {'-'*35} {'-'*8} {'-'*10} {'-'*10}")
    print(f"  {'sequential':<35} {seq_tokens:>8} {seq_time:>9.3f}s {seq_tokens/seq_time:>9.2f}")
    print(f"  {'continuous batching':<35} {batch_tokens:>8} {batch_time:>9.3f}s {batch_tokens/batch_time:>9.2f}")
    print()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_benchmark(n)
