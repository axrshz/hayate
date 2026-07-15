# hayate

Learning-focused inference engine for `Qwen/Qwen3-4B` on NVIDIA GPUs.

Features:

- BF16 inference with Flash Attention
- KV caching and continuous batching
- Variable-length prompt batches
- Optional `torch.compile` and prefix caching

## Setup

Requires Python 3.12+, CUDA, and an NVIDIA GPU.

```bash
uv sync --locked
source .venv/bin/activate
```

## Inference

```python
from hayate.engine.engine import Engine

engine = Engine("Qwen/Qwen3-4B")
result = engine.generate_text("Explain artificial general intelligence")
print(result.response)
```

Enable compilation for repeated single-request or uniform-batch workloads:

```python
engine = Engine("Qwen/Qwen3-4B", compile=True)
```

Enable prefix caching when requests share prompt prefixes:

```python
engine = Engine(
    "Qwen/Qwen3-4B",
    enable_prefix_cache=True,
    prefix_cache_max_tokens=4096,
)
```

## Benchmark

```bash
# Prefill and decode
python benchmark.py --suite micro

# Saturated throughput
python benchmark.py --suite offline

# Poisson-arrival serving load
python benchmark.py --suite online

# All suites with JSON output
python benchmark.py --suite all --json benchmark-results.json
```

The benchmark uses fixed input/output lengths and reports prompt throughput,
output throughput, TTFT, TPOT, goodput, and peak VRAM. Run
`python benchmark.py --help` for workload and batch controls.

## Limitations

- Architecture and weights are specific to Qwen3-4B.
- Sampling is greedy only.
- Paged attention is not implemented.
- Variable-length attention requires PyTorch 2.10.
