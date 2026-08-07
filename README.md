# hayate

Learning-focused inference engine for `Qwen/Qwen3-4B` on NVIDIA GPUs.

Features:

- BF16 inference with Flash Attention
- KV caching and continuous batching
- Chucked prefill and greedy decoding
- Variable-length prompt batches
- Supports `torch.compile` and prefix caching

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

Enable `torch.compile` with default mode:

```python
engine = Engine("Qwen/Qwen3-4B", compile=True)
```

Enable prefix caching:

```python
engine = Engine(
    "Qwen/Qwen3-4B",
    enable_prefix_cache=True,
    prefix_cache_max_tokens=4096,
)
```

## Benchmark

```bash
# Default: batch 1, 512 prompt tokens, 64 decode steps
python benchmark.py

# Custom workload
python benchmark.py --batch-size 4 --context-tokens 2048 --decode-steps 128

# Save the same result as JSON
python benchmark.py --json benchmark-results.json
```

The benchmark measures prefill latency and throughput, decode step latency and
throughput, and peak VRAM for one fixed workload. Run `python benchmark.py --help`
for controls.
