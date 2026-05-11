# hayate

Inference engine for Qwen3-4B. WIP.

## Setup

`hayate` requires Python `3.12+`.

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Run

Run the benchmark with the default non-verbose output:

```bash
python benchmark.py 10
```

Use `--verbose` if you want the detailed per-mode breakdown:

```bash
python benchmark.py 10 --verbose
```

Pass `--compile` to wrap the model in `torch.compile(..., dynamic=True)`. The
default compile mode avoids CUDA Graph private pools, which keeps memory use lower
on 24GB GPUs.

```bash
python benchmark.py 10 --compile
```

You can still opt into more aggressive compile modes for larger GPUs:

```bash
python benchmark.py 10 --compile --compile-mode reduce-overhead
```

`reduce-overhead` can be faster, but it captures CUDA graphs. Decode changes cache
length every step, so graph private pools can accumulate and trigger CUDA OOM.

PyTorch SDPA backend selection is exposed through `Engine` and the benchmark. The
default `auto` mode lets PyTorch pick the best available kernel; use `flash`,
`efficient`, or `math` to force a backend for GPU experiments:

```bash
python benchmark.py 10 --sdpa-backend efficient
```

Forced fused backends can fail if the active PyTorch/CUDA build does not support the
current mask, dtype, GQA, or sequence shape. Use `auto` for the safest dispatch path.

Prefix caching is available as an opt-in engine feature for workloads where prompts
share token prefixes:

```python
from hayate.engine.engine import Engine

engine = Engine("Qwen/Qwen3-4B", enable_prefix_cache=True)
```

The cache stores reusable prompt KV prefixes with a token budget. The default budget
is 4096 cached prefix tokens; pass `prefix_cache_max_tokens=...` to tune VRAM usage.

To benchmark prefix caching, run the benchmark with a generated shared-prefix
workload:

```bash
python benchmark.py 10 --prefix-cache
```

Tune the synthetic workload with `--prefix-shared-tokens`,
`--prefix-suffix-tokens`, and `--prefix-cache-max-tokens`.

## Benchmark

`Qwen/Qwen3-4B` on an `NVIDIA A40 (47.7GB)`, 10 requests, 5 reps.

without `--compile`:

```text
mode                       mean        p50        p95  total tok/s
-------------------- ---------- ---------- ---------- ------------
single request           9.872s     9.801s    10.254s       48.73
submit all upfront      12.925s    12.875s    13.072s      372.15
staggered arrivals      13.105s    13.080s    13.508s      361.49
```

with `--compile`:

```text
mode                       mean        p50        p95  total tok/s
-------------------- ---------- ---------- ---------- ------------
single request           4.382s     4.373s     4.571s      109.78
submit all upfront       9.459s     9.396s     9.727s      508.51
staggered arrivals       9.704s     9.627s    10.230s      485.16
```

## Todo

- [x] model architecture
- [x] kv caching
- [x] greedy decoding
- [x] continuous batching
- [x] torch.compile
- [x] prefix caching
- [x] pytorch fused sdpa
- [ ] paged attention