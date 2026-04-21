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

Pass `--compile` to wrap the model in `torch.compile(..., dynamic=True)` so the
compiled graph handles varying batch size and cache length without re-specializing.
Expect a multi-minute warmup on the first prefill and the first decode; subsequent
calls are fast.

```bash
python benchmark.py 10 --compile
```

## Benchmark

`Qwen/Qwen3-4B` on an `NVIDIA A40 (47.7GB)`, 10 requests, 5 reps.

without `--compile`:

```text
mode                       mean        p50        p95  total tok/s
-------------------- ---------- ---------- ---------- ------------
single request           7.356s     7.326s     7.453s       65.39
submit all upfront      17.405s    17.405s    17.411s      276.36
staggered arrivals      17.335s    17.236s    17.906s      273.78
```

with `--compile`:

```text
mode                       mean        p50        p95  total tok/s
-------------------- ---------- ---------- ---------- ------------
single request           5.337s     5.312s     5.558s       90.13
submit all upfront      12.912s    12.907s    12.925s      372.52
staggered arrivals      13.003s    12.882s    13.639s      364.56
```

## Todo

- [x] model architecture
- [x] kv caching
- [x] greedy decoding
- [x] continuous batching
- [x] torch.compile
- [ ] pytorch fused sdpa
- [ ] prefix caching
- [ ] paged attention
- [ ] turboquant kv cache
- [ ] custom kernels
