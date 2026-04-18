# hayate

Inference engine for `Qwen/Qwen3-4B`. WIP.

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

benchmarks on an `NVIDIA A40 (47.7GB)`:

```text
============================================================
  Workload
============================================================

  model:                Qwen/Qwen3-4B
  device:               NVIDIA A40 (47.7GB)
  seed:                 1337
  requests:             10
  repetitions:          5
  arrival gap (ms):     25.0
  prompt tokens (min):  175
  prompt tokens (mean): 225.0
  prompt tokens (max):  322
  max_tokens/request:   256

============================================================
  Benchmark Summary
============================================================

  mode                       mean        p50        p95  total tok/s
  -------------------- ---------- ---------- ---------- ------------
  single request           8.237s     8.218s     8.540s       58.39
  continous batching      17.361s    17.353s    17.596s      273.31
```

## Todo

- [x] model architecture
- [x] kv caching
- [x] greedy decoding
- [x] continuous batching
- [x] torch.compile
- [ ] paged attention
- [ ] turboquant kv cache
- [ ] custom kernels
