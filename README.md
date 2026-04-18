# hayate

Inference engine for `Qwen/Qwen3-4B`. WIP.

## Setup

`hayate` requires Python `3.12+`.

```bash
uv venv
source .venv/bin/activate
uv pip install -U pip
uv pip install -e .
```

## Run

Run the benchmark:

```bash
python benchmark.py 10
```

Compile the model before benchmarking:

```bash
python benchmark.py 10 --compile
```

Use `--verbose` if you want the detailed per-mode breakdown:

```bash
python benchmark.py 10 --verbose
```

`torch.compile` has a noticeable upfront cost on the first run, but later generations can be faster.

## Benchmark

```bash
python benchmark.py 10
python benchmark.py 10 --compile
python benchmark.py 10 --prompt-tokens 224 --max-tokens 256 --verbose
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
