# hayate

Learning-focused inference engine for `Qwen/Qwen3-4B` on NVIDIA GPUs.

Features:

- BF16 inference with Flash Attention
- KV caching and continuous batching
- Chunked prefill and greedy decoding
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

Sampling is greedy by default (`temperature=0`). Set a positive temperature to
sample from the token distribution, and optionally restrict it with top-k and
top-p filtering:

```python
result = engine.generate_text(
    "Explain artificial general intelligence",
    max_tokens=100,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
)
```

These options also apply to every prompt in a batched `generate_text` call.

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

Long prompts are processed in chunks of 512 tokens by default. This lets active
requests decode between prompt chunks. Set `prefill_chunk_size` when creating
the engine to tune the number of prompt tokens processed per request per tick:

```python
engine = Engine("Qwen/Qwen3-4B", prefill_chunk_size=256)
```

The engine stores request KV states in a shared paged pool. Its default capacity
is 16,384 total token positions across active requests, with 128-token pages. Set
`max_cache_tokens` and `kv_page_size` when creating the engine to tune this pool.
Page sizes below 128 tokens are rejected because FlexAttention reuses the page
size as its block size and only compiles blocks its kernel tiles divide.
The configured capacity includes prompt and cached generated-token positions;
requests that exceed the available pool raise `MemoryError`.

## HTTP stream

The HTTP interface sends text as Server-Sent Events (SSE). Install the API
packages and start the server:

```bash
uv sync --locked --extra api
python -m hayate.api --model Qwen/Qwen3-4B --host 127.0.0.1 --port 8000
```

Send a `POST` request to `/generate` with a prompt:

```bash
curl -N http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain artificial intelligence","max_tokens":100}'
```

The server sends a `started` event, then a `token` event after each new token.
Each `token` event includes the token ID, token count, and full response text.
The server sends a `completed` event when generation ends. Each `started` event
includes the request ID.

Send a `POST` request to `/requests/{request_id}/cancel` to stop that request.
The engine stops the request after that model step. The server also stops a
request when its client disconnects. Use `/docs` to view the API.

## Benchmark

```bash
# Run one request with 512 prompt tokens and 64 output tokens.
python benchmark.py

# Set a larger batch, prompt, and output length.
python benchmark.py --batch-size 4 --context-tokens 2048 --decode-steps 128

# Save the measurements and settings in a JSON file.
python benchmark.py --json benchmark-results.json
```

The benchmark measures prefill latency and throughput, decode step latency and
throughput, and peak VRAM for one fixed workload. Run `python benchmark.py --help`
for controls.

### Measured results

These results use an NVIDIA RTX 4090 and `Qwen/Qwen3-4B`. Each request uses 512
prompt tokens and 64 decode steps. Each value is the median of 3 repetitions.
The first table uses eager mode, with `--compile` disabled. Throughput is the
total for all requests in the batch. The eager runs completed through batch 20
without an out-of-memory (OOM) error.

| Batch | Prefill throughput | Decode throughput | Peak VRAM |
| ---: | ---: | ---: | ---: |
| 2 | 13.8k tok/s | 58.5 tok/s | 10.67 GB |
| 4 | 16.4k tok/s | 118.5 tok/s | 10.76 GB |
| 8 | 16.6k tok/s | 224.5 tok/s | 10.95 GB |
| 16 | 16.6k tok/s | 459.5 tok/s | 11.33 GB |
| 20 | 16.7k tok/s | 544.4 tok/s | 11.52 GB |

Decode throughput increased 9.3× when batch size increased 10×. This is 93% of
the linear gain. Prefill throughput was almost the same at batches 8, 16, and 20.

At batch 20, the `torch.compile` result was 12.7k prefill tok/s and 62.2 decode
tok/s. Peak VRAM was 17.04 GB. The eager result was 16.7k prefill tok/s, 544.4
decode tok/s, and 11.52 GB peak VRAM. The compile result was 0.76× the eager
prefill throughput and 0.11× the eager decode throughput. It used 48% more peak
VRAM.

These results show that decode throughput was near the linear rate as batch size
increased. Prefill throughput was almost the same from batch 8 to batch 20. In
this benchmark, `torch.compile` did not increase speed.
