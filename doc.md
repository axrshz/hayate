# Hayate Engine Guide

This document explains how this project is put together and how a prompt turns
into generated tokens. It is written as a learning guide, so it focuses on the
main design decisions, the data flow, and the tensor shapes that matter.

Hayate is a small inference engine for `Qwen/Qwen3-4B`. It implements:

- A Qwen3-style transformer in PyTorch.
- Manual Hugging Face safetensor weight loading.
- Greedy decoding.
- KV caching.
- Continuous batching.
- Optional prefix caching.
- PyTorch SDPA backend selection.
- Optional `torch.compile`.
- A benchmark harness for single-request, all-at-once batching, staggered arrivals,
  and prefix-cache workloads.

The project is intentionally compact. That makes it a good place to learn the
moving parts of LLM serving without first needing to understand a large production
engine.

## Repository Map

```text
hayate/
  engine/
    engine.py          Request objects, scheduler, prefill/decode, cache movement.
    prefix_cache.py    LRU token-prefix cache for reusable prompt KV tensors.
  model/
    qwen.py            Qwen3Model wrapper: embeddings, blocks, masks, logits.
    block.py           Transformer block: attention + MLP with residuals.
    attention.py       Grouped-query attention using PyTorch SDPA.
    linear.py          Gated feed-forward network.
    rope.py            Rotary embedding table construction and application.
    cache.py           Per-request KV cache container.
  utils.py             Weight download, validation, and safetensor loading.

benchmark.py           Benchmark runner.
main.py                Tiny manual usage example.
README.md              Setup, run commands, current benchmark numbers.
doc.md                 This guide.
```

## Big Picture

At a high level, generation has two phases:

1. Prefill:
   The full prompt is processed. This creates the initial KV cache and produces
   logits for the first generated token.

2. Decode:
   One generated token is processed per step. The engine reuses the KV cache so it
   does not recompute the whole prompt and previous output tokens.

For a single request, the flow looks like this:

```text
prompt string
  -> tokenizer.encode(...)
  -> Request(prompt_tokens=[...])
  -> prefill full prompt or uncached suffix
  -> sample first token
  -> decode one token at a time
  -> stop on EOS or max_tokens
  -> tokenizer.decode(generated_tokens)
```

For multiple requests, the engine keeps a queue and a current batch. Each call to
`generate()` admits queued work, prefills new requests, and decodes active requests.

## The Model

The model implementation lives in `hayate/model/`.

### Hardcoded Qwen3-4B Config

`Qwen3Model` in `hayate/model/qwen.py` hardcodes the Qwen3-4B architecture:

```text
vocab_size              151936
hidden_size             2560
num_layers              36
num_attention_heads     32
num_key_value_heads      8
head_dim                 128
intermediate_size       9728
rms_norm_eps            1e-6
rope_theta              1000000
max_position_embeddings 40960
dtype                   bfloat16
```

This means the engine is specialized for `Qwen/Qwen3-4B`. It is not a generic
Hugging Face model loader yet.

The model is built from:

- `embed_tokens`: token id -> hidden vector.
- `layers`: 36 transformer blocks.
- `norm`: final RMSNorm.
- `out_head`: hidden vector -> vocabulary logits.
- `cos` and `sin` buffers: precomputed RoPE tables.

### Transformer Block

`hayate/model/block.py` defines one transformer block:

```text
x
  -> input RMSNorm
  -> grouped-query self-attention
  -> residual add
  -> post-attention RMSNorm
  -> gated MLP
  -> residual add
```

This matches the standard decoder-only transformer block shape used by Qwen.

### Feed Forward Network

`hayate/model/linear.py` implements the gated MLP:

```text
silu(gate_proj(x)) * up_proj(x)
  -> down_proj(...)
```

This is the SwiGLU-style feed-forward block common in modern LLMs.

### Grouped Query Attention

`hayate/model/attention.py` implements grouped-query attention.

Qwen3-4B uses:

```text
num query heads     32
num KV heads         8
query groups         4 query heads per KV head
```

The projections produce:

```text
q: (B, T, 32, 128)
k: (B, T,  8, 128)
v: (B, T,  8, 128)
```

Then they are transposed for attention:

```text
q: (B, 32, T, D)
k: (B,  8, T, D)
v: (B,  8, T, D)
```

The code applies QK norm, then RoPE to `q` and `k`, then concatenates any previous
KV cache:

```text
full_k = cat([prev_k, k], dim=2)
full_v = cat([prev_v, v], dim=2)
```

Attention is then computed with:

```python
torch.nn.functional.scaled_dot_product_attention(
    q,
    full_k,
    full_v,
    attn_mask=attn_mask,
    dropout_p=0.0,
    is_causal=False,
    enable_gqa=True,
)
```

`enable_gqa=True` tells PyTorch to handle the mismatch between 32 query heads and
8 KV heads.

### SDPA Backend Modes

The project exposes four SDPA modes:

```text
auto
flash
efficient
math
```

They are selected through `--sdpa-backend` in `benchmark.py` or `sdpa_backend=...`
when constructing `Engine`.

- `auto`: lets PyTorch pick the best available backend.
- `flash`: forces the FlashAttention SDPA backend.
- `efficient`: forces the memory-efficient attention backend.
- `math`: forces the plain math fallback.

For normal experiments, `auto` is the safest choice. Forced modes are useful when
you want to compare kernels, but they can fail if a GPU, dtype, shape, GQA setting,
or mask is unsupported.

### RoPE

`hayate/model/rope.py` builds RoPE tables:

```text
cos: (max_position_embeddings, head_dim)
sin: (max_position_embeddings, head_dim)
```

During attention, `apply_rope_vectorized()` receives:

```text
x:            (B, H, T, D)
position_ids: (B, T)
```

It gathers the right rows from `cos` and `sin` for each request and token position.
This matters because batched requests can have different cache lengths and different
left-padding amounts.

## Weight Loading

Weights are loaded by `hayate/utils.py`.

The loader does not call `AutoModelForCausalLM.from_pretrained`. Instead, it:

1. Downloads safetensors and tokenizer files if needed.
2. Validates sharded checkpoint completeness.
3. Builds a map from Hugging Face tensor names to local model parameters.
4. Opens each safetensor shard.
5. Copies matching tensors into the PyTorch modules.
6. Ties `lm_head` to embeddings if no explicit `lm_head.weight` exists.

The weight map includes names like:

```text
model.embed_tokens.weight
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.k_proj.weight
model.layers.0.self_attn.v_proj.weight
model.layers.0.self_attn.o_proj.weight
model.layers.0.self_attn.q_norm.weight
model.layers.0.self_attn.k_norm.weight
model.layers.0.input_layernorm.weight
model.layers.0.post_attention_layernorm.weight
model.layers.0.mlp.gate_proj.weight
model.layers.0.mlp.up_proj.weight
model.layers.0.mlp.down_proj.weight
model.norm.weight
lm_head.weight
```

Recent hardening in the loader is important for cloud runs. A partial Hugging Face
download can leave one safetensor shard on disk and make naive code think the model
is present. The loader now validates `model.safetensors.index.json` and shard names
such as `model-00001-of-00003.safetensors`, so missing or corrupt shards fail with
a clear error.

## Request Objects

The engine represents each generation job with `Request` in `hayate/engine/engine.py`.

Important fields:

```text
id                unique request id
prompt            original prompt string
max_tokens        generation limit
prompt_tokens     tokenized prompt
tokens            generated token ids only
kv_cache          per-request KV cache
cache_pos         number of valid cached positions
prefix_cache_len  number of prompt tokens reused from prefix cache
is_completed      whether generation is done
is_prefill        whether this request still needs prefill
response          decoded generated text
```

Notice that `tokens` contains generated tokens, not prompt tokens. Prompt tokens
live separately in `prompt_tokens`.

## KV Cache Layout

`hayate/model/cache.py` defines a `Cache` object:

```text
k: (num_layers, H_kv, L, D)
v: (num_layers, H_kv, L, D)
```

For Qwen3-4B:

```text
num_layers = 36
H_kv       = 8
D          = 128
L          = current cached sequence length
```

Each request owns a compact per-request cache. The engine temporarily stacks those
per-request caches into a batched cache before a model forward pass.

## Engine Initialization

Creating an engine does this:

```python
engine = Engine(
    "Qwen/Qwen3-4B",
    compile=False,
    compile_mode="default",
    enable_prefix_cache=False,
    prefix_cache_max_tokens=4096,
    sdpa_backend="auto",
)
```

Step by step:

1. Validate `sdpa_backend`.
2. Validate `compile_mode`.
3. Construct `Qwen3Model`.
4. Load checkpoint weights into the model.
5. Move the model to `cuda` if available, otherwise CPU.
6. Optionally wrap the model with `torch.compile`.
7. Load the tokenizer.
8. Load all configured EOS tokens from `generation_config.json`.
9. Create the request queue and current batch list.
10. Optionally create the prefix cache.

### Compile Modes

`compile=True` enables `torch.compile`.

The default compile mode is:

```python
torch.compile(model, dynamic=True)
```

`dynamic=True` is important because batch size, prompt length, and cache length can
change between forward passes.

The project also allows:

```text
default
reduce-overhead
max-autotune
max-autotune-no-cudagraphs
```

`reduce-overhead` can be fast, but it uses CUDA Graph private pools. Decode changes
cache length every step, so on 24GB GPUs that mode can accumulate graph memory and
OOM. The default mode is more memory-friendly.

## Adding a Request

The public API is:

```python
result = engine.generate_text("Explain AGI", max_tokens=100)
```

or:

```python
results = engine.generate_text(
    ["Explain AGI", "What is vLLM?", "Tell me about SGLang"],
    max_tokens=100,
)
```

Internally `generate_text()`:

1. Converts a single prompt into a one-item list.
2. Creates a `Request` for each prompt.
3. Assigns an id.
4. Calls `add_request()`.
5. Repeatedly calls `generate()` until no work remains.
6. Returns one `Request` for a single input or a list for multiple inputs.

`add_request()` calls `_prepare_request()` and pushes the request into `self.pool`.

## Request Preparation

`_prepare_request()` does several important things:

1. Tokenizes the prompt if needed.
2. Rejects unsupported `use_cache=False`.
3. Validates `max_tokens` and context length.
4. Creates an empty `Cache`.
5. Looks for a reusable prefix if prefix caching is enabled.

The context-length check is:

```text
len(prompt_tokens) + max_tokens - 1 <= max_position_embeddings
```

The `- 1` exists because if you request `max_tokens` generated tokens, the prompt
positions plus the first `max_tokens - 1` generated positions are what need RoPE
positions during forward passes.

## Continuous Batching Scheduler

The scheduler state is:

```text
self.pool           queue of waiting requests
self.current_batch  active requests
```

`_get_next_batch()`:

1. Drops completed requests from `current_batch`.
2. Pulls queued requests into `current_batch` until `MAX_BATCH_SIZE`.
3. Returns the current active batch.

`generate()` then splits active requests into:

```text
prefill_requests: requests where is_prefill is True
decode_requests:  requests where is_prefill is False
```

Then:

1. Prefill requests are processed in chunks of `MAX_PREFILL_BATCH`.
2. Decode requests are processed together in one decode step.
3. `generate()` returns `True` while work remains.

This is the core continuous batching loop. New requests can join the batch while
older requests are still decoding.

## Prefill Step

`prefill_batch()` handles first-pass prompt processing.

For each request:

1. Ensure prompt tokens exist.
2. Ensure prompt is non-empty.
3. If a prefix cache gave too much cache, slice it back so at least one prompt token
   remains to run through the model.
4. Build the suffix tokens that still need prefill:

```text
suffix = prompt_tokens[cache_pos:]
```

If no prefix cache was used, `cache_pos` is 0 and suffix is the whole prompt.

If a 2049-token prefix was reused and the full prompt is 2085 tokens, suffix is the
remaining 36 tokens.

### Left Padding During Prefill

Requests in a prefill batch can have different suffix lengths. The engine left-pads
them to the same length:

```text
short suffix: [pad, pad, real, real]
long suffix:  [real, real, real, real]
```

It also records:

```text
pad_lengths_py = [number_of_left_pad_tokens_per_request]
```

The model uses `pad_lengths` to:

- Compute correct RoPE positions.
- Mask out pad tokens in attention.
- Avoid adding pad tokens to the compact per-request cache.

After the model forward pass, the engine samples one token from the last logits and
marks the request as no longer needing prefill.

## Decode Step

`decode_batch()` handles one generated token per active request.

It builds:

```text
tokens = [[last_generated_token_for_req_0],
          [last_generated_token_for_req_1],
          ...]
```

Shape:

```text
(B, 1)
```

Then it runs `_forward_pass()`, samples the next token, appends it to each request,
and increments `cache_pos`.

Decode repeats until each request hits either:

- One of the configured EOS tokens.
- `max_tokens`.

The engine uses every EOS token from the model's generation config, not just
`tokenizer.eos_token_id`. For Qwen3-4B this matters because generation config can
contain more than one stop token.

## Batched Cache Gather

The model wants a batched cache:

```text
prev_k: (num_layers, B, H_kv, L_max, D)
prev_v: (num_layers, B, H_kv, L_max, D)
```

But each request stores a compact cache:

```text
request.kv_cache.k: (num_layers, H_kv, L_i, D)
```

`_gather_caches()`:

1. Reads each request's cache length.
2. Finds `L_max`.
3. Allocates zero-padded batched tensors.
4. Copies each compact cache into its row.
5. Creates `cache_lens`, a tensor of real per-request cache lengths.

The zeros are padding. The model attention mask makes sure no request can attend
to padded cache columns.

## Model Forward Pass

`_forward_pass()` is the bridge between engine and model:

1. Gather request caches into batched `prev_k` and `prev_v`.
2. Convert Python pad lengths to a tensor if needed.
3. Run the model under `torch.no_grad()`.
4. Scatter updated caches back to per-request storage.
5. Return only the last-token logits:

```python
return logits[:, -1, :]
```

The model itself returns logits for all input positions, but generation only needs
the final position in the current forward pass.

## Attention Mask Construction

The model builds one additive mask with shape:

```text
attn_mask: (B, 1, T, L_full)
```

where:

```text
T      = current input length
L_prev = previous cache length after batching
L_full = L_prev + T
```

The mask combines three ideas:

1. Causal masking:
   A query can only attend to previous keys and itself.

2. Cache padding masking:
   If request A has 100 cached tokens and request B has 90, the batched cache uses
   `L_max = 100`. Request B must not attend to columns 90 through 99.

3. New-token left-padding masking:
   During batched prefill, shorter suffixes are left-padded. Real tokens must not
   attend to those pad tokens.

The final boolean mask is converted to an additive floating-point mask:

```text
False -> 0
True  -> -inf
```

This is passed into SDPA.

## Batched Cache Scatter

The model returns updated cache tensors:

```text
new_k: (num_layers, B, H_kv, L_prev + T, D)
new_v: (num_layers, B, H_kv, L_prev + T, D)
```

`_scatter_caches()` turns that back into compact per-request caches.

For each request, it keeps:

```text
old valid cache columns: [0, old_L)
new real token columns:  [L_prev + pad_len, L_prev + T)
```

Then it concatenates them:

```text
compact_cache = cat([old_cache, new_real_tokens], dim=sequence)
```

This removes both cache-region padding and prefill left-padding.

## Prefix Cache

Prefix caching lives in `hayate/engine/prefix_cache.py`.

The idea:

If many prompts begin with the same token prefix, the KV tensors for that prefix can
be reused.

Example:

```text
request 1: [A, B, C, D, X, Y]
request 2: [A, B, C, D, P, Q]
```

After request 1 prefills, the cache can store KV tensors for `[A, B, C, D, X, Y]`.
When request 2 arrives, the prefix cache can find the common prefix `[A, B, C, D]`
and reuse the first 4 positions.

### Prefix Cache Entries

The cache maps:

```text
tuple(prompt_token_prefix) -> Cache
```

It uses an `OrderedDict` as an LRU:

- Reads move an entry to the end.
- New writes go to the end.
- Old entries are evicted from the front.

The budget is measured in cached prefix tokens, not bytes:

```text
prefix_cache_max_tokens = 4096
```

### Prefix Cache Lookup

`get(tokens, max_prefix_len)` scans entries and finds the longest common prefix up
to `max_prefix_len`.

The engine sets:

```text
max_prefix_len = len(prompt_tokens) - 1
```

It intentionally leaves at least one prompt token to prefill, because a cached full
prompt by itself does not produce the next-token logits.

### Prefix Cache Storage

After prefill, `_store_prompt_prefix()` stores the request's prompt cache if:

- Prefix caching is enabled.
- The request uses cache.
- A KV cache exists.
- The KV cache covers the full prompt.

The stored cache may be sliced to the configured token budget.

### What Prefix Caching Speeds Up

Prefix caching saves prefill compute. It does not make decode steps disappear.

That means speedup depends on workload shape:

- Large shared prefix, short generation: big speedup.
- Small shared prefix, long generation: small speedup.

Your observed results show exactly that:

```text
shared prefix about 259 tokens, max_tokens 256 -> 1.01x speedup
shared prefix about 2049 tokens, max_tokens 32 -> 1.52x speedup
```

In the second case, cached prompt tokens were `100401`, which matches:

```text
49 cache hits * 2049 shared tokens = 100401
```

That is a strong sanity check that the prefix cache is working.

## Sampling

Sampling is greedy:

```python
torch.argmax(logits, dim=-1, keepdim=True)
```

There is no temperature, top-k, top-p, repetition penalty, or beam search yet.

The response is decoded from generated tokens only:

```python
tokenizer.decode(request.tokens, skip_special_tokens=True)
```

The prompt is not included in `response`.

## Benchmark Script

`benchmark.py` is the performance harness.

Default command:

```bash
python benchmark.py 10
```

Default settings:

```text
n_requests            10
repetitions            5
max_tokens           256
arrival_gap_ms        25.0
prompt token range   160 to 320
seed                 1337
sdpa backend         auto
compile              off
prefix cache         off
```

The benchmark creates synthetic prompts from random lowercase fragments. It uses
the real tokenizer to hit target token counts.

### Benchmark Mode 1: Single Request

`benchmark_single_request()` runs requests one at a time:

```text
for each repetition:
  for each request:
    generate_text(one_prompt)
```

Latency unit:

```text
one request
```

This mode measures single-request behavior and small-batch decode overhead.

### Benchmark Mode 2: Submit All Upfront

`benchmark_static_batch()` submits all prompts together:

```text
generate_text([prompt_0, prompt_1, ..., prompt_n])
```

Latency unit:

```text
one full run containing all requests
```

This mode shows the benefit of batching prefill and decode.

### Benchmark Mode 3: Staggered Arrivals

`benchmark_continuous_batch()` simulates online traffic.

`run_continuous_batch_once()`:

1. Starts a wall-clock timer.
2. Submits request `i` at `i * arrival_gap_ms`.
3. Calls `engine.generate()` repeatedly.
4. Records each request's arrival-to-completion latency.

Latency unit:

```text
one request, measured from submitted_at to completed_at
```

This mode is closer to serving behavior because requests enter while other requests
are already decoding.

### Prefix Cache Benchmark

Run:

```bash
python benchmark.py 10 --prefix-cache
```

This does the normal three benchmark modes first. Then it creates a separate
shared-prefix workload:

```text
shared prefix + unique suffix
shared prefix + unique suffix
...
```

It compares two engines:

```text
prefix cache off
prefix cache on
```

The summary reports:

- Mean, p50, p95 latency.
- Total tokens per second.
- Cached prompt tokens.
- Elapsed speedup.

To make prefix caching visible, use a large shared prefix and smaller generation:

```bash
python benchmark.py 10 --prefix-cache \
  --prefix-shared-tokens 2048 \
  --prefix-suffix-tokens 32 \
  --max-tokens 32
```

### Compile Benchmarking

Run:

```bash
python benchmark.py 10 --compile
```

This wraps the model with:

```python
torch.compile(model, dynamic=True)
```

You can choose another compile mode:

```bash
python benchmark.py 10 --compile --compile-mode reduce-overhead
```

On 24GB GPUs, prefer the default compile mode. `reduce-overhead` can OOM because
CUDA Graph private pools can accumulate as decode cache length changes.

### SDPA Benchmarking

Run:

```bash
python benchmark.py 10 --sdpa-backend auto
python benchmark.py 10 --sdpa-backend flash
python benchmark.py 10 --sdpa-backend efficient
python benchmark.py 10 --sdpa-backend math
```

Use `auto` for normal runs. Use the others for experiments.

## Reading Benchmark Output

The compact table prints:

```text
mode                       mean        p50        p95  total tok/s
-------------------- ---------- ---------- ---------- ------------
single request           ...
submit all upfront       ...
staggered arrivals       ...
```

For `single request`, `mean` is mean per-request latency.

For `submit all upfront`, `mean` is mean full-batch run latency, not per-request
latency.

For `staggered arrivals`, `mean` is mean request latency from arrival to completion.

`total tok/s` counts:

```text
prompt tokens + generated tokens
```

This is useful for comparing overall engine throughput, especially when prompt
lengths vary.

## Current Performance Interpretation

From your newer SDPA benchmark on an RTX 3090-class 24GB card:

```text
without compile
single request        9.872s   48.73 tok/s
submit all upfront   12.925s  372.15 tok/s
staggered arrivals   13.105s  361.49 tok/s

with compile
single request        4.382s  109.78 tok/s
submit all upfront    9.459s  508.51 tok/s
staggered arrivals    9.704s  485.16 tok/s
```

Compile speedups:

```text
single request       about 2.25x
submit all upfront   about 1.37x
staggered arrivals   about 1.35x
```

This shape makes sense. Single-request decode is overhead-heavy, so compile helps a
lot. Batched modes already use the GPU better, so compile still helps but less
dramatically.

Compared with the old pre-SDPA benchmark, SDPA mostly improved batched throughput.
Single-request without compile can get slower because SDPA dispatch and mask
handling can cost more for very small decode shapes. With compile, that overhead is
reduced and single-request improves too.

## Key Design Tradeoffs

### Simplicity Over Generality

The model config is hardcoded for Qwen3-4B. This keeps the code readable, but it
means adding another model requires code changes.

### Compact Per-Request Caches

Each request stores a compact cache. The engine pads and stacks caches only for a
forward pass. This keeps request state simple, but gather/scatter copies cost time.
Production engines often use paged KV caches to avoid much of this copying.

### Full Logits During Prefill

The model currently computes vocabulary logits for every input position, even
though the engine only uses the last one. This is simple but wasteful for long
prefills. A useful optimization would be to project only the final position during
generation.

### Greedy Only

Greedy decoding is easy to reason about and benchmark. Adding sampling controls
would make the engine more useful but also add more state and options.

### Prefix Cache Token Budget

The prefix cache budget is measured in tokens, not bytes. This is easy to understand
but not perfect. Real memory depends on layers, KV heads, head dimension, dtype, and
stored cache length.

## Common Failure Modes

### Incomplete Checkpoint Download

Symptom:

```text
Missing required checkpoint tensors
```

or a safetensor read error.

Cause:

An interrupted Hugging Face download left only some shards on disk.

Fix:

Delete the local `Qwen3-4B/` directory and rerun, or download manually:

```bash
huggingface-cli download Qwen/Qwen3-4B --local-dir Qwen3-4B
```

### CUDA OOM With Compile

If using:

```bash
--compile --compile-mode reduce-overhead
```

try plain:

```bash
--compile
```

If fragmentation remains an issue:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Forced SDPA Backend Fails

If `flash` or `efficient` fails, use:

```bash
--sdpa-backend auto
```

PyTorch's automatic dispatch is usually the safest option.

## Good Next Learning Projects

1. Add tests with a tiny toy model shape.
   Test cache growth, mask behavior, EOS stopping, and prefix cache equivalence.

2. Optimize prefill logits.
   Avoid computing full vocab logits for every prefill token.

3. Add sampling.
   Implement temperature, top-k, and top-p.

4. Add paged KV cache.
   Replace compact per-request cache copying with page/block allocation.

5. Add streaming output.
   Return tokens as they are generated instead of only at completion.

6. Add model config loading.
   Read architecture fields from `AutoConfig` instead of hardcoding Qwen3-4B.

7. Add memory reporting.
   Print allocated, reserved, and peak CUDA memory around each benchmark mode.

## Mental Model Summary

The shortest version:

```text
Engine owns requests.
Requests own compact KV caches.
generate() batches active requests.
prefill_batch() processes prompt suffixes.
decode_batch() processes one generated token per request.
Qwen3Model builds masks and runs transformer blocks.
Attention concatenates old KV cache with new K/V.
Engine scatters the updated batched cache back into each request.
Prefix cache reuses prompt KV tensors across requests with common token prefixes.
Benchmark.py measures the same engine under different traffic patterns.
```

Once that loop clicks, the rest of the project becomes much easier to read.
