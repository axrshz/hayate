# Engine Walkthrough

This guide explains `hayate/engine/engine.py` from top to bottom. The goal is to
make the engine understandable if you are still new to inference engines.

The engine has one main job:

```text
Take one or more prompts, run the model step by step, and return generated text.
```

To do that efficiently, it also manages:

- Request state.
- Tokenization.
- KV caches.
- Prefix cache reuse.
- Batching.
- Prefill.
- Decode.
- Stop conditions.

## The Short Version

The most important loop is:

```text
generate_text()
  -> create Request objects
  -> add each Request to a queue
  -> while generate() has work:
       add queued requests to current batch
       prefill new requests
       decode old requests one token
  -> return completed Request objects
```

Inside each model call:

```text
per-request compact KV caches
  -> gather into one padded batched KV cache
  -> run Qwen3Model
  -> scatter updated KV cache back into each request
```

That gather-model-scatter pattern is the heart of this engine.

## Imports

At the top:

```python
import torch
from typing import List, Union
from transformers import AutoTokenizer, GenerationConfig
from dataclasses import dataclass, field
from queue import Queue
```

The engine uses:

- `torch` for tensors, CUDA, argmax sampling, and optional `torch.compile`.
- `AutoTokenizer` to turn strings into token ids and token ids back into text.
- `GenerationConfig` to read the model's stop tokens.
- `dataclass` to define `Request` cleanly.
- `Queue` to hold waiting requests.

Then it imports local project pieces:

```python
from hayate.model import Qwen3Model
from hayate.utils import load_weights
from hayate.model.cache import Cache
from hayate.engine.prefix_cache import PrefixCache
from hayate.model.attention import SDPA_BACKENDS
```

These are:

- `Qwen3Model`: the actual neural network.
- `load_weights`: loads Hugging Face safetensors into the model.
- `Cache`: stores one request's KV cache.
- `PrefixCache`: stores reusable prompt-prefix KV caches.
- `SDPA_BACKENDS`: valid PyTorch attention backend names.

## Device Selection

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

If CUDA is available, the engine uses the GPU. Otherwise it falls back to CPU.

For this project, CPU is useful for syntax/import checks, but Qwen3-4B inference is
intended for GPU.

## Constants

```python
COMPILE_MODES = ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs")
MAX_BATCH_SIZE = 20
MAX_DECODE_BATCH = 20
MAX_PREFILL_BATCH = 8
DEFAULT_PREFIX_CACHE_MAX_TOKENS = 4096
```

Meaning:

- `COMPILE_MODES`: allowed `torch.compile` modes.
- `MAX_BATCH_SIZE`: maximum active requests at once.
- `MAX_DECODE_BATCH`: currently defined but not used directly.
- `MAX_PREFILL_BATCH`: prefill requests are processed in chunks of 8.
- `DEFAULT_PREFIX_CACHE_MAX_TOKENS`: default token budget for prefix cache.

The reason prefill has its own smaller batch size is that prefill can be much more
memory-heavy than decode. A prompt may contain hundreds or thousands of tokens, but
decode only processes one new token per request per step.

## The Request Dataclass

```python
@dataclass
class Request:
    id: int = 0
    prompt: str = ""
    max_tokens: int = 100

    prompt_tokens: List[int] = field(default_factory=list)
    tokens: List[int] = field(default_factory=list)
    kv_cache: Cache | None = None

    cache_pos: int = 0
    prefix_cache_len: int = 0
    is_completed: bool = False
    is_prefill: bool = True
    use_cache: bool = True

    response: str | None = None
```

A `Request` is one generation job.

Important fields:

- `prompt`: the user's input string.
- `prompt_tokens`: the tokenized prompt.
- `tokens`: generated tokens only. This does not include the prompt.
- `max_tokens`: maximum number of generated tokens.
- `kv_cache`: this request's KV cache.
- `cache_pos`: how many positions are currently cached.
- `prefix_cache_len`: how much prompt prefix was reused from prefix cache.
- `is_prefill`: whether this request still needs prompt prefill.
- `is_completed`: whether generation is done.
- `response`: decoded text from generated tokens.

Beginner mental model:

```text
Request = prompt + generated tokens + cache + status flags
```

## Engine Initialization

The `Engine` constructor is:

```python
class Engine:
    def __init__(
        self,
        model_name: str,
        compile: bool = False,
        compile_mode: str = "default",
        enable_prefix_cache: bool = False,
        prefix_cache_max_tokens: int = DEFAULT_PREFIX_CACHE_MAX_TOKENS,
        sdpa_backend: str = "auto",
    ):
```

You normally create it like:

```python
engine = Engine("Qwen/Qwen3-4B")
```

Or with extra options:

```python
engine = Engine(
    "Qwen/Qwen3-4B",
    compile=True,
    enable_prefix_cache=True,
    sdpa_backend="auto",
)
```

### Validate Options

First, the engine checks that the requested attention backend and compile mode are
valid:

```python
if sdpa_backend not in SDPA_BACKENDS:
    raise ValueError(...)
if compile_mode not in COMPILE_MODES:
    raise ValueError(...)
```

This fails early if the user passes a typo like `--sdpa-backend flahs`.

### Build the Model

```python
self.model = Qwen3Model(sdpa_backend=sdpa_backend)
```

This creates the neural network structure: embeddings, transformer blocks, final
norm, and output head.

Then the engine copies a few model constants:

```python
self.num_layers = self.model.num_layers
self.num_kv_groups = self.model.num_kv_groups
self.head_dim = self.model.head_dim
self.max_position_embeddings = self.model.max_position_embeddings
```

The engine needs these when it builds batched cache tensors and validates prompt
lengths.

### Load Weights

```python
load_weights(self.model, model_name)
```

This fills the empty PyTorch model with weights from `Qwen/Qwen3-4B`.

At this point, the model is still on the default device from construction. Then:

```python
self.model = self.model.to(device)
```

moves it to GPU if available.

### Optional Compile

```python
if compile:
    if compile_mode == "default":
        self.model = torch.compile(self.model, dynamic=True)
    else:
        self.model = torch.compile(self.model, dynamic=True, mode=compile_mode)
```

`torch.compile` asks PyTorch to optimize the model execution.

`dynamic=True` matters because this engine uses changing shapes:

- Batch size can change.
- Prompt length can change.
- Cache length grows during decode.

The default compile mode is safer for 24GB GPUs. `reduce-overhead` may be faster on
some systems, but it can use more CUDA graph memory.

### Load Tokenizer and Stop Tokens

```python
self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
self.stop_token_ids = self._load_stop_token_ids(model_name)
```

The tokenizer turns text into token ids.

Stop tokens are token ids that mean "generation should stop now". For Qwen3-4B,
there can be more than one. That is why the engine reads the generation config
instead of only using `tokenizer.eos_token_id`.

### Create Scheduler State

```python
self.pool: Queue = Queue()
self.current_batch: List[Request] = []
self.request_id = 0
```

These are the scheduler's state:

- `pool`: waiting requests.
- `current_batch`: active requests.
- `request_id`: increasing id counter.

### Optional Prefix Cache

```python
self.prefix_cache = PrefixCache(prefix_cache_max_tokens) if enable_prefix_cache else None
```

If enabled, the prefix cache can reuse KV tensors for prompts with shared prefixes.

## Loading Stop Tokens

```python
def _load_stop_token_ids(self, model_name: str) -> set[int]:
```

This method returns a set of token ids that should stop generation.

It tries to load:

```python
generation_config = GenerationConfig.from_pretrained(model_name)
eos_token_id = generation_config.eos_token_id
```

`eos_token_id` might be:

- A single integer.
- A list of integers.
- Missing.

So the method handles both forms.

Then it also adds:

```python
self.tokenizer.eos_token_id
```

The result is a `set[int]`, which makes this check fast later:

```python
tok in self.stop_token_ids
```

## Adding a Request

```python
def add_request(self, request: Request):
    self._prepare_request(request)
    self.pool.put(request)
```

This does two things:

1. Prepare the request.
2. Put it in the waiting queue.

The engine does not immediately run the model here. It only queues the request.

## Preparing a Request

```python
def _prepare_request(self, request: Request):
```

This method makes a request ready for scheduling.

### Step 1: Tokenize

```python
if not request.prompt_tokens:
    request.prompt_tokens = self.tokenizer.encode(request.prompt)
```

If the caller gave only text, the engine creates token ids.

Example:

```text
"Explain AGI"
  -> [849, 21435, 468...]  # example only
```

The exact ids depend on the tokenizer.

### Step 2: Require KV Cache

```python
if not request.use_cache:
    raise ValueError(...)
```

This engine requires KV caching. Without it, decode would only see the previous
token and would lose the prompt context. A true no-cache mode would need to replay
the whole prompt plus generated tokens every step.

### Step 3: Validate Lengths

```python
self._validate_request_lengths(request)
```

This checks:

- `max_tokens >= 1`
- prompt is not empty
- prompt plus generation fits in the model context window

### Step 4: Reuse an Existing Cache if Present

```python
if request.kv_cache is not None and request.kv_cache.length > 0:
    request.cache_pos = request.kv_cache.length
    request.prefix_cache_len = request.cache_pos
    return
```

Most normal requests do not arrive with a cache. This branch is for a request that
already has cached KV tensors attached.

### Step 5: Create a New Cache

```python
request.kv_cache = Cache()
```

Now the request has an empty KV cache container.

### Step 6: Try Prefix Cache

If prefix caching is disabled:

```python
if self.prefix_cache is None:
    return
```

If enabled, the engine asks:

```python
max_prefix_len = max(len(request.prompt_tokens) - 1, 0)
match = self.prefix_cache.get(request.prompt_tokens, max_prefix_len=max_prefix_len)
```

Why `len(prompt_tokens) - 1`?

Because even if the whole prompt is already cached, the engine still needs to run
at least one prompt token through the model to produce first-token logits.

If the prefix cache finds a match:

```python
request.kv_cache = match.cache
request.cache_pos = match.length
request.prefix_cache_len = match.length
```

Now prefill only needs to process the remaining prompt suffix.

## Validating Request Length

```python
required_positions = len(request.prompt_tokens) + request.max_tokens - 1
```

This is how many RoPE positions might be needed.

Example:

```text
prompt length = 100
max_tokens    = 20

required_positions = 100 + 20 - 1 = 119
```

Why minus one?

The first generated token is produced from the prompt prefill. To generate 20
tokens, the model needs positions for:

- All prompt tokens.
- The first 19 generated tokens as inputs to decode future tokens.

Then it checks:

```python
required_positions <= self.max_position_embeddings
```

For Qwen3-4B here, `max_position_embeddings` is `40960`.

## Clearing Prefix Cache

```python
def clear_prefix_cache(self):
    if self.prefix_cache is not None:
        self.prefix_cache.clear()
```

This removes all saved prefix KV tensors.

The benchmark uses this to avoid warmup requests polluting measurement.

## Storing Prompt Prefixes

```python
def _store_prompt_prefix(self, request: Request):
```

After prefill, this method may store the request's prompt KV cache in the prefix
cache.

It only stores if:

- Prefix caching is enabled.
- The request uses cache.
- The request has a KV cache.
- The cache is at least as long as the prompt.

Then:

```python
self.prefix_cache.put(request.prompt_tokens, request.kv_cache)
```

Later requests with the same beginning tokens can reuse it.

## Sampling

```python
def sample(self, logits):
    return torch.argmax(logits, dim=-1, keepdim=True)
```

This is greedy decoding.

It chooses the token with the largest logit.

Shape example:

```text
logits:      (B, vocab_size)
next_tokens: (B, 1)
```

There is no randomness here. Same model, same prompt, same output.

## Finalizing a Generated Token

```python
def _finalize_generated_token(self, request: Request, tok: int):
    request.tokens.append(tok)
    if tok in self.stop_token_ids or len(request.tokens) >= request.max_tokens:
        request.is_completed = True
        request.response = self.tokenizer.decode(request.tokens, skip_special_tokens=True)
```

This method is called every time the engine samples a token.

It:

1. Adds the token to the request's generated token list.
2. Checks stop conditions.
3. If done, decodes the generated tokens into text.

Stop conditions:

- The token is an EOS/stop token.
- The request has generated `max_tokens`.

Important detail:

```text
response = decoded generated tokens only
```

The prompt is not included in `response`.

## Why Gather Caches?

Each request stores its own compact cache:

```text
request 0 cache length = 100
request 1 cache length = 80
request 2 cache length = 120
```

The model wants one batched tensor. But tensors in a batch need the same shape. So
the engine pads shorter caches up to the longest cache length.

This is what `_gather_caches()` does.

## Gathering Caches

```python
def _gather_caches(self, requests: List[Request]):
```

It returns:

```text
prev_k:      (num_layers, B, H_kv, L_max, D)
prev_v:      (num_layers, B, H_kv, L_max, D)
cache_lens:  (B,)
```

Or:

```text
None, None, None
```

if no request has any cache yet.

### Step 1: Collect Cache Lengths

```python
cache_lens_py = []
ref = None
for r in requests:
    cache = r.kv_cache
    if cache is not None and cache.k is not None:
        cache_lens_py.append(cache.k.shape[2])
        if ref is None:
            ref = cache.k
    else:
        cache_lens_py.append(0)
```

`cache.k.shape[2]` is the sequence length `L`.

If every request has no cache, `ref` stays `None` and the method returns no cache.

### Step 2: Allocate Padded Batched Cache

```python
max_cache_len = max(cache_lens_py)
prev_k = torch.zeros(self.num_layers, batch_size, H_kv, max_cache_len, D, ...)
prev_v = torch.zeros(self.num_layers, batch_size, H_kv, max_cache_len, D, ...)
```

Suppose:

```text
cache_lens_py = [100, 80, 120]
```

Then:

```text
L_max = 120
```

The batched cache has room for 120 positions for every request.

### Step 3: Copy Compact Caches Into the Batch

```python
prev_k[:, i, :, :L_i, :] = r.kv_cache.k
prev_v[:, i, :, :L_i, :] = r.kv_cache.v
```

For a request with only 80 cached tokens, positions 80 through 119 stay zero.

Those zero positions are padding. The model gets `cache_lens` so it can mask them.

## Why Scatter Caches?

After the model forward pass, the model returns updated batched caches:

```text
new_k: (num_layers, B, H_kv, L_prev + T, D)
new_v: (num_layers, B, H_kv, L_prev + T, D)
```

But each request should keep a compact cache again. It should not store padding.

That is what `_scatter_caches()` does.

## Scattering Caches

```python
def _scatter_caches(self, requests, new_k, new_v, L_prev, num_tokens, pad_lengths_py=None):
```

Inputs:

- `requests`: the requests in this forward pass.
- `new_k`, `new_v`: batched updated caches from the model.
- `L_prev`: previous padded cache length.
- `num_tokens`: current input length `T`.
- `pad_lengths_py`: left-padding lengths for prefill, or `None` for decode.

For each request:

```python
old_L = r.cache_pos
pl = pad_lengths_py[i] if pad_lengths_py is not None else 0
```

Then it slices:

```python
k_old = new_k[:, i, :, :old_L, :]
k_new = new_k[:, i, :, L_prev + pl : L_prev + num_tokens, :]
```

Meaning:

- Keep the old real cache columns.
- Keep only the new real token columns.
- Skip left-padding columns.

Then:

```python
r.kv_cache.k = torch.cat([k_old, k_new], dim=2)
r.kv_cache.v = torch.cat([v_old, v_new], dim=2)
```

Now the request cache is compact again.

## The Shared Forward Pass Helper

```python
def _forward_pass(self, tokens, requests: List[Request], pad_lengths_py=None):
```

Both prefill and decode call this method.

It does five things:

1. Gather per-request caches into batched caches.
2. Convert pad lengths into a tensor.
3. Run the model.
4. Scatter updated caches back to requests.
5. Return last-token logits.

### Step 1: Gather

```python
prev_k, prev_v, cache_lens = self._gather_caches(requests)
L_prev = prev_k.shape[3] if prev_k is not None else 0
T = tokens.shape[1]
```

`L_prev` is the padded previous cache length.

`T` is how many input tokens the model will process in this pass.

For prefill, `T` can be large.

For decode, `T` is always 1.

### Step 2: Convert Pad Lengths

```python
if pad_lengths_py is not None:
    pad_lengths_tensor = torch.tensor(pad_lengths_py, dtype=torch.long, device=tokens.device)
```

Prefill uses left padding, so it passes pad lengths.

Decode does not use left padding, so it passes `None`.

### Step 3: Run the Model

```python
with torch.no_grad():
    logits, new_k, new_v = self.model(
        tokens,
        prev_k=prev_k,
        prev_v=prev_v,
        cache_lens=cache_lens,
        pad_lengths=pad_lengths_tensor,
    )
```

`torch.no_grad()` means PyTorch does not store gradients. This is inference, not
training.

The model receives:

- Current input tokens.
- Previous KV cache.
- Real cache lengths.
- Left-padding lengths if needed.

### Step 4: Scatter

```python
self._scatter_caches(requests, new_k, new_v, L_prev, T, pad_lengths_py=pad_lengths_py)
```

Now each request owns an updated compact cache.

### Step 5: Return Last Logits

```python
return logits[:, -1, :]
```

The model returns logits for every input position:

```text
logits: (B, T, vocab_size)
```

Generation only needs the last position:

```text
last logits: (B, vocab_size)
```

## Prefill Batch

```python
def prefill_batch(self, requests: List[Request]):
```

Prefill processes prompts for requests that have not yet generated any tokens.

### Why Prefill Exists

If the prompt is:

```text
"Explain AGI"
```

the model must read the prompt before it can predict the first generated token.

That first full-prompt pass is prefill.

### Step 1: Build Suffix Tokens

The method starts:

```python
all_tokens = []
suffix_tokens = []
```

For each request, it ensures the prompt is tokenized.

Then it handles prefix cache:

```python
max_prefix_len = len(request.prompt_tokens) - 1
if request.cache_pos > max_prefix_len:
    request.cache_pos = max_prefix_len
    request.prefix_cache_len = min(request.prefix_cache_len, request.cache_pos)
    if request.kv_cache is not None:
        request.kv_cache = request.kv_cache.slice(request.cache_pos)
```

This protects against a full-prompt cache. The engine keeps one prompt token for
prefill so it can produce first-token logits.

Then:

```python
all_tokens.append(request.prompt_tokens)
suffix_tokens.append(request.prompt_tokens[request.cache_pos:])
```

If there is no prefix cache:

```text
cache_pos = 0
suffix = full prompt
```

If the prefix cache reused 100 tokens from a 120-token prompt:

```text
cache_pos = 100
suffix = final 20 prompt tokens
```

### Step 2: Pad Suffixes

Different requests can have different suffix lengths.

Example:

```text
request A suffix length = 3
request B suffix length = 5
```

To batch them, both rows need length 5:

```text
request A: [pad, pad, a, b, c]
request B: [d,   e,   f, g, h]
```

The code:

```python
max_len = max(len(t) for t in suffix_tokens)
pad_id = self.tokenizer.pad_token_id or 0
padded = [([pad_id] * (max_len - len(t))) + t for t in suffix_tokens]
pad_lengths_py = [max_len - len(t) for t in suffix_tokens]
tokens = torch.tensor(padded, device=device)
```

Important:

- Padding is on the left.
- `pad_lengths_py` remembers how much left padding each request got.
- The model uses pad lengths to create correct positions and masks.

### Step 3: Run Forward Pass

```python
last_logits = self._forward_pass(tokens, requests, pad_lengths_py=pad_lengths_py)
```

This fills or extends each request's KV cache.

### Step 4: Sample First Generated Token

```python
next_tokens = self.sample(last_logits)
```

This gives one generated token per request.

### Step 5: Update Requests

```python
for i, request in enumerate(requests):
    request.prompt_tokens = all_tokens[i]
    self._finalize_generated_token(request, next_tokens[i].item())
    request.cache_pos = len(all_tokens[i])
    request.is_prefill = False
    self._store_prompt_prefix(request)
```

After prefill:

- The request has one generated token.
- `cache_pos` equals full prompt length.
- `is_prefill` becomes `False`.
- The prompt KV cache may be stored in prefix cache.

If the first generated token is already EOS, `_finalize_generated_token()` marks the
request completed.

## Decode Batch

```python
def decode_batch(self, requests: List[Request]):
```

Decode is the repeated one-token-at-a-time phase.

### Step 1: Build Input Tokens

```python
tokens = torch.tensor([[r.tokens[-1]] for r in requests], device=device)
```

Each request feeds its last generated token back into the model.

Shape:

```text
(B, 1)
```

Example:

```text
request 0 last token = 100
request 1 last token = 200

tokens = [[100],
          [200]]
```

### Step 2: Run Forward Pass

```python
last_logits = self._forward_pass(tokens, requests, pad_lengths_py=None)
```

No left padding is needed because every request contributes exactly one token.

The KV cache gives the model the full context.

### Step 3: Sample and Update

```python
next_tokens = self.sample(last_logits)

for i, request in enumerate(requests):
    request.cache_pos += 1
    tok = next_tokens[i].item()
    self._finalize_generated_token(request, tok)
```

Each request gets one more token. Then the scheduler will call decode again on the
next loop, unless the request is completed.

## Getting the Next Batch

```python
def _get_next_batch(self):
    self.current_batch = [r for r in self.current_batch if not r.is_completed]
    while not self.pool.empty() and len(self.current_batch) < MAX_BATCH_SIZE:
        self.current_batch.append(self.pool.get())
    return self.current_batch
```

This is the admission step.

It:

1. Removes completed requests from the active batch.
2. Pulls waiting requests from the queue.
3. Stops when the active batch reaches `MAX_BATCH_SIZE`.

This is what allows continuous batching. New requests can join while older requests
are still decoding.

## One Generate Tick

```python
def generate(self):
```

This method advances the whole engine by one scheduler tick.

### Step 1: Refresh Active Batch

```python
self.current_batch = self._get_next_batch()
if not self.current_batch:
    return False
```

If there is no work, return `False`.

### Step 2: Split Prefill and Decode

```python
prefill_requests = [r for r in self.current_batch if r.is_prefill]
decode_requests  = [r for r in self.current_batch if not r.is_prefill]
```

New requests go to prefill.

Older active requests go to decode.

### Step 3: Prefill New Requests

```python
for i in range(0, len(prefill_requests), MAX_PREFILL_BATCH):
    chunk = prefill_requests[i : i + MAX_PREFILL_BATCH]
    self.prefill_batch(chunk)
```

Prefill is chunked because prompt processing can use a lot of memory.

### Step 4: Decode Existing Requests

```python
if decode_requests:
    self.decode_batch(decode_requests)
```

All decode requests are processed together for one token.

### Step 5: Report That Work Happened

```python
return True
```

`generate_text()` uses this return value to keep looping.

## Public API: generate_text

```python
def generate_text(self, prompts: Union[str, List[str]], max_tokens: int = 100):
```

This is the friendly method you call from outside.

### Step 1: Normalize Input

```python
if isinstance(prompts, str):
    prompts = [prompts]
```

The rest of the method can treat everything as a list.

### Step 2: Create Requests

```python
requests = []
for prompt in prompts:
    req = Request(id=self.request_id, prompt=prompt, max_tokens=max_tokens)
    self.request_id += 1
    self.add_request(req)
    requests.append(req)
```

Each prompt becomes a `Request`.

Each request is prepared and put into the queue.

### Step 3: Run Until Done

```python
while self.generate():
    pass
```

This keeps ticking the scheduler until no queued or active requests remain.

### Step 4: Return Results

```python
return requests[0] if len(requests) == 1 else requests
```

For one prompt, you get one `Request`.

For many prompts, you get a list of `Request` objects.

Each completed request has:

```text
request.tokens
request.response
request.prompt_tokens
request.prefix_cache_len
```

## End-to-End Example: One Prompt

Imagine:

```python
engine.generate_text("Explain AGI", max_tokens=3)
```

Here is the flow:

```text
1. generate_text() receives "Explain AGI".
2. It wraps it into ["Explain AGI"].
3. It creates Request(id=0, prompt="Explain AGI", max_tokens=3).
4. add_request() tokenizes the prompt and creates an empty Cache.
5. The request is put into pool.
6. generate() pulls it from pool into current_batch.
7. Since is_prefill=True, prefill_batch() runs the prompt.
8. The model returns logits and KV cache.
9. The engine samples token 1.
10. Request is_prefill becomes False.
11. generate() runs again.
12. decode_batch() feeds token 1 into the model with KV cache.
13. The engine samples token 2.
14. generate() runs again.
15. decode_batch() feeds token 2 into the model with KV cache.
16. The engine samples token 3.
17. max_tokens is reached, so request is_completed=True.
18. generate() runs again, sees no active work, and returns False.
19. generate_text() returns the completed Request.
```

## End-to-End Example: Three Prompts

Imagine:

```python
engine.generate_text(
    ["A", "B", "C"],
    max_tokens=2,
)
```

Flow:

```text
1. Create three requests.
2. Put all three in pool.
3. generate() admits all three into current_batch.
4. All three need prefill, so prefill_batch([A, B, C]) runs them together.
5. Each request gets its first generated token.
6. generate() runs again.
7. All three are now decode requests.
8. decode_batch([A, B, C]) runs one token for all three together.
9. Each request reaches max_tokens=2 and completes.
10. generate_text() returns a list of three completed requests.
```

This is batching. Instead of running the model separately for each prompt at every
step, the engine groups them.

## End-to-End Example: Prefix Cache

Suppose prefix cache is enabled:

```python
engine = Engine("Qwen/Qwen3-4B", enable_prefix_cache=True)
```

And two prompts share a beginning:

```text
prompt 1: "Long shared document... Question: A"
prompt 2: "Long shared document... Question: B"
```

Flow:

```text
1. Request 1 arrives.
2. Prefix cache is empty, so request 1 prefills the full prompt.
3. After prefill, the engine stores request 1's prompt KV cache.
4. Request 2 arrives.
5. _prepare_request() asks PrefixCache for the longest matching token prefix.
6. PrefixCache returns a sliced KV cache for the shared prefix.
7. Request 2 only prefills the suffix after the shared prefix.
8. Decode proceeds normally.
```

Prefix caching helps when the shared prefix is large and generation is relatively
short. It saves prefill work, not decode work.

## How PrefixCache Works

The actual prefix cache is in `prefix_cache.py`.

It stores:

```text
tuple(token_ids) -> Cache
```

It uses `OrderedDict`, so it can behave like an LRU cache.

### PrefixCacheMatch

```python
@dataclass(frozen=True)
class PrefixCacheMatch:
    length: int
    cache: Cache | None
```

This is returned from `get()`.

If there is no hit:

```text
length = 0
cache = None
```

If there is a hit:

```text
length = number of reused prefix tokens
cache = sliced KV cache for those tokens
```

### get()

`get(tokens, max_prefix_len)` finds the longest cached prefix that matches the new
prompt.

Example:

```text
cached: [10, 20, 30, 40]
new:    [10, 20, 30, 99]

match length = 3
```

It moves the used entry to the end of the `OrderedDict`, marking it recently used.

### put()

`put(tokens, cache)` stores a prompt cache.

It may slice the cache to fit the token budget:

```python
store_len = min(len(tokens), cache.length, self.max_tokens)
```

Then it evicts old entries until:

```text
total_tokens <= max_tokens
```

## Important Tensor Shapes

These shapes are useful to keep in your head.

### Request Cache

One request stores:

```text
k: (num_layers, H_kv, L, D)
v: (num_layers, H_kv, L, D)
```

For Qwen3-4B:

```text
num_layers = 36
H_kv = 8
D = 128
```

### Batched Cache

The model receives:

```text
prev_k: (num_layers, B, H_kv, L_max, D)
prev_v: (num_layers, B, H_kv, L_max, D)
```

### Tokens

Prefill:

```text
tokens: (B, T_prompt_or_suffix)
```

Decode:

```text
tokens: (B, 1)
```

### Logits

The model returns:

```text
logits: (B, T, vocab_size)
```

The engine uses:

```text
logits[:, -1, :] -> (B, vocab_size)
```

### Sampled Tokens

```text
next_tokens: (B, 1)
```

## Why Left Padding Is Used in Prefill

Suppose two prompt suffixes have different lengths:

```text
request 0 suffix: [a, b]
request 1 suffix: [c, d, e, f]
```

To make one tensor:

```text
request 0: [pad, pad, a, b]
request 1: [c,   d,   e, f]
```

This is left padding.

The engine then passes:

```text
pad_lengths = [2, 0]
```

The model uses this to:

- Give real tokens correct position ids.
- Prevent real tokens from attending to pad tokens.
- Let `_scatter_caches()` skip pad tokens when saving caches.

## Why Decode Does Not Need Padding

During decode, every active request contributes exactly one token:

```text
tokens: (B, 1)
```

So no request is shorter than another in the current step. The histories may have
different lengths, but that difference is handled by padded KV caches and
`cache_lens`, not by padding the new token.

## Current Limitations

This engine is intentionally simple. Some limitations:

- Greedy sampling only.
- No streaming API.
- No paged KV cache.
- No no-cache decode path.
- No per-request sampling settings.
- Model architecture is hardcoded for Qwen3-4B.
- Full vocabulary logits are computed for every prefill position.
- The scheduler is single-threaded and synchronous.

These are not failures. They are natural next steps for a learning engine.

## Reading Strategy for Beginners

If you want to understand the code, read in this order:

1. `Request`
   Understand what state one generation job needs.

2. `generate_text()`
   This is the public entry point.

3. `generate()`
   This is the scheduler tick.

4. `prefill_batch()`
   This handles prompts and first generated tokens.

5. `decode_batch()`
   This handles one generated token per active request.

6. `_forward_pass()`
   This is the bridge from engine to model.

7. `_gather_caches()` and `_scatter_caches()`
   These explain how per-request state becomes a batch and then becomes per-request
   state again.

8. `_prepare_request()` and `PrefixCache`
   These explain prefix reuse.

## Final Mental Model

The engine is a loop around one model.

```text
Requests enter a queue.
The scheduler admits them into a batch.
New requests do prefill.
Old requests do decode.
Every model call updates KV caches.
Completed requests leave the batch.
The loop stops when no work remains.
```

Once you understand that, the rest of the engine is mostly careful bookkeeping:

- Which tokens are new?
- Which KV cache positions are real?
- Which positions are padding?
- Which requests are done?
- Which prompt prefixes can be reused?

That bookkeeping is what turns a plain transformer forward pass into an inference
engine.
