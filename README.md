# hayate

a minimal llm inference engine inspired by vLLM built from scratch using qwen3 0.6b as the target model. my goal is to understand how modern inference systems work by building one.

---

### phase 1 — transformer forward pass

- [x] load qwen3 0.6b weights from huggingface (use `transformers` only for tokenizer + weight loading)
- [x] implement token embeddings + rotary positional embeddings (rope)
- [x] implement grouped query attention (gqa) — qwen3 uses 16q / 8kv heads
- [x] implement swiglu mlp block
- [x] implement rmsnorm + residual connections
- [x] implement greedy decoding loop (autoregressive, one token at a time)
- [x] verify outputs match huggingface's generate() for the same prompt

---

### phase 2 — kv cache

- [ ] allocate k/v tensors per layer to store past tokens
- [ ] modify attention to only compute new token against cached k/v
- [ ] handle cache correctly with gqa (kv heads != q heads)
- [ ] add a max sequence length limit and handle overflow
- [ ] measure tokens/sec before and after, log the difference

---

### phase 3 — continuous batching

- [ ] implement a request queue that holds multiple prompts
- [ ] implement naive static batching with padding first
- [ ] refactor into iteration-level scheduling (continuous batching)
  - [ ] each step, build a batch from active sequences at their current position
  - [ ] evict finished sequences immediately, admit new ones mid-flight
- [ ] benchmark throughput vs number of concurrent requests

---

### phase 4 — paged attention

- [ ] define a block size (e.g. 16 tokens per block)
- [ ] implement a block allocator (free list of physical blocks)
- [ ] give each sequence a block table (logical → physical mapping)
- [ ] allocate blocks on demand as sequences grow
- [ ] free blocks when a sequence finishes
- [ ] implement copy-on-write for shared prefix blocks
- [ ] compare max concurrent sequences vs naive pre-allocated kv cache

---

### stretch goals (pick what interests you)

- [ ] int8 weight quantization
- [ ] speculative decoding with a draft model
- [ ] openai-compatible `/v1/completions` api via fastapi
- [ ] custom triton kernels (flash attention concepts)

---
