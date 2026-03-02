## Hayate

minimal inference engine for qwen 3 0.6B.

### Checklist

- [x] qwen 3 architecture
- [ ] kv cache
- [ ] continuous batching
- [ ] paged attention

### Project Structure

```text
hayate/
├── model.py          # Model architecture + HF weight loading
├── engine.py         # KV cache allocator, scheduler, continuous batching (planned)
├── sampler.py        # Greedy / top-k sampling
├── config.py         # Qwen3-0.6B hyperparameters
├── bench.py          # Tokens/sec, memory usage (planned)
└── main.py           # CLI entry point
```
