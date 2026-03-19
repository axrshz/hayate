## Hayate

minimal inference engine for qwen 3 0.6B. (wip.)

### Checklist

- [x] qwen 3 architecture
- [x] kv cache
- [x] continuous batching (position-bucketed with per-request KV merge/split)
- [ ] paged attention

### Project Structure

```text
hayate/
├── model.py          # Model architecture + HF weight loading
├── engine.py         # KV cache + continuous batching scheduler
├── sampler.py        # Greedy / top-k sampling
├── config.py         # Qwen3-0.6B hyperparameters
├── bench.py          # Tokens/sec, memory usage (planned)
└── main.py           # CLI entry point
```

### Run

Single request:

```bash
python main.py --prompt "Explain KV cache briefly."
```

Continuous batching (multiple requests can arrive at different scheduler steps):

```bash
python main.py --batch-prompts "hello" "tell me a joke" "explain attention" --arrival-steps 0,2,2
```
