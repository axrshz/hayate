import torch
from typing import List, Union
from transformers import AutoTokenizer
from dataclasses import dataclass, field
from queue import Queue

from hayate.model import Qwen3Model
from hayate.utils import load_weights
from hayate.model.cache import Cache

device = "cuda" if torch.cuda.is_available() else "cpu"

MAX_BATCH_SIZE = 20
MAX_DECODE_BATCH = 20
MAX_PREFILL_BATCH = 8

@dataclass
class Request:
    id: int = 0
    prompt: str = ""
    max_tokens: int = 100
    temperature: float = 0

    prompt_tokens: List[int] = field(default_factory=list)
    tokens: List[int] = field(default_factory=list)
    kv_cache: Cache | None = field(default_factory=lambda: Cache(n_layers=36))

    cache_pos: int = 0
    is_completed: bool = False
    is_prefill: bool = True
    use_cache: bool = True

    response: str = None


class Engine:
    def __init__(self, model_name: str):
        self.model = Qwen3Model()
        self.num_layers = self.model.num_layers
        load_weights(self.model, model_name)
        self.model = self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.pool = Queue()
        self.current_batch = []
        self.request_id = 0


    def add_request(self, request: Request):
        """adds a request in the pool"""
        self.pool.put(request)

    def sample(self, logits):
        """select the next token greedily via argmax"""
        return torch.argmax(logits, dim=-1, keepdim=True)
    
    def _forward_pass(self, tokens, caches, cache_positions, pad_lengths=None):
        """does one forward pass of the model and returns the next token"""
        with torch.no_grad():
            logits = self.model(tokens, caches=caches, start_positions=cache_positions, pad_lengths=pad_lengths)
        return logits[:, -1, :]

    def prefill_batch(self, requests: List[Request]):
        """batched prefill for multiple requests in a single forward pass"""

        all_tokens = [self.tokenizer.encode(r.prompt) for r in requests]
        max_len = max(len(t) for t in all_tokens)

        pad_id = self.tokenizer.pad_token_id or 0
        padded = [([pad_id] * (max_len - len(t))) + t for t in all_tokens]
        pad_lengths = [max_len - len(t) for t in all_tokens]
        tokens = torch.tensor(padded, device=device)  # (B, max_len)

        caches = [r.kv_cache if r.use_cache else None for r in requests]
        start_positions = [0] * len(requests)

        logits = self._forward_pass(tokens, caches, start_positions, pad_lengths=pad_lengths)
        next_tokens = self.sample(logits)  # (B, 1)

        for i, request in enumerate(requests):
            request.prompt_tokens = all_tokens[i]
            request.tokens.append(next_tokens[i].item())

            actual_len = len(all_tokens[i])
            pl = pad_lengths[i]

            # Trim pad-token entries from the KV cache so decode
            # starts with clean positional state
            if pl > 0 and request.use_cache:
                for layer_idx in range(self.num_layers):
                    cached = request.kv_cache.get(layer_idx)
                    if cached is not None:
                        k, v = cached
                        request.kv_cache.update(layer_idx, (k[:, :, pl:, :], v[:, :, pl:, :]))

            request.cache_pos = actual_len
            request.is_prefill = False

    def decode_batch(self, requests: List[Request]):
        """the decode step for a batch of requests in a single forward pass."""
        # stack the last token of each request: (B, 1)
        tokens = torch.tensor([[r.tokens[-1]] for r in requests], device=device)
        caches = [r.kv_cache for r in requests]  
        start_positions = [r.cache_pos for r in requests]

        logits = self._forward_pass(tokens, caches, start_positions)
        next_tokens = self.sample(logits)

        for i, request in enumerate(requests):
            request.cache_pos += 1
            tok = next_tokens[i].item()
            request.tokens.append(tok)
            if tok == self.tokenizer.eos_token_id or len(request.tokens) >= request.max_tokens:
                request.is_completed = True
                request.response = self.tokenizer.decode(request.tokens)

    def _get_next_batch(self):
        self.current_batch = [_req for _req in self.current_batch if not _req.is_completed]  # in the current


        while not self.pool.empty() and len(self.current_batch) < MAX_BATCH_SIZE:
            self.current_batch.append(self.pool.get())

        return self.current_batch

    def generate(self):
        self.current_batch = self._get_next_batch()
        if not self.current_batch:
            return False
 
        prefill_requests = [r for r in self.current_batch if r.is_prefill]
        decode_requests  = [r for r in self.current_batch if not r.is_prefill]

        for i in range(0, len(prefill_requests), MAX_PREFILL_BATCH):
            chunk = prefill_requests[i : i + MAX_PREFILL_BATCH]
            self.prefill_batch(chunk)


        if decode_requests:
            self.decode_batch(decode_requests)
 
        return True
    
    def generate_text(self, prompts: Union[str, List[str]], max_tokens: int = 100, temperature: float = 0.0):
        """the public api for generation"""
        if isinstance(prompts, str):
            prompts = [prompts]
        
        requests = []
        for prompt in prompts:
            req = Request(id=self.request_id, prompt=prompt, max_tokens=max_tokens, temperature=temperature)
            self.request_id += 1
            self.add_request(req)
            requests.append(req)
        

        while self.generate():
            pass

        return requests[0] if len(requests) == 1 else requests
        

if __name__ == "__main__":
    engine = Engine("Qwen/Qwen3-4B")
    print(engine.generate_text("Explain AGI"))