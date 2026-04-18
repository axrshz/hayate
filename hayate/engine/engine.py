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

    prompt_tokens: List[int] = field(default_factory=list)
    tokens: List[int] = field(default_factory=list)
    kv_cache: Cache | None = None

    cache_pos: int = 0
    is_completed: bool = False
    is_prefill: bool = True
    use_cache: bool = True

    response: str | None = None


class Engine:
    def __init__(self, model_name: str, compile: bool = False):
        self.model = Qwen3Model()
        self.num_layers = self.model.num_layers
        self.num_kv_groups = self.model.num_kv_groups
        self.head_dim = self.model.head_dim
        load_weights(self.model, model_name)
        self.model = self.model.to(device)
        if compile:
            # dynamic=True tells Dynamo to treat varying (B, T, L_prev) as symbolic,
            # avoiding a recompile per unique shape combo. First forward still pays
            # a multi-minute trace/compile cost.
            self.model = torch.compile(self.model, dynamic=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.pool: Queue = Queue()
        self.current_batch: List[Request] = []
        self.request_id = 0

    def add_request(self, request: Request):
        """adds a request in the pool"""
        if request.kv_cache is None and request.use_cache:
            request.kv_cache = Cache()
        self.pool.put(request)

    def sample(self, logits):
        """select the next token greedily via argmax"""
        return torch.argmax(logits, dim=-1, keepdim=True)

    def _gather_caches(self, requests: List[Request]):
        """Stack per-request caches into a single batched, right-padded tensor.

        Returns (prev_k, prev_v, cache_lens) where prev_k/prev_v have shape
        (num_layers, B, H_kv, L_max, D), or (None, None, None) when no request has
        any cached K/V yet (pure prefill).
        """
        batch_size = len(requests)
        cache_lens_py: List[int] = []
        ref: torch.Tensor | None = None
        for r in requests:
            cache = r.kv_cache
            if cache is not None and cache.k is not None:
                cache_lens_py.append(cache.k.shape[2])
                if ref is None:
                    ref = cache.k
            else:
                cache_lens_py.append(0)

        if ref is None:
            return None, None, None

        max_cache_len = max(cache_lens_py)
        _, H_kv, _, D = ref.shape
        dtype, dev = ref.dtype, ref.device

        prev_k = torch.zeros(self.num_layers, batch_size, H_kv, max_cache_len, D, dtype=dtype, device=dev)
        prev_v = torch.zeros(self.num_layers, batch_size, H_kv, max_cache_len, D, dtype=dtype, device=dev)

        for i, r in enumerate(requests):
            L_i = cache_lens_py[i]
            if L_i > 0 and r.kv_cache is not None and r.kv_cache.k is not None:
                prev_k[:, i, :, :L_i, :] = r.kv_cache.k
                prev_v[:, i, :, :L_i, :] = r.kv_cache.v

        cache_lens = torch.tensor(cache_lens_py, dtype=torch.long, device=dev)
        return prev_k, prev_v, cache_lens

    def _scatter_caches(self, requests: List[Request], new_k, new_v,
                        L_prev: int, num_tokens: int, pad_lengths_py: List[int] | None = None):
        """Split updated batched caches back into per-request Cache storage.

        new_k, new_v: (num_layers, B, H_kv, L_prev + T, D) stacked model outputs.

        For each request i, form the compacted cache by concatenating:
          - cache-region valid part:  cols [0, old_L_i)
          - new-region real tokens:   cols [L_prev + pl_i, L_prev + T)
        into a single (num_layers, H_kv, old_L_i + T - pl_i, D) tensor.
        """
        for i, r in enumerate(requests):
            if r.kv_cache is None or not r.use_cache:
                continue
            old_L = r.cache_pos
            pl = pad_lengths_py[i] if pad_lengths_py is not None else 0

            k_old = new_k[:, i, :, :old_L, :]
            v_old = new_v[:, i, :, :old_L, :]
            k_new = new_k[:, i, :, L_prev + pl : L_prev + num_tokens, :]
            v_new = new_v[:, i, :, L_prev + pl : L_prev + num_tokens, :]

            r.kv_cache.k = torch.cat([k_old, k_new], dim=2)
            r.kv_cache.v = torch.cat([v_old, v_new], dim=2)

    def _forward_pass(self, tokens, requests: List[Request], pad_lengths_py=None):
        """One model forward pass over a batch of requests. Returns last-token logits (B, V)."""
        prev_k, prev_v, cache_lens = self._gather_caches(requests)
        L_prev = prev_k.shape[3] if prev_k is not None else 0
        T = tokens.shape[1]

        pad_lengths_tensor = None
        if pad_lengths_py is not None:
            pad_lengths_tensor = torch.tensor(pad_lengths_py, dtype=torch.long, device=tokens.device)

        with torch.no_grad():
            logits, new_k, new_v = self.model(
                tokens,
                prev_k=prev_k,
                prev_v=prev_v,
                cache_lens=cache_lens,
                pad_lengths=pad_lengths_tensor,
            )

        self._scatter_caches(requests, new_k, new_v, L_prev, T, pad_lengths_py=pad_lengths_py)
        return logits[:, -1, :]

    def prefill_batch(self, requests: List[Request]):
        """batched prefill for multiple requests in a single forward pass"""
        all_tokens = [self.tokenizer.encode(r.prompt) for r in requests]
        max_len = max(len(t) for t in all_tokens)

        pad_id = self.tokenizer.pad_token_id or 0
        padded = [([pad_id] * (max_len - len(t))) + t for t in all_tokens]
        pad_lengths_py = [max_len - len(t) for t in all_tokens]
        tokens = torch.tensor(padded, device=device)

        last_logits = self._forward_pass(tokens, requests, pad_lengths_py=pad_lengths_py)
        next_tokens = self.sample(last_logits)

        for i, request in enumerate(requests):
            request.prompt_tokens = all_tokens[i]
            request.tokens.append(next_tokens[i].item())
            request.cache_pos = len(all_tokens[i])
            request.is_prefill = False

    def decode_batch(self, requests: List[Request]):
        """single-step decode for a batch of requests in a single forward pass"""
        tokens = torch.tensor([[r.tokens[-1]] for r in requests], device=device)

        last_logits = self._forward_pass(tokens, requests, pad_lengths_py=None)
        next_tokens = self.sample(last_logits)

        for i, request in enumerate(requests):
            request.cache_pos += 1
            tok = next_tokens[i].item()
            request.tokens.append(tok)
            if tok == self.tokenizer.eos_token_id or len(request.tokens) >= request.max_tokens:
                request.is_completed = True
                request.response = self.tokenizer.decode(request.tokens, skip_special_tokens=True)

    def _get_next_batch(self):
        self.current_batch = [r for r in self.current_batch if not r.is_completed]
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

    def generate_text(self, prompts: Union[str, List[str]], max_tokens: int = 100):
        """the public api for generation"""
        if isinstance(prompts, str):
            prompts = [prompts]

        requests = []
        for prompt in prompts:
            req = Request(id=self.request_id, prompt=prompt, max_tokens=max_tokens)
            self.request_id += 1
            self.add_request(req)
            requests.append(req)

        while self.generate():
            pass

        return requests[0] if len(requests) == 1 else requests


if __name__ == "__main__":
    engine = Engine("Qwen/Qwen3-4B")
    result = engine.generate_text("Explain AGI")
    print(result.response)
