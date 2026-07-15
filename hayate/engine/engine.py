import torch
from typing import List, Union
from transformers import AutoTokenizer, GenerationConfig

from hayate.model import Qwen3Model
from hayate.utils import load_weights
from hayate.model.cache import Cache
from hayate.engine.prefix_cache import PrefixCache
from hayate.engine.constants import (
    COMPILE_MODES,
    DEFAULT_PREFIX_CACHE_MAX_TOKENS,
    MAX_PREFILL_BATCH,
    device,
)
from hayate.engine.request import Request
from hayate.engine.cache_ops import gather_caches, scatter_caches
from hayate.engine.sampler import Sampler
from hayate.engine.scheduler import Scheduler


class Engine:
    def __init__(
        self,
        model_name: str,
        compile: bool = False,
        compile_mode: str = "default",
        enable_prefix_cache: bool = False,
        prefix_cache_max_tokens: int = DEFAULT_PREFIX_CACHE_MAX_TOKENS,
    ):
        if compile_mode not in COMPILE_MODES:
            raise ValueError(f"unknown compile mode '{compile_mode}'. Valid options: {COMPILE_MODES}")

        # Construct on the meta device so PyTorch does not initialize an 8GB set
        # of random CPU weights that the checkpoint immediately overwrites.
        with torch.device("meta"):
            self.model = Qwen3Model()
        self.num_layers = self.model.num_layers
        self.num_kv_groups = self.model.num_kv_groups
        self.head_dim = self.model.head_dim
        self.max_position_embeddings = self.model.max_position_embeddings
        self.model.to_empty(device=device)
        self.model.initialize_rope(device)
        model_dir = load_weights(self.model, model_name)
        self.model.eval()
        if compile:
            # dynamic=True lets Dynamo handle varying batch size and cache length.
            # The default compile mode is friendlier to 24GB GPUs than reduce-overhead,
            # which uses CUDA Graph private pools for every new decode cache length.
            if compile_mode == "default":
                self.model = torch.compile(self.model, dynamic=True)
            else:
                self.model = torch.compile(self.model, dynamic=True, mode=compile_mode)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)
        self.stop_token_ids = self._load_stop_token_ids(model_dir)
        self.scheduler = Scheduler()
        self.prefix_cache = PrefixCache(prefix_cache_max_tokens) if enable_prefix_cache else None
        self.sampler = Sampler(self.tokenizer, self.stop_token_ids)

    def _load_stop_token_ids(self, model_name: str) -> set[int]:
        """Collect every configured EOS token, not just tokenizer.eos_token_id."""
        token_ids = set()
        try:
            generation_config = GenerationConfig.from_pretrained(model_name)
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, int):
                token_ids.add(eos_token_id)
            elif eos_token_id is not None:
                token_ids.update(int(tok) for tok in eos_token_id)
        except Exception:
            pass

        if self.tokenizer.eos_token_id is not None:
            token_ids.add(int(self.tokenizer.eos_token_id))
        return token_ids

    def add_request(self, request: Request):
        """Adds a request to the pool."""
        self._prepare_request(request)
        self.scheduler.add(request)

    def _prepare_request(self, request: Request):
        """Tokenize and seed the request with the longest reusable prompt prefix."""
        if not request.prompt_tokens:
            request.prompt_tokens = self.tokenizer.encode(request.prompt)

        if not request.use_cache:
            raise ValueError("use_cache=False is not supported by decode; leave request caching enabled")

        self._validate_request_lengths(request)

        if request.kv_cache is not None and request.kv_cache.length > 0:
            request.cache_pos = request.kv_cache.length
            request.prefix_cache_len = request.cache_pos
            return

        request.kv_cache = Cache()
        if self.prefix_cache is None:
            return

        # Keep at least one prompt token for prefill so we can compute next-token logits.
        max_prefix_len = max(len(request.prompt_tokens) - 1, 0)
        match = self.prefix_cache.get(request.prompt_tokens, max_prefix_len=max_prefix_len)
        if match.cache is None or match.length == 0:
            return

        request.kv_cache = match.cache
        request.cache_pos = match.length
        request.prefix_cache_len = match.length

    def _validate_request_lengths(self, request: Request):
        if request.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if not request.prompt_tokens:
            raise ValueError("prompt must tokenize to at least one token")

        required_positions = len(request.prompt_tokens) + request.max_tokens - 1
        if required_positions > self.max_position_embeddings:
            raise ValueError(
                "request exceeds model context window: "
                f"prompt tokens ({len(request.prompt_tokens)}) + generated-token positions "
                f"({request.max_tokens - 1}) = {required_positions}, "
                f"max supported positions = {self.max_position_embeddings}"
            )

    def clear_prefix_cache(self):
        """Drop all cached prompt prefixes."""
        if self.prefix_cache is not None:
            self.prefix_cache.clear()

    def _store_prompt_prefix(self, request: Request):
        if self.prefix_cache is None or not request.use_cache or request.kv_cache is None:
            return
        if not request.prompt_tokens or request.kv_cache.length < len(request.prompt_tokens):
            return
        self.prefix_cache.put(request.prompt_tokens, request.kv_cache)

    def _forward_pass(self, tokens, requests: List[Request], pad_lengths_py=None):
        """One model forward pass over a batch of requests. Returns last-token logits (B, V)."""
        prev_k, prev_v, cache_lens, cache_lens_py = gather_caches(requests, self.num_layers)
        L_prev = prev_k.shape[3] if prev_k is not None else 0
        T = tokens.shape[1]

        pad_lengths_tensor = None
        if pad_lengths_py is not None and any(pad_lengths_py):
            pad_lengths_tensor = torch.tensor(pad_lengths_py, dtype=torch.long, device=tokens.device)

        # Flash SDPA cannot consume an explicit padding mask. PyTorch's varlen
        # Flash kernel handles packed uneven sequences without falling back to
        # quadratic math attention. A cached multi-token suffix also needs the
        # varlen kernel's bottom-right causal alignment.
        uneven_cache = len(set(cache_lens_py)) > 1
        use_varlen = pad_lengths_tensor is not None or uneven_cache or (L_prev > 0 and T > 1)

        # torch.compile currently graph-breaks around varlen_attn and is slower
        # than eager execution for uneven batches. Keep compiled dense/uniform
        # paths while routing packed attention through the original module.
        forward_model = self.model
        if use_varlen and hasattr(self.model, "_orig_mod"):
            forward_model = self.model._orig_mod

        with torch.inference_mode():
            logits, new_k, new_v = forward_model(
                tokens,
                prev_k=prev_k,
                prev_v=prev_v,
                cache_lens=cache_lens,
                pad_lengths=pad_lengths_tensor,
                use_varlen=use_varlen,
            )

        scatter_caches(requests, new_k, new_v, T, pad_lengths_py=pad_lengths_py)
        return logits[:, -1, :]

    def prefill_batch(self, requests: List[Request]):
        """Batched prefill for multiple requests in a single forward pass."""
        all_tokens = []
        suffix_tokens = []
        for request in requests:
            if not request.prompt_tokens:
                request.prompt_tokens = self.tokenizer.encode(request.prompt)
            if not request.prompt_tokens:
                raise ValueError("prompt must tokenize to at least one token")

            # A full prompt cache cannot produce next-token logits by itself. Leave the
            # final prompt token for prefill so the model emits the first sampled token.
            max_prefix_len = len(request.prompt_tokens) - 1
            if request.cache_pos > max_prefix_len:
                request.cache_pos = max_prefix_len
                request.prefix_cache_len = min(request.prefix_cache_len, request.cache_pos)
                if request.kv_cache is not None:
                    request.kv_cache = request.kv_cache.slice(request.cache_pos)

            all_tokens.append(request.prompt_tokens)
            suffix_tokens.append(request.prompt_tokens[request.cache_pos:])

        max_len = max(len(t) for t in suffix_tokens)

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        padded = [([pad_id] * (max_len - len(t))) + t for t in suffix_tokens]
        pad_lengths_py = [max_len - len(t) for t in suffix_tokens]
        tokens = torch.tensor(padded, device=device)

        last_logits = self._forward_pass(tokens, requests, pad_lengths_py=pad_lengths_py)
        next_tokens = self.sampler.sample(last_logits)
        next_token_ids = next_tokens.flatten().tolist()

        for i, request in enumerate(requests):
            request.prompt_tokens = all_tokens[i]
            self.sampler.finalize(request, next_token_ids[i])
            request.cache_pos = len(all_tokens[i])
            request.is_prefill = False
            self._store_prompt_prefix(request)

    def decode_batch(self, requests: List[Request]):
        """Single-step decode for a batch of requests in a single forward pass."""
        tokens = torch.tensor([[r.tokens[-1]] for r in requests], device=device)

        last_logits = self._forward_pass(tokens, requests, pad_lengths_py=None)
        next_tokens = self.sampler.sample(last_logits)
        next_token_ids = next_tokens.flatten().tolist()

        for i, request in enumerate(requests):
            request.cache_pos += 1
            tok = next_token_ids[i]
            self.sampler.finalize(request, tok)

    def generate(self):
        """Advance the engine by one scheduler tick."""
        batch = self.scheduler.tick()
        if not batch:
            return False

        prefill_requests = [r for r in batch if r.is_prefill]
        decode_requests = [r for r in batch if not r.is_prefill]

        for i in range(0, len(prefill_requests), MAX_PREFILL_BATCH):
            chunk = prefill_requests[i : i + MAX_PREFILL_BATCH]
            self.prefill_batch(chunk)

        if decode_requests:
            self.decode_batch(decode_requests)

        return True

    def generate_text(self, prompts: Union[str, List[str]], max_tokens: int = 100):
        """The public API for generation."""
        if isinstance(prompts, str):
            prompts = [prompts]

        requests = []
        for prompt in prompts:
            req = Request(id=self.scheduler.request_id, prompt=prompt, max_tokens=max_tokens)
            self.scheduler.request_id += 1
            self.add_request(req)
            requests.append(req)

        while self.generate():
            pass

        return requests[0] if len(requests) == 1 else requests


if __name__ == "__main__":
    engine = Engine("Qwen/Qwen3-4B")
    result = engine.generate_text("Explain AGI")
    print(result.response)
