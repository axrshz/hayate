from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from sampler import sample_next_token


class KVCache:
    def __init__(self, n_layers):
        self.cache = [None] * n_layers

    def get(self, layer_idx):
        return self.cache[layer_idx]

    def update(self, layer_idx, value):
        self.cache[layer_idx] = value

    def reset(self):
        for i in range(len(self.cache)):
            self.cache[i] = None


@dataclass
class GenerationRequest:
    request_id: str
    input_ids: torch.Tensor
    max_new_tokens: int
    temperature: float = 0.8
    top_k: int = 50
    eos_token_id: Optional[int] = None
    arrival_step: int = 0


@dataclass
class GenerationResult:
    request_id: str
    output_ids: torch.Tensor
    finish_reason: str
    arrival_step: int
    finish_step: int


class _RequestState:
    def __init__(self, req: GenerationRequest, n_layers: int):
        self.req = req
        self.cache = KVCache(n_layers=n_layers)
        self.prompt_len = req.input_ids.shape[1]
        self.prompt_cursor = 0
        self.processed_tokens = 0
        self.generated: List[torch.Tensor] = []
        self.pending_decode_token: Optional[torch.Tensor] = None
        self.finished = req.max_new_tokens <= 0
        self.finish_reason = "length"
        self.finish_step = req.arrival_step if self.finished else -1

    def next_input_token(self) -> torch.Tensor:
        # During prefill, feed prompt tokens one-by-one.
        if self.prompt_cursor < self.prompt_len:
            token = self.req.input_ids[:, self.prompt_cursor : self.prompt_cursor + 1]
            self.prompt_cursor += 1
            return token

        # During decode, feed last generated token.
        if self.pending_decode_token is None:
            raise RuntimeError("Decode token is missing for request.")
        return self.pending_decode_token

    def update_after_logits(self, logits: torch.Tensor, step_no: int) -> None:
        self.processed_tokens += 1

        # Keep prefill cheap: only sample when we reached the last prompt token.
        if self.prompt_cursor < self.prompt_len:
            return

        next_token = sample_next_token(logits[:, -1, :], temperature=self.req.temperature, top_k=self.req.top_k)
        self.generated.append(next_token)
        self.pending_decode_token = next_token

        if self.req.eos_token_id is not None and torch.all(next_token.squeeze(-1) == self.req.eos_token_id):
            self.finished = True
            self.finish_reason = "eos"
            self.finish_step = step_no
            return

        if len(self.generated) >= self.req.max_new_tokens:
            self.finished = True
            self.finish_reason = "length"
            self.finish_step = step_no

    def output_ids(self) -> torch.Tensor:
        if self.generated:
            return torch.cat(self.generated, dim=1)
        return self.req.input_ids.new_empty((self.req.input_ids.size(0), 0))


def _merge_layer_cache(states: List[_RequestState], layer_idx: int):
    layer_entries = [s.cache.get(layer_idx) for s in states]
    if any(entry is None for entry in layer_entries):
        return None

    keys = torch.cat([entry[0] for entry in layer_entries], dim=0)
    values = torch.cat([entry[1] for entry in layer_entries], dim=0)
    return keys, values


def _split_layer_cache(merged_cache, batch_sizes: List[int]):
    if merged_cache is None:
        return [None] * len(batch_sizes)

    keys, values = merged_cache
    split_keys = torch.split(keys, batch_sizes, dim=0)
    split_values = torch.split(values, batch_sizes, dim=0)
    return list(zip(split_keys, split_values))


class _BatchKVCache:
    def __init__(self, n_layers: int, states: List[_RequestState]):
        self.n_layers = n_layers
        self.states = states
        self.batch_sizes = [s.req.input_ids.shape[0] for s in states]

    def get(self, layer_idx):
        return _merge_layer_cache(self.states, layer_idx)

    def update(self, layer_idx, value):
        per_state = _split_layer_cache(value, self.batch_sizes)
        for state, split_value in zip(self.states, per_state):
            state.cache.update(layer_idx, split_value)


class ContinuousBatchingEngine:
    def __init__(self, model):
        self.model = model
        self.n_layers = model.cfg["n_layers"]
        self._next_seq_no = 0

    @torch.inference_mode()
    def generate(self, requests: List[GenerationRequest]) -> List[GenerationResult]:
        if not requests:
            return []

        pending = sorted(requests, key=lambda r: (r.arrival_step, self._next_seq()))
        active: List[_RequestState] = []
        completed: List[GenerationResult] = []
        global_step = 0

        while pending or active:
            while pending and pending[0].arrival_step <= global_step:
                req = pending.pop(0)
                active.append(_RequestState(req=req, n_layers=self.n_layers))

            if not active:
                global_step += 1
                continue

            buckets: Dict[int, List[_RequestState]] = {}
            for state in active:
                if state.finished:
                    continue
                buckets.setdefault(state.processed_tokens, []).append(state)

            for _, bucket_states in sorted(buckets.items(), key=lambda kv: kv[0]):
                batch_tokens = torch.cat([s.next_input_token() for s in bucket_states], dim=0)
                merged_cache = _BatchKVCache(n_layers=self.n_layers, states=bucket_states)
                self.model.current_pos = bucket_states[0].processed_tokens
                logits = self.model(batch_tokens, cache=merged_cache)

                for i, state in enumerate(bucket_states):
                    state_logits = logits[i : i + 1]
                    state.update_after_logits(state_logits, global_step)

            still_active = []
            for state in active:
                if state.finished:
                    completed.append(
                        GenerationResult(
                            request_id=state.req.request_id,
                            output_ids=state.output_ids(),
                            finish_reason=state.finish_reason,
                            arrival_step=state.req.arrival_step,
                            finish_step=state.finish_step,
                        )
                    )
                else:
                    still_active.append(state)
            active = still_active
            global_step += 1

        completed.sort(key=lambda r: r.request_id)
        return completed

    def _next_seq(self):
        seq = self._next_seq_no
        self._next_seq_no += 1
        return seq
