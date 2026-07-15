from __future__ import annotations

import torch


class Cache:
    """Per-request KV cache stored as a single stacked tensor across layers.

    Layout: k, v each have shape (num_layers, H_kv, L, D).
    Storing all layers together keeps gather/scatter at the engine level at O(B)
    Python→CUDA calls per forward instead of O(num_layers × B).
    """

    __slots__ = ("k", "v", "_length")

    def __init__(self):
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None
        self._length = 0

    @classmethod
    def from_tensors(cls, k: torch.Tensor | None, v: torch.Tensor | None) -> "Cache":
        cache = cls()
        cache.k = k
        cache.v = v
        cache._length = 0 if k is None else k.shape[2]
        return cache

    @property
    def length(self) -> int:
        return self._length

    @property
    def capacity(self) -> int:
        return 0 if self.k is None else self.k.shape[2]

    def append(self, k: torch.Tensor, v: torch.Tensor, capacity: int | None = None) -> None:
        """Append KV states, reserving storage to avoid O(sequence_length²) copies."""
        if k.shape != v.shape:
            raise ValueError("key and value cache tensors must have matching shapes")
        new_tokens = k.shape[2]
        if new_tokens == 0:
            return

        required = self._length + new_tokens
        if self.k is None or self.v is None or self.capacity < required:
            new_capacity = max(required, capacity or 0, max(1, self.capacity * 2))
            shape = (*k.shape[:2], new_capacity, k.shape[3])
            new_k = torch.empty(shape, dtype=k.dtype, device=k.device)
            new_v = torch.empty(shape, dtype=v.dtype, device=v.device)
            if self.k is not None and self.v is not None and self._length:
                new_k[:, :, : self._length, :].copy_(self.k[:, :, : self._length, :])
                new_v[:, :, : self._length, :].copy_(self.v[:, :, : self._length, :])
            self.k, self.v = new_k, new_v

        self.k[:, :, self._length : required, :].copy_(k)
        self.v[:, :, self._length : required, :].copy_(v)
        self._length = required

    def slice(self, end: int, clone: bool = False) -> "Cache":
        if end < 0:
            raise ValueError("cache slice end must be non-negative")
        if self.k is None or self.v is None:
            return Cache()
        if end > self.length:
            raise ValueError(f"cache slice end {end} exceeds cache length {self.length}")

        k = self.k[:, :, :end, :]
        v = self.v[:, :, :end, :]
        if clone:
            k = k.contiguous().clone()
            v = v.contiguous().clone()
        return Cache.from_tensors(k, v)

    def reset(self) -> None:
        self.k = None
        self.v = None
        self._length = 0
