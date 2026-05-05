from __future__ import annotations

import torch


class Cache:
    """Per-request KV cache stored as a single stacked tensor across layers.

    Layout: k, v each have shape (num_layers, H_kv, L, D).
    Storing all layers together keeps gather/scatter at the engine level at O(B)
    Python→CUDA calls per forward instead of O(num_layers × B).
    """

    __slots__ = ("k", "v")

    def __init__(self):
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    @classmethod
    def from_tensors(cls, k: torch.Tensor | None, v: torch.Tensor | None) -> "Cache":
        cache = cls()
        cache.k = k
        cache.v = v
        return cache

    @property
    def length(self) -> int:
        return 0 if self.k is None else self.k.shape[2]

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
