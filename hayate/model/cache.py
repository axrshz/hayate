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

    @property
    def length(self) -> int:
        return 0 if self.k is None else self.k.shape[2]

    def reset(self) -> None:
        self.k = None
        self.v = None
