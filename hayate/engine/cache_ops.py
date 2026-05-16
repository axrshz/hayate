from typing import List

import torch

from hayate.engine.request import Request


def gather_caches(requests: List[Request], num_layers: int):
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

    prev_k = torch.zeros(num_layers, batch_size, H_kv, max_cache_len, D, dtype=dtype, device=dev)
    prev_v = torch.zeros(num_layers, batch_size, H_kv, max_cache_len, D, dtype=dtype, device=dev)

    for i, r in enumerate(requests):
        L_i = cache_lens_py[i]
        if L_i > 0 and r.kv_cache is not None and r.kv_cache.k is not None:
            prev_k[:, i, :, :L_i, :] = r.kv_cache.k
            prev_v[:, i, :, :L_i, :] = r.kv_cache.v

    cache_lens = torch.tensor(cache_lens_py, dtype=torch.long, device=dev)
    return prev_k, prev_v, cache_lens


def scatter_caches(
    requests: List[Request],
    new_k,
    new_v,
    L_prev: int,
    num_tokens: int,
    pad_lengths_py: List[int] | None = None,
):
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
