import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

from .rope import apply_rope_vectorized

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover - kept for older local torch installs
    SDPBackend = None
    sdpa_kernel = None


SDPA_BACKENDS = ("auto", "flash", "efficient", "math")


def _sdpa_context(backend: str):
    if backend == "auto":
        return nullcontext()
    if sdpa_kernel is None or SDPBackend is None:
        raise RuntimeError("explicit SDPA backend selection requires torch.nn.attention.sdpa_kernel")

    backend_map = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "math": SDPBackend.MATH,
    }
    try:
        return sdpa_kernel(backends=[backend_map[backend]])
    except KeyError as exc:
        raise ValueError(f"unknown SDPA backend '{backend}'. Valid options: {SDPA_BACKENDS}") from exc


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, num_heads, num_kv_groups, head_dim, dtype=None, sdpa_backend: str = "auto"):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"
        if sdpa_backend not in SDPA_BACKENDS:
            raise ValueError(f"unknown SDPA backend '{sdpa_backend}'. Valid options: {SDPA_BACKENDS}")

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_kv_groups
        self.sdpa_backend = sdpa_backend

        self.q_proj = nn.Linear(d_in, num_heads * head_dim, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(num_heads * head_dim, d_in, bias=False, dtype=dtype)

        # QK-Norm: RMSNorm applied per-head before RoPE
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6, dtype=dtype)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6, dtype=dtype)

    def forward(self, x, attn_mask, cos, sin, position_ids, prev_k, prev_v):
        """
        x:             (B, T, D_in)
        attn_mask:     (B, 1, T, L_full) additive float mask, 0 or -inf
        cos, sin:      (max_pos, head_dim)
        position_ids:  (B, T) long, absolute positions for RoPE
        prev_k, prev_v: optional (B, H_kv, L_prev, D) batched prior cache; None on fresh prefill
        """
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_groups, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_groups, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.transpose(1, 2)  # (B, H,    T, D)
        k = k.transpose(1, 2)  # (B, H_kv, T, D)
        v = v.transpose(1, 2)

        q = apply_rope_vectorized(q, cos, sin, position_ids)
        k = apply_rope_vectorized(k, cos, sin, position_ids)

        # Batched cache concat. Shape-stable w.r.t. per-request cache lengths:
        # prior per-request padding in the cache region is already masked via attn_mask.
        if prev_k is not None:
            full_k = torch.cat([prev_k, k], dim=2)
            full_v = torch.cat([prev_v, v], dim=2)
        else:
            full_k, full_v = k, v

        with _sdpa_context(self.sdpa_backend):
            context = F.scaled_dot_product_attention(
                q, full_k, full_v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
                enable_gqa=True,
            )
        context = context.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        return self.o_proj(context), full_k, full_v
