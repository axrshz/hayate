import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope_vectorized

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.attention.varlen import varlen_attn
except ImportError:  # pragma: no cover - kept for older local torch installs
    SDPBackend = None
    sdpa_kernel = None
    varlen_attn = None


def _flash_sdpa_context():
    if sdpa_kernel is None or SDPBackend is None:
        raise RuntimeError("Flash SDPA requires torch.nn.attention.sdpa_kernel")
    # Explicit masks are not accepted by the Flash backend in current PyTorch.
    # Keep the math backend available as a correctness fallback; optimized paths
    # below avoid the mask and therefore still select Flash Attention.
    return sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.MATH])


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, num_heads, num_kv_groups, head_dim, dtype=None):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_kv_groups

        self.q_proj = nn.Linear(d_in, num_heads * head_dim, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(num_heads * head_dim, d_in, bias=False, dtype=dtype)

        # QK-Norm: RMSNorm applied per-head before RoPE
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6, dtype=dtype)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6, dtype=dtype)

    def forward(self, x, cos, sin, position_ids, prev_k, prev_v, varlen_metadata=None):
        """
        x:             (B, T, D_in)
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

        if varlen_metadata is not None:
            if varlen_attn is None:
                raise RuntimeError("variable-length Flash Attention requires PyTorch 2.10+")
            valid_q, valid_k, cu_q, cu_k, max_q, max_k, is_causal = varlen_metadata
            q_packed = q.transpose(1, 2)[valid_q]
            k_packed = full_k.transpose(1, 2)[valid_k]
            v_packed = full_v.transpose(1, 2)[valid_k]
            packed_context = varlen_attn(
                q_packed,
                k_packed,
                v_packed,
                cu_q,
                cu_k,
                max_q,
                max_k,
                is_causal=is_causal,
            )
            context = torch.zeros(
                B, T, self.num_heads, self.head_dim, dtype=x.dtype, device=x.device
            )
            context[valid_q] = packed_context
        else:
            # Fresh unpadded prefills and uniform one-token decode need no explicit
            # mask, allowing PyTorch to dispatch directly to Flash Attention.
            with _flash_sdpa_context():
                context = F.scaled_dot_product_attention(
                    q,
                    full_k,
                    full_v,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=T > 1,
                    enable_gqa=True,
                ).transpose(1, 2)
        context = context.reshape(B, T, self.num_heads * self.head_dim)
        # The engine already owns the prior cache. Return only the newly computed
        # states so it does not stack and copy the full history after every layer.
        return self.o_proj(context), k, v
