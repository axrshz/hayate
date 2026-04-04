import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope_vectorized
from hayate.utils import pad_to


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

    def forward(self, x, mask, cos, sin, position_ids, caches):
        batch_size, num_tokens, _ = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim)

        # QK-Norm: normalize before RoPE, while shape is (B, T, heads, head_dim)
        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        queries = queries.transpose(1, 2)

        keys = apply_rope_vectorized(keys, cos, sin, position_ids)
        queries = apply_rope_vectorized(queries, cos, sin, position_ids)

        next_caches = []
        keys_list, values_list = [], []
        for i in range(batch_size):
            k_i, v_i = keys[i:i+1], values[i:i+1]
            if caches[i] is not None:
                prev_k, prev_v = caches[i]
                k_i = torch.cat([prev_k, k_i], dim=2)
                v_i = torch.cat([prev_v, v_i], dim=2)
            next_caches.append((k_i, v_i))
            keys_list.append(k_i)
            values_list.append(v_i)

        max_seq = max(k.shape[2] for k in keys_list)

        keys_padded = torch.cat([pad_to(k, max_seq) for k in keys_list], dim=0)
        values_padded = torch.cat([pad_to(v, max_seq) for v in values_list], dim=0)

        # mask: (B, 1, T_q, T_k) bool, True = invalid — SDPA additive mask uses 0 / -inf
        attn_mask = torch.zeros(
            batch_size, 1, num_tokens, max_seq,
            device=queries.device, dtype=torch.float32,
        )
        attn_mask = attn_mask.masked_fill(mask, float("-inf"))

        # PyTorch SDPA (Flash/Memory-efficient/Math); enable_gqa avoids expanding KV heads
        context_vec = F.scaled_dot_product_attention(
            queries,
            keys_padded,
            values_padded,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=True,
        )
        context_vec = context_vec.transpose(1, 2).reshape(
            batch_size, num_tokens, self.num_heads * self.head_dim
        )
        return self.o_proj(context_vec), next_caches