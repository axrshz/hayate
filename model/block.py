import torch
import torch.nn as nn

from .attention import GroupedQueryAttention
from .linear import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size=2560, num_heads=32, num_kv_groups=8,
                 head_dim=128, intermediate_size=9728, rms_norm_eps=1e-6,
                 dtype=torch.bfloat16):
        super().__init__()
        self.self_attn = GroupedQueryAttention(
            d_in=hidden_size, num_heads=num_heads,
            num_kv_groups=num_kv_groups, head_dim=head_dim, dtype=dtype
        )
        self.mlp = FeedForward(emb_dim=hidden_size, hidden_dim=intermediate_size, dtype=dtype)
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype)

    def forward(self, x, mask, cos, sin, start_positions, caches):
        shortcut = x
        x = self.input_layernorm(x)
        x, next_caches = self.self_attn(x, mask, cos, sin, start_positions, caches)
        x = x + shortcut

        shortcut = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = x + shortcut
        return x, next_caches