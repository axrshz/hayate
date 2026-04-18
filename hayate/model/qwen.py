import torch
import torch.nn as nn

from .rope import compute_rope_params
from .block import TransformerBlock


class Qwen3Model(nn.Module):
    def __init__(self):
        super().__init__()

        vocab_size = 151_936
        hidden_size = 2560
        num_layers = 36
        num_heads = 32
        num_kv_groups = 8
        head_dim = 128
        intermediate_size = 9728
        rms_norm_eps = 1e-6
        rope_theta = 1_000_000
        max_position_embeddings = 40_960

        self.num_layers = num_layers
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, dtype=torch.bfloat16)
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size, num_heads=num_heads,
                num_kv_groups=num_kv_groups, head_dim=head_dim,
                intermediate_size=intermediate_size, rms_norm_eps=rms_norm_eps,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps, dtype=torch.bfloat16)
        self.out_head = nn.Linear(hidden_size, vocab_size, bias=False, dtype=torch.bfloat16)

        cos, sin = compute_rope_params(
            head_dim=head_dim, theta_base=rope_theta,
            context_length=max_position_embeddings
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, token_ids, prev_k=None, prev_v=None,
                cache_lens=None, pad_lengths=None):
        """
        token_ids:    (B, T) int64
        prev_k,
        prev_v:       optional (num_layers, B, H_kv, L_prev, D) stacked prior caches;
                      None on fresh prefill.
        cache_lens:   optional (B,) int64 tensor of per-request valid cache length
                      (values in [0, L_prev]); required when prev_k is not None.
        pad_lengths:  optional (B,) int64 tensor of per-request left-pad lengths in the
                      new-token region; only used during padded prefill.

        Returns: logits (B, T, V), new_k, new_v (each (num_layers, B, H_kv, L_prev+T, D)).
        """
        B, T = token_ids.shape
        x = self.embed_tokens(token_ids)

        L_prev = prev_k.shape[3] if prev_k is not None else 0
        L_full = L_prev + T

        arange_t = torch.arange(T, device=x.device, dtype=torch.long).unsqueeze(0)  # (1, T)

        if cache_lens is not None:
            cl_1d = cache_lens.view(-1, 1)  # (B, 1)
        else:
            cl_1d = torch.zeros(B, 1, device=x.device, dtype=torch.long)

        if pad_lengths is not None:
            pl_1d = pad_lengths.view(-1, 1)  # (B, 1)
            position_ids = (arange_t + cl_1d - pl_1d).clamp(min=0)
        else:
            position_ids = arange_t + cl_1d  # broadcasts to (B, T)

        # Attention mask over the concatenated [cache | new] tensor of length L_full.
        # mask[b, r, j] = True means key j is invalid for query at row r of request b.
        k_pos = torch.arange(L_full, device=x.device, dtype=torch.long).view(1, 1, -1)  # (1, 1, L_full)
        q_col = (L_prev + arange_t).unsqueeze(-1)                                       # (1, T, 1)

        # Causal mask: within the new-token region, row r at column L_prev+r can only
        # attend to columns <= L_prev+r; cache-region columns (< L_prev) are never
        # causal-violated by this definition.
        mask = k_pos > q_col  # (1, T, L_full)

        if L_prev > 0:
            # Cache-region padding: columns [cache_lens[b], L_prev) are zero-padded and invalid.
            cl_mask = cache_lens.view(-1, 1, 1)  # (B, 1, 1)
            mask_cache_pad = (k_pos < L_prev) & (k_pos >= cl_mask)  # (B, 1, L_full)
            mask = mask | mask_cache_pad

        if pad_lengths is not None:
            # New-region left-padding: columns [L_prev, L_prev+pl[b]) in the new region are
            # pad tokens; real queries must not attend to them. Pad queries (row r < pl[b])
            # intentionally keep their causal prefix unmasked so their attention output is
            # finite — their logits are discarded downstream anyway.
            pl_mask = pad_lengths.view(-1, 1, 1)  # (B, 1, 1)
            real_row = arange_t.unsqueeze(-1) >= pl_mask  # (B, T, 1)
            col_is_new_pad = (k_pos >= L_prev) & ((k_pos - L_prev) < pl_mask)  # (B, 1, L_full)
            mask = mask | (col_is_new_pad & real_row)

        mask = mask.unsqueeze(1)  # (B, 1, T, L_full)
        attn_mask = torch.zeros_like(mask, dtype=x.dtype).masked_fill(mask, float("-inf"))

        new_k_list = []
        new_v_list = []
        for layer_idx, block in enumerate(self.layers):
            pk = prev_k[layer_idx] if prev_k is not None else None
            pv = prev_v[layer_idx] if prev_v is not None else None
            x, nk, nv = block(x, attn_mask, self.cos, self.sin, position_ids, pk, pv)
            new_k_list.append(nk)
            new_v_list.append(nv)

        # Consolidate per-layer outputs into a single stacked tensor so the engine
        # scatter can slice per-request in O(B) Python ops instead of O(B × num_layers).
        new_k = torch.stack(new_k_list, dim=0)
        new_v = torch.stack(new_v_list, dim=0)

        x = self.norm(x)
        logits = self.out_head(x)
        return logits, new_k, new_v
