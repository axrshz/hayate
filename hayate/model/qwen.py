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
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta

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

    def initialize_rope(self, device: torch.device | str) -> None:
        """Materialize non-persistent RoPE buffers after meta-device construction."""
        self.cos, self.sin = compute_rope_params(
            head_dim=self.head_dim,
            theta_base=self.rope_theta,
            context_length=self.max_position_embeddings,
            device=device,
        )

    def forward(self, token_ids, prev_k=None, prev_v=None,
                cache_lens=None, pad_lengths=None, use_varlen=False):
        """
        token_ids:    (B, T) int64
        prev_k,
        prev_v:       optional (num_layers, B, H_kv, L_prev, D) stacked prior caches;
                      None on fresh prefill.
        cache_lens:   optional (B,) int64 tensor of per-request valid cache length
                      (values in [0, L_prev]); required when prev_k is not None.
        pad_lengths:  optional (B,) int64 tensor of per-request left-pad lengths in the
                      new-token region; only used during padded prefill.

        Returns: logits (B, 1, V), new_k, new_v (each (num_layers, B, H_kv, T, D)).
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

        varlen_metadata = None
        if use_varlen:
            if pad_lengths is None:
                pad_lengths = torch.zeros(B, device=x.device, dtype=torch.long)
            query_lens = T - pad_lengths
            if cache_lens is None:
                cache_lens = torch.zeros(B, device=x.device, dtype=torch.long)
            key_lens = cache_lens + query_lens

            cu_q = torch.zeros(B + 1, device=x.device, dtype=torch.int32)
            cu_k = torch.zeros(B + 1, device=x.device, dtype=torch.int32)
            cu_q[1:] = query_lens.cumsum(0).to(torch.int32)
            cu_k[1:] = key_lens.cumsum(0).to(torch.int32)

            valid_q = arange_t >= pad_lengths.view(-1, 1)
            cache_pos = torch.arange(L_prev, device=x.device).view(1, -1)
            valid_cache = cache_pos < cache_lens.view(-1, 1)
            valid_k = torch.cat([valid_cache, valid_q], dim=1)
            varlen_metadata = (valid_q, valid_k, cu_q, cu_k, T, L_full, T > 1)

        new_k_list = []
        new_v_list = []
        for layer_idx, block in enumerate(self.layers):
            pk = prev_k[layer_idx] if prev_k is not None else None
            pv = prev_v[layer_idx] if prev_v is not None else None
            x, nk, nv = block(x, self.cos, self.sin, position_ids, pk, pv, varlen_metadata)
            new_k_list.append(nk)
            new_v_list.append(nv)

        # Consolidate per-layer outputs into a single stacked tensor so the engine
        # scatter can slice per-request in O(B) Python ops instead of O(B × num_layers).
        new_k = torch.stack(new_k_list, dim=0)
        new_v = torch.stack(new_v_list, dim=0)

        x = self.norm(x)
        # Generation only consumes the final real token's logits. Avoid projecting
        # every prompt token through the very large vocabulary head during prefill.
        logits = self.out_head(x[:, -1:, :])
        return logits, new_k, new_v
