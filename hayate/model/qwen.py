from typing import List, Optional

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

    def forward(self, token_ids, caches, start_positions, pad_lengths=None):
        batch_size, num_tokens = token_ids.shape
        x = self.embed_tokens(token_ids)

        # Build position_ids: (B, num_tokens) with correct absolute positions
        positions = torch.arange(num_tokens, device=x.device, dtype=torch.long).unsqueeze(0)
        sp_tensor = torch.tensor(start_positions, device=x.device, dtype=torch.long).unsqueeze(1)
        position_ids = positions + sp_tensor  # (B, num_tokens)

        if pad_lengths is not None:
            position_ids = position_ids.clone()
            for i, pl in enumerate(pad_lengths):
                if pl > 0:
                    position_ids[i, :pl] = 0
                    position_ids[i, pl:] = torch.arange(num_tokens - pl, device=x.device)

        # Causal mask
        max_seq = max(start_positions) + num_tokens
        full_mask = torch.triu(
            torch.ones(max_seq, max_seq, device=x.device, dtype=torch.bool),
            diagonal=1
        )

        mask_rows = []
        for i, sp in enumerate(start_positions):
            row = full_mask[sp : sp + num_tokens, :max_seq]
            mask_rows.append(row)
        mask = torch.stack(mask_rows, dim=0).unsqueeze(1)

        # Block real tokens from attending to left-pad positions
        if pad_lengths is not None:
            for i, pl in enumerate(pad_lengths):
                if pl > 0:
                    mask[i, :, pl:, :pl] = True

        for layer_idx, block in enumerate(self.layers):
            layer_caches = [(c.get(layer_idx) if c is not None else None) for c in caches]
            x, new_layer_caches = block(x, mask, self.cos, self.sin, position_ids, layer_caches)

            for i, c in enumerate(caches):
                if c is not None:
                    c.update(layer_idx, new_layer_caches[i])

        x = self.norm(x)
        logits = self.out_head(x.to(torch.bfloat16))
        return logits