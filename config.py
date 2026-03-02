from dataclasses import asdict, dataclass

import torch


@dataclass
class Qwen3Config:
    vocab_size: int = 151_936
    context_length: int = 40_960
    emb_dim: int = 1024
    n_heads: int = 16
    n_layers: int = 28
    hidden_dim: int = 3072
    head_dim: int = 128
    qk_norm: bool = True
    n_kv_groups: int = 8
    rope_base: float = 1_000_000.0
    dtype: torch.dtype = torch.bfloat16

    def to_dict(self) -> dict:
        return asdict(self)


QWEN3_0_6B_CONFIG = Qwen3Config()
