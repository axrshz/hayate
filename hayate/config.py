from dataclasses import dataclass


@dataclass
class HayateConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool = True
    attention_bias: bool = True

    @staticmethod
    def _resolve_rope_theta(hf_config) -> float:
        rope_theta = getattr(hf_config, "rope_theta", None)
        if rope_theta is not None:
            return float(rope_theta)
        rope_params = getattr(hf_config, "rope_parameters", None)
        if isinstance(rope_params, dict):
            return float(rope_params.get("rope_theta", 10000.0))
        return 10000.0

    @classmethod
    def from_hf_config(cls, hf_config) -> "HayateConfig":
        head_dim = getattr(hf_config, "head_dim", None)
        if head_dim is None:
            head_dim = hf_config.hidden_size // hf_config.num_attention_heads
        return cls(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=hf_config.num_key_value_heads,
            head_dim=int(head_dim),
            rms_norm_eps=hf_config.rms_norm_eps,
            rope_theta=cls._resolve_rope_theta(hf_config),
            max_position_embeddings=hf_config.max_position_embeddings,
            tie_word_embeddings=getattr(hf_config, "tie_word_embeddings", True),
            attention_bias=getattr(hf_config, "attention_bias", True),
        )
