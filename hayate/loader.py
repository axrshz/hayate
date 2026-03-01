from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import HayateConfig
from .model import HayateQwenForCausalLM


def _resolve_dtype(device: str) -> torch.dtype:
    if device.startswith("cuda"):
        return torch.float16
    return torch.float32


def load_hf_and_hayate(
    model_name: str = "Qwen/Qwen3-0.6B",
    device: str = "cpu",
) -> tuple[AutoTokenizer, AutoModelForCausalLM, HayateQwenForCausalLM]:
    dtype = _resolve_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=None,
    )

    config = HayateConfig.from_hf_config(hf_model.config)
    hayate_model = HayateQwenForCausalLM(config)

    missing, unexpected = hayate_model.load_state_dict(hf_model.state_dict(), strict=False)
    if missing:
        raise RuntimeError(
            "Missing required Hayate parameters while loading from Hugging Face.\n"
            f"Missing keys ({len(missing)}): {missing}\n"
        )
    if unexpected:
        print(
            "Warning: ignored extra Hugging Face keys not used by Hayate "
            f"({len(unexpected)} keys)."
        )

    hf_model = hf_model.to(device=device)
    hayate_model = hayate_model.to(device=device, dtype=dtype)
    hf_model.eval()
    hayate_model.eval()

    return tokenizer, hf_model, hayate_model
