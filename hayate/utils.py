import os
from glob import glob

import torch
from safetensors import safe_open
from huggingface_hub import snapshot_download


def download_weights(repo_id: str, local_dir: str):
    """download model weights from huggingface if not already present locally"""
    if os.path.isdir(local_dir) and glob(os.path.join(local_dir, "*.safetensors")):
        return
    print(f"Downloading weights from {repo_id} to {local_dir}/ ...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        allow_patterns=["*.safetensors", "*.json", "tokenizer*"],
    )


def assign(left, right, tensor_name="unknown"):
    """copy a checkpoint tensor into a model parameter after validating shape"""
    if left.shape != right.shape:
        raise ValueError(
            f"Shape mismatch in tensor '{tensor_name}'. "
            f"Left: {tuple(left.shape)}, Right: {tuple(right.shape)}"
        )

    with torch.no_grad():
        if isinstance(right, torch.Tensor):
            left.copy_(right.to(dtype=left.dtype, device=left.device))
        else:
            left.copy_(torch.as_tensor(right, dtype=left.dtype, device=left.device))

    return left


def qwen_weight_map(model):
    """map HF checkpoint tensor names to this project's 4B Qwen parameter objects"""
    weight_map = {
        "model.embed_tokens.weight": model.embed_tokens.weight,
        "model.norm.weight": model.norm.weight,
        "lm_head.weight": model.out_head.weight,
    }

    for layer_idx in range(model.num_layers):
        block = model.layers[layer_idx]
        attn = block.self_attn

        weight_map.update(
            {
                f"model.layers.{layer_idx}.self_attn.q_proj.weight": attn.q_proj.weight,
                f"model.layers.{layer_idx}.self_attn.k_proj.weight": attn.k_proj.weight,
                f"model.layers.{layer_idx}.self_attn.v_proj.weight": attn.v_proj.weight,
                f"model.layers.{layer_idx}.self_attn.o_proj.weight": attn.o_proj.weight,
                f"model.layers.{layer_idx}.self_attn.q_norm.weight": attn.q_norm.weight,
                f"model.layers.{layer_idx}.self_attn.k_norm.weight": attn.k_norm.weight,
                f"model.layers.{layer_idx}.input_layernorm.weight": block.input_layernorm.weight,
                f"model.layers.{layer_idx}.mlp.gate_proj.weight": block.mlp.gate_proj.weight,
                f"model.layers.{layer_idx}.mlp.up_proj.weight": block.mlp.up_proj.weight,
                f"model.layers.{layer_idx}.mlp.down_proj.weight": block.mlp.down_proj.weight,
                f"model.layers.{layer_idx}.post_attention_layernorm.weight": block.post_attention_layernorm.weight,
            }
        )

    return weight_map


def load_weights(model, repo_id: str):
    """download (if needed) and explicitly load safetensor shards into the Qwen model"""
    local_dir = repo_id.split("/")[-1]
    download_weights(repo_id, local_dir)

    weight_map = qwen_weight_map(model)
    required_weights = {name for name in weight_map if name != "lm_head.weight"}
    loaded_weights = set()
    lm_head_loaded = False

    for file in sorted(glob(os.path.join(local_dir, "*.safetensors"))):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                if weight_name not in weight_map:
                    continue

                assign(weight_map[weight_name], f.get_tensor(weight_name), weight_name)
                loaded_weights.add(weight_name)

                if weight_name == "lm_head.weight":
                    lm_head_loaded = True

    missing_weights = sorted(required_weights - loaded_weights)
    if missing_weights:
        missing_preview = ", ".join(missing_weights[:5])
        if len(missing_weights) > 5:
            missing_preview += ", ..."
        raise KeyError(f"Missing required checkpoint tensors: {missing_preview}")

    if not lm_head_loaded:
        model.out_head.weight = model.embed_tokens.weight