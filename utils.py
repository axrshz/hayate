import os
from glob import glob

import torch
from safetensors import safe_open
from huggingface_hub import snapshot_download


def pad_to(t, length):
    """pad from the last dimension in pairs of 2"""
    pad = length - t.shape[2]
    if pad == 0:
        return t
    return torch.nn.functional.pad(t, (0, 0, 0, pad))


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


def load_weights(model, repo_id: str):
    """download (if needed) and copy weights from safetensors into the model"""
    local_dir = repo_id.split("/")[-1]
    download_weights(repo_id, local_dir)

    embed_weight_loaded = False
    for file in glob(os.path.join(local_dir, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                _weight_name = weight_name.replace("model.", "")
                try:
                    randn_weight = model.get_parameter(_weight_name)
                except AttributeError:
                    continue
                true_weight = f.get_tensor(weight_name)

                with torch.no_grad():
                    randn_weight.copy_(true_weight)

            if not embed_weight_loaded and "model.embed_tokens.weight" in f.keys():
                randn_weight = model.get_parameter("out_head.weight")
                true_weight = f.get_tensor("model.embed_tokens.weight")
                with torch.no_grad():
                    randn_weight.copy_(true_weight)
                embed_weight_loaded = True