import argparse

import torch
from transformers import AutoTokenizer

from engine import KVCache
from model import load_pretrained_qwen3_0_6b
from sampler import sample_next_token


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def pick_dtype(device):
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


@torch.inference_mode()
def generate(model, input_ids, max_new_tokens, temperature, top_k, eos_token_id=None):
    tokens = input_ids
    generated = []

    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()
    logits = model(tokens, cache=cache)

    for _ in range(max_new_tokens):
        next_logits = logits[:, -1, :]
        next_token = sample_next_token(next_logits, temperature=temperature, top_k=top_k)
        generated.append(next_token)

        if eos_token_id is not None and torch.all(next_token.squeeze(-1) == eos_token_id):
            break
        logits = model(next_token, cache=cache)

    if generated:
        return torch.cat(generated, dim=1)
    return input_ids.new_empty((input_ids.size(0), 0))


def build_prompt(tokenizer, prompt, use_chat_template):
    if not use_chat_template:
        return prompt
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    parser = argparse.ArgumentParser(description="Minimal Qwen3 0.6B inference with custom model.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text.")
    parser.add_argument("--repo-id", type=str, default="Qwen/Qwen3-0.6B-Base", help="HF repo id.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature; <=0 means greedy.")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k for sampling; 0 disables top-k.")
    parser.add_argument("--chat", action="store_true", help="Use tokenizer chat template.")
    parser.add_argument("--cache-dir", type=str, default=None, help="Optional HF local download directory.")
    args = parser.parse_args()

    device = pick_device()
    dtype = pick_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(args.repo_id, trust_remote_code=True)
    model = load_pretrained_qwen3_0_6b(
        repo_id=args.repo_id,
        dtype=dtype,
        device=device,
        local_dir=args.cache_dir,
    )

    prompt_text = build_prompt(tokenizer, args.prompt, args.chat)
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)

    output_ids = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_token_id=tokenizer.eos_token_id,
    )

    completion_ids = output_ids[0]
    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=False)
    print(completion_text)


if __name__ == "__main__":
    main()
