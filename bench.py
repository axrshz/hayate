import argparse
import statistics
import time

import torch
from transformers import AutoTokenizer

from engine import KVCache
from model import load_pretrained_qwen3_0_6b


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def pick_dtype(device):
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def build_prompt(tokenizer, prompt, use_chat_template):
    if not use_chat_template:
        return prompt
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def run_bench_once(model, input_ids, decode_tokens, device, run_no_cache=True):
    result = {}

    # Prefill + cached decode path
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()
    sync_device(device)
    t0 = time.perf_counter()
    logits = model(input_ids, cache=cache)
    sync_device(device)
    t1 = time.perf_counter()
    prefill_time = t1 - t0
    result["prefill_time_s"] = prefill_time
    result["prefill_toks_per_s"] = input_ids.shape[1] / prefill_time

    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    sync_device(device)
    t0 = time.perf_counter()
    for _ in range(decode_tokens):
        logits = model(next_token, cache=cache)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    sync_device(device)
    t1 = time.perf_counter()
    cached_decode_time = t1 - t0
    result["cached_decode_time_s"] = cached_decode_time
    result["cached_decode_toks_per_s"] = decode_tokens / cached_decode_time

    if run_no_cache:
        tokens = input_ids
        sync_device(device)
        t0 = time.perf_counter()
        for _ in range(decode_tokens):
            logits = model(tokens)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
        sync_device(device)
        t1 = time.perf_counter()
        no_cache_time = t1 - t0
        result["no_cache_decode_time_s"] = no_cache_time
        result["no_cache_decode_toks_per_s"] = decode_tokens / no_cache_time
        result["decode_speedup_x"] = result["cached_decode_toks_per_s"] / result["no_cache_decode_toks_per_s"]

    if device.type == "cuda":
        result["max_cuda_mem_gb"] = torch.cuda.max_memory_allocated(device) / (1024**3)

    return result


def mean(values):
    return statistics.mean(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen3 0.6B prefill/decode speed.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain key-value cache in one paragraph.",
        help="Prompt text used for prefill.",
    )
    parser.add_argument("--repo-id", type=str, default="Qwen/Qwen3-0.6B-Base", help="HF repo id.")
    parser.add_argument("--decode-tokens", type=int, default=64, help="How many decode tokens to benchmark.")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs.")
    parser.add_argument("--chat", action="store_true", help="Use chat template for prompt formatting.")
    parser.add_argument("--cache-dir", type=str, default=None, help="Optional HF local download directory.")
    parser.add_argument("--skip-no-cache", action="store_true", help="Skip slow no-cache decode baseline.")
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
    model.eval()

    prompt_text = build_prompt(tokenizer, args.prompt, args.chat)
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    all_stats = []
    for _ in range(args.runs):
        stats = run_bench_once(
            model=model,
            input_ids=input_ids,
            decode_tokens=args.decode_tokens,
            device=device,
            run_no_cache=not args.skip_no_cache,
        )
        all_stats.append(stats)

    prefill_tps = mean([s["prefill_toks_per_s"] for s in all_stats])
    cached_tps = mean([s["cached_decode_toks_per_s"] for s in all_stats])

    print(f"device: {device}, dtype: {dtype}")
    print(f"prompt_tokens: {input_ids.shape[1]}, decode_tokens: {args.decode_tokens}, runs: {args.runs}")
    print(f"prefill_tokens_per_sec: {prefill_tps:.2f}")
    print(f"cached_decode_tokens_per_sec: {cached_tps:.2f}")

    if not args.skip_no_cache:
        no_cache_tps = mean([s["no_cache_decode_toks_per_s"] for s in all_stats])
        speedup_x = mean([s["decode_speedup_x"] for s in all_stats])
        print(f"no_cache_decode_tokens_per_sec: {no_cache_tps:.2f}")
        print(f"kv_cache_speedup_x: {speedup_x:.2f}")

    if device.type == "cuda":
        mem_gb = mean([s["max_cuda_mem_gb"] for s in all_stats])
        print(f"max_cuda_memory_gb: {mem_gb:.2f}")


if __name__ == "__main__":
    main()
