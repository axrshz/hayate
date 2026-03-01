import argparse

import torch

from hayate.loader import load_hf_and_hayate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Hayate Phase 1 against Hugging Face.")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Hugging Face model id")
    parser.add_argument("--prompt", default="Write a short haiku about wind.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    tokenizer, hf_model, hayate_model = load_hf_and_hayate(args.model, args.device)

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(args.device)

    hf_out = hf_model.generate(
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        use_cache=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    hy_out = hayate_model.greedy_generate(
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    )

    same = torch.equal(hf_out, hy_out)
    print(f"Exact token match: {same}")
    print(f"HF output ids: {hf_out[0].tolist()}")
    print(f"HY output ids: {hy_out[0].tolist()}")

    if not same:
        mismatch_idx = None
        compare_len = min(hf_out.size(1), hy_out.size(1))
        for i in range(compare_len):
            if hf_out[0, i].item() != hy_out[0, i].item():
                mismatch_idx = i
                break

        print(f"First mismatch index: {mismatch_idx}")
        print("HF text:")
        print(tokenizer.decode(hf_out[0], skip_special_tokens=True))
        print("Hayate text:")
        print(tokenizer.decode(hy_out[0], skip_special_tokens=True))
        raise SystemExit(1)

    print("Verification passed.")


if __name__ == "__main__":
    main()
