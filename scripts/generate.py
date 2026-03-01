import argparse

import torch

from hayate.loader import load_hf_and_hayate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run greedy decoding with Hayate.")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Hugging Face model id")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max tokens to generate")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    tokenizer, _, hayate_model = load_hf_and_hayate(args.model, args.device)

    inputs = tokenizer(args.prompt, return_tensors="pt").input_ids.to(args.device)
    output_ids = hayate_model.greedy_generate(
        input_ids=inputs,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    )
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
