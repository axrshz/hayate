import torch


def sample_next_token(logits, temperature=1.0, top_k=0):
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    scaled = logits / temperature

    if top_k and top_k > 0:
        k = min(top_k, scaled.size(-1))
        values, _ = torch.topk(scaled, k=k)
        cutoff = values[..., -1, None]
        scaled = torch.where(scaled < cutoff, torch.full_like(scaled, float("-inf")), scaled)

    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1)
