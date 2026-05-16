import torch

from hayate.engine.request import Request


class Sampler:
    """Greedy sampler with stop-token finalization."""

    def __init__(self, tokenizer, stop_token_ids: set[int]):
        self.tokenizer = tokenizer
        self.stop_token_ids = stop_token_ids

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Select the next token greedily via argmax."""
        return torch.argmax(logits, dim=-1, keepdim=True)

    def finalize(self, request: Request, tok: int) -> None:
        """Append a sampled token and complete the request if it hit a stop condition."""
        request.tokens.append(tok)
        if tok in self.stop_token_ids or len(request.tokens) >= request.max_tokens:
            request.is_completed = True
            request.response = self.tokenizer.decode(request.tokens, skip_special_tokens=True)
