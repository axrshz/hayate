from dataclasses import dataclass, field
from typing import List

from hayate.model.cache import Cache


@dataclass
class Request:
    id: int = 0
    prompt: str = ""
    max_tokens: int = 100

    prompt_tokens: List[int] = field(default_factory=list)
    tokens: List[int] = field(default_factory=list)
    kv_cache: Cache | None = None

    cache_pos: int = 0
    prefix_cache_len: int = 0
    is_completed: bool = False
    is_prefill: bool = True
    use_cache: bool = True

    response: str | None = None
