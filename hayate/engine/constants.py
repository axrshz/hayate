import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

COMPILE_MODES = ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs")
MAX_BATCH_SIZE = 20
MAX_DECODE_BATCH = 20
MAX_PREFILL_BATCH = 8
DEFAULT_PREFIX_CACHE_MAX_TOKENS = 4096
