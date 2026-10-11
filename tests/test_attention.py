import unittest

import torch

from hayate.model.attention import GroupedQueryAttention
from hayate.model.rope import compute_rope_params


class CachedChunkAttentionTests(unittest.TestCase):
    """Check causal attention when a contiguous cache precedes a token chunk."""

    def test_cached_chunk_matches_incremental_attention(self):
        # Each token in a cached chunk must see all history and earlier chunk tokens.
        torch.manual_seed(19)
        dtype = torch.float32
        module = GroupedQueryAttention(32, 4, 2, 8, dtype=dtype).eval()
        history_len = 3
        query_len = 4
        batch_size = 2
        x = torch.randn(batch_size, query_len, 32, dtype=dtype)
        prev_k = torch.randn(batch_size, 2, history_len, 8, dtype=dtype)
        prev_v = torch.randn_like(prev_k)
        positions = torch.arange(history_len, history_len + query_len).expand(
            batch_size, -1
        )
        cos, sin = compute_rope_params(8, context_length=16)

        with torch.inference_mode():
            actual, new_k, new_v = module(
                x, cos, sin, positions, prev_k, prev_v
            )
            expected = []
            for query_idx in range(query_len):
                keys = torch.cat((prev_k, new_k[:, :, :query_idx, :]), dim=2)
                values = torch.cat((prev_v, new_v[:, :, :query_idx, :]), dim=2)
                output, _, _ = module(
                    x[:, query_idx : query_idx + 1],
                    cos,
                    sin,
                    positions[:, query_idx : query_idx + 1],
                    keys,
                    values,
                )
                expected.append(output)

        torch.testing.assert_close(actual, torch.cat(expected, dim=1))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA varlen attention required")
class VariableLengthAttentionTests(unittest.TestCase):
    """Check packed variable-length attention against separate request runs."""

    def test_packed_cached_chunks_match_incremental_attention(self):
        # Uneven cache histories must remain causal when each request adds a chunk.
        torch.manual_seed(11)
        dtype = torch.bfloat16
        module = GroupedQueryAttention(64, 4, 2, 16, dtype=dtype).cuda().eval()
        cache_lens = torch.tensor([2, 4], device="cuda", dtype=torch.long)
        query_len = 3
        max_cache_len = int(cache_lens.max())
        x = torch.randn(2, query_len, 64, device="cuda", dtype=dtype)
        prev_k = torch.randn(2, 2, max_cache_len, 16, device="cuda", dtype=dtype)
        prev_v = torch.randn_like(prev_k)

        query_lens = torch.full_like(cache_lens, query_len)
        key_lens = cache_lens + query_lens
        cu_q = torch.cat(
            (
                torch.zeros(1, device="cuda", dtype=torch.int32),
                query_lens.cumsum(0).to(torch.int32),
            )
        )
        cu_k = torch.cat(
            (
                torch.zeros(1, device="cuda", dtype=torch.int32),
                key_lens.cumsum(0).to(torch.int32),
            )
        )
        valid_q = torch.ones(2, query_len, device="cuda", dtype=torch.bool)
        cache_positions = torch.arange(max_cache_len, device="cuda").view(1, -1)
        valid_cache = cache_positions < cache_lens.view(-1, 1)
        valid_k = torch.cat((valid_cache, valid_q), dim=1)
        metadata = (
            valid_q,
            valid_k,
            cu_q,
            cu_k,
            query_len,
            max_cache_len + query_len,
            True,
        )
        positions = cache_lens.view(-1, 1) + torch.arange(query_len, device="cuda")
        cos, sin = compute_rope_params(16, context_length=16, device="cuda")

        with torch.inference_mode():
            actual, new_k, new_v = module(
                x, cos, sin, positions, prev_k, prev_v, metadata
            )
            for row, history_len in enumerate(cache_lens.tolist()):
                expected = []
                for query_idx in range(query_len):
                    keys = torch.cat(
                        (
                            prev_k[row : row + 1, :, :history_len],
                            new_k[row : row + 1, :, :query_idx],
                        ),
                        dim=2,
                    )
                    values = torch.cat(
                        (
                            prev_v[row : row + 1, :, :history_len],
                            new_v[row : row + 1, :, :query_idx],
                        ),
                        dim=2,
                    )
                    output, _, _ = module(
                        x[row : row + 1, query_idx : query_idx + 1],
                        cos,
                        sin,
                        positions[row : row + 1, query_idx : query_idx + 1],
                        keys,
                        values,
                    )
                    expected.append(output)
                torch.testing.assert_close(
                    actual[row : row + 1],
                    torch.cat(expected, dim=1),
                    rtol=2e-2,
                    atol=2e-2,
                )

    def test_packed_batch_matches_individual_flash_attention(self):
        # Use unequal prompt lengths so the packed path must ignore left padding.
        torch.manual_seed(7)
        dtype = torch.bfloat16
        module = GroupedQueryAttention(64, 4, 2, 16, dtype=dtype).cuda().eval()
        lengths = torch.tensor([3, 5], device="cuda", dtype=torch.long)
        pads = 5 - lengths
        x = torch.randn(2, 5, 64, device="cuda", dtype=dtype)
        positions = (torch.arange(5, device="cuda").view(1, -1) - pads.view(-1, 1)).clamp(min=0)
        valid = torch.arange(5, device="cuda").view(1, -1) >= pads.view(-1, 1)
        cu = torch.tensor([0, 3, 8], device="cuda", dtype=torch.int32)
        metadata = (valid, valid, cu, cu, 5, 5, True)
        cos, sin = compute_rope_params(16, context_length=16, device="cuda")

        with torch.inference_mode():
            actual, _, _ = module(x, cos, sin, positions, None, None, metadata)
            # Compare each request's real output positions with an unpadded run.
            for index, length in enumerate(lengths.tolist()):
                start = 5 - length
                expected, _, _ = module(
                    x[index : index + 1, start:],
                    cos,
                    sin,
                    torch.arange(length, device="cuda").view(1, -1),
                    None,
                    None,
                )
                torch.testing.assert_close(
                    actual[index, start:], expected[0], rtol=2e-2, atol=2e-2
                )


if __name__ == "__main__":
    unittest.main()
