"""Independent mathematical fixtures and real PyTorch attention comparisons."""

import unittest

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from attention import reference_attention, tiled_attention


class AttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def tensors(self, batch=1, heads=2, kv_heads=None, queries=7, keys=None,
                dim=5, value_dim=3, dtype=torch.float64):
        generator = torch.Generator().manual_seed(71)
        keys = queries if keys is None else keys
        kv_heads = heads if kv_heads is None else kv_heads
        return tuple(torch.randn(shape, generator=generator, dtype=dtype) for shape in (
            (batch, heads, queries, dim), (batch, kv_heads, keys, dim),
            (batch, kv_heads, keys, value_dim)))

    def independent(self, q, k, v, causal, offset):
        mask = None
        if causal:
            mask = torch.arange(k.shape[2])[None, :] <= (torch.arange(q.shape[2])[:, None] + offset)
        k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

    def test_hand_calculated_uniform_and_causal_values(self):
        q = k = torch.zeros((1, 1, 2, 1), dtype=torch.float64)
        v = torch.tensor([[[[2.0], [4.0]]]], dtype=torch.float64)
        for function in (reference_attention, tiled_attention):
            torch.testing.assert_close(function(q, k, v, causal=False),
                torch.tensor([[[[3.0], [3.0]]]], dtype=torch.float64), atol=0, rtol=0)
            torch.testing.assert_close(function(q, k, v, causal=True),
                torch.tensor([[[[2.0], [3.0]]]], dtype=torch.float64), atol=0, rtol=0)

    def test_multiple_shapes_offsets_and_gqa_match_independent_math(self):
        for kwargs, offset in (({}, 0), ({'batch': 2, 'heads': 4, 'queries': 11}, 0),
            ({'heads': 4, 'kv_heads': 2, 'queries': 13}, 0),
            ({'queries': 1, 'keys': 11}, 10), ({'queries': 3, 'keys': 9}, 6)):
            q, k, v = self.tensors(**kwargs)
            for causal in (True, False):
                expected = self.independent(q, k, v, causal, offset)
                with self.subTest(shape=q.shape, causal=causal, offset=offset):
                    torch.testing.assert_close(reference_attention(q, k, v, causal=causal, query_offset=offset), expected,
                                               atol=1e-12, rtol=1e-12)
                    for tile_q, tile_k in ((1, 1), (3, 4), (64, 64)):
                        actual = tiled_attention(q, k, v, causal=causal, query_offset=offset,
                                                 block_q=tile_q, block_k=tile_k)
                        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    def test_future_values_do_not_change_earlier_causal_outputs(self):
        q, k, v = self.tensors(queries=9)
        changed_k, changed_v = k.clone(), v.clone()
        changed_k[:, :, 5:] *= 100
        changed_v[:, :, 5:] += 10000
        for function in (reference_attention, tiled_attention):
            before = function(q, k, v, causal=True)
            after = function(q, changed_k, changed_v, causal=True)
            torch.testing.assert_close(before[:, :, :5], after[:, :, :5], atol=0, rtol=0)

    def test_fully_masked_rows_are_zero_and_do_not_poison_later_rows(self):
        q, k, v = self.tensors(queries=7, keys=9)
        expected = self.independent(q, k, v, True, -3)
        for function in (reference_attention, tiled_attention):
            actual = function(q, k, v, causal=True, query_offset=-3)
            self.assertTrue(torch.isfinite(actual).all())
            self.assertEqual(torch.count_nonzero(actual[:, :, :3]).item(), 0)
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    def test_later_larger_tile_maximum_rescales_old_accumulators(self):
        q = torch.ones((1, 1, 1, 1), dtype=torch.float64)
        k = torch.tensor([[[[-1000.0], [-1001.0], [1000.0], [1001.0]]]], dtype=torch.float64)
        v = torch.tensor([[[[99.0], [98.0], [2.0], [4.0]]]], dtype=torch.float64)
        # The first two weights underflow relative to the second tile's maximum.
        expected = torch.tensor([[[[(2.0 + 4.0 * torch.exp(torch.tensor(1.0, dtype=torch.float64)).item()) /
                                    (1.0 + torch.exp(torch.tensor(1.0, dtype=torch.float64)).item())]]]], dtype=torch.float64)
        actual = tiled_attention(q, k, v, causal=False, block_q=1, block_k=2)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    def test_low_precision_uses_stable_accumulation_and_preserves_output_dtype(self):
        for dtype, tolerance in ((torch.float32, 2e-5), (torch.float16, 2e-3), (torch.bfloat16, 2e-2)):
            q, k, v = self.tensors(dtype=dtype, queries=15, dim=8)
            expected = self.independent(q.double(), k.double(), v.double(), True, 0)
            for function in (reference_attention, tiled_attention):
                output = function(q, k, v, causal=True)
                self.assertEqual(output.dtype, dtype)
                torch.testing.assert_close(output.double(), expected, atol=tolerance, rtol=tolerance)
        q = torch.full((1, 1, 4, 16), 1000.0, dtype=torch.float16)
        k = q.clone()
        v = torch.arange(4, dtype=torch.float16).reshape(1, 1, 4, 1)
        for function in (reference_attention, tiled_attention):
            output = function(q, k, v, causal=False)
            self.assertTrue(torch.isfinite(output).all())
            torch.testing.assert_close(output, torch.full_like(output, 1.5))

    def test_noncontiguous_inputs_work_without_mutation(self):
        q, k, v = self.tensors(queries=8, dim=8, value_dim=8)
        q, k, v = (x.transpose(-1, -2) for x in (q, k, v))
        before = [x.clone() for x in (q, k, v)]
        expected = self.independent(q, k, v, True, 0)
        for function in (reference_attention, tiled_attention):
            torch.testing.assert_close(function(q, k, v), expected, atol=1e-12, rtol=1e-12)
        for x, saved in zip((q, k, v), before):
            self.assertTrue(torch.equal(x, saved))

    def test_invalid_shapes_dtypes_devices_or_tile_sizes_fail(self):
        q, k, v = self.tensors()
        invalid = [(q[0], k, v), (q, k[:, :, :0], v[:, :, :0]),
                   (q, k.float(), v), (q, k, v[:, :, :-1]),
                   (q, k[:, :, :, :-1], v), (q.long(), k.long(), v.long()),
                   (q, k.to('meta'), v)]
        for function in (reference_attention, tiled_attention):
            for values in invalid:
                with self.assertRaises(ValueError):
                    function(*values)
        for tile in (0, -1, True, 2.5):
            with self.assertRaises(ValueError):
                tiled_attention(q, k, v, block_q=tile)
        with self.assertRaises(ValueError):
            tiled_attention(*self.tensors(heads=3, kv_heads=2))


if __name__ == '__main__':
    unittest.main()
