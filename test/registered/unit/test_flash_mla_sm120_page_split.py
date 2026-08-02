"""Unit tests for the SM120 sparse-MLA page-split reference mask.

The mask decides which pbs=64 sub-pages the page-split kernel writes. A
sub-page the attention kernel reads but the mask misses would be served from
stale buffer contents, so the index -> sub-page derivation and its
out-of-range handling are pinned here.
"""

import unittest

import torch

from sglang.kernels.ops.attention.flash_mla_sm120 import (
    _PBS_DST,
    fill_referenced_subpage_mask,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestReferencedSubpageMask(CustomTestCase):
    def _mask(self, indices, num_dst_pages: int) -> torch.Tensor:
        mask = torch.zeros(num_dst_pages, dtype=torch.uint8)
        return fill_referenced_subpage_mask(
            mask=mask, token_indices=torch.tensor(indices, dtype=torch.int32)
        )

    def test_token_maps_to_its_subpage(self):
        """Page-split preserves token addressing, so token t lives in
        sub-page t // 64 — the mapping the attention kernel indexes with."""
        self.assertEqual(_PBS_DST, 64)
        mask = self._mask([0, 63, 64, 191, 4096], num_dst_pages=128)
        marked = mask.nonzero().flatten().tolist()
        self.assertEqual(marked, [0, 1, 2, 64])

    def test_unreferenced_subpages_stay_clear(self):
        """The win comes from skipping sub-pages nothing reads: a decode step
        touching a handful of tokens must not mark the whole pool."""
        mask = self._mask([5000, 5001], num_dst_pages=2048)
        self.assertEqual(int(mask.sum()), 2)  # sub-page 0 plus the touched one
        self.assertEqual(mask[5000 // _PBS_DST].item(), 1)

    def test_subpage_zero_is_always_marked(self):
        """decode_dsv4_kernel.cuh gathers slot 0 for every invalid candidate
        and only masks the result after the QK MMA, so slot 0 must hold real
        bytes; leaving it unwritten risks 0 * NaN in the PV stage."""
        mask = self._mask([5000], num_dst_pages=2048)
        self.assertEqual(mask[0].item(), 1)

    def test_out_of_range_padding_is_clamped(self):
        """Padding entries (negative or past the pool) must clamp into range
        rather than scatter out of bounds; marking an extra sub-page is safe,
        an out-of-bounds write is not."""
        num_dst_pages = 16
        mask = self._mask([-1, -999, 10**9, 3 * _PBS_DST], num_dst_pages)
        self.assertEqual(mask.shape[0], num_dst_pages)
        self.assertEqual(mask[0].item(), 1)  # negatives clamp to sub-page 0
        self.assertEqual(mask[num_dst_pages - 1].item(), 1)  # huge clamps to last
        self.assertEqual(mask[3].item(), 1)

    def test_mask_is_reset_between_calls(self):
        """The mask buffer is reused across layers and steps; a stale mark
        would copy a page nothing reads (wasted bandwidth) or, worse, hide a
        regression in this derivation."""
        mask = torch.zeros(64, dtype=torch.uint8)
        fill_referenced_subpage_mask(
            mask=mask, token_indices=torch.tensor([2000], dtype=torch.int32)
        )
        self.assertEqual(mask[2000 // _PBS_DST].item(), 1)
        fill_referenced_subpage_mask(
            mask=mask, token_indices=torch.tensor([100], dtype=torch.int32)
        )
        # The previous step's sub-page is cleared; only slot 0 (always) and
        # this step's sub-page remain.
        self.assertEqual(mask[2000 // _PBS_DST].item(), 0)
        self.assertEqual(mask.nonzero().flatten().tolist(), [0, 100 // _PBS_DST])

    def test_multi_dim_indices_are_flattened(self):
        """Callers pass (batch, topk) indices straight through."""
        mask = self._mask([[0, 64], [128, 192]], num_dst_pages=8)
        self.assertEqual(mask.nonzero().flatten().tolist(), [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
