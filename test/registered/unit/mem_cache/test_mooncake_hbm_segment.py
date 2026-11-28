"""Unit tests for mounting donated HBM as a Mooncake store segment."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.storage.mooncake_store import mooncake_store as ms
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ALIGN = 4 << 20


class FakeStore:
    def __init__(self, ret=0, has_api=True):
        self.ret = ret
        self.mounts = []
        self.unmounts = []
        if has_api:
            self.mount_external_segment = self._mount

    def _mount(self, buffer, size, protocol, location):
        self.mounts.append((buffer, size, protocol, location))
        return {"ret": self.ret, "segment_ids": ["seg-0"] if self.ret == 0 else []}

    def unmount_segment(self, ids):
        self.unmounts.append(list(ids))
        return 0


def make_store(fake):
    store = object.__new__(ms.MooncakeStore)
    store.store = fake
    store.config = SimpleNamespace(protocol="rdma")
    return store


_real_empty = torch.empty


def cpu_empty(size, dtype, device):
    # Stand-in for a CUDA allocation; the test only checks pointers.
    return _real_empty(size, dtype=dtype)


class TestHbmSegmentSize(CustomTestCase):
    def test_parse(self):
        cases = {"0": 0, "": 0, "1gb": 1 << 30, str(ALIGN + 123): ALIGN}
        for value, expected in cases.items():
            with envs.MOONCAKE_HBM_SEGMENT_SIZE.override(value):
                self.assertEqual(ms.hbm_segment_bytes(), expected, value)


@patch.object(torch.cuda, "current_device", return_value=3)
@patch.object(ms.torch, "empty", side_effect=cpu_empty)
class TestMountHbmSegment(CustomTestCase):
    def test_disabled_does_nothing(self, *_):
        fake = FakeStore()
        store = make_store(fake)
        with envs.MOONCAKE_HBM_SEGMENT_SIZE.override("0"):
            store._mount_hbm_segment()
        self.assertEqual(fake.mounts, [])
        store.close()
        self.assertEqual(fake.unmounts, [])

    def test_mount_aligned_and_unmount_on_close(self, *_):
        fake = FakeStore()
        store = make_store(fake)
        with envs.MOONCAKE_HBM_SEGMENT_SIZE.override("1gb"):
            store._mount_hbm_segment()
        ((base, size, protocol, location),) = fake.mounts
        self.assertEqual(base % ALIGN, 0)
        self.assertEqual((size, protocol, location), (1 << 30, "rdma", "cuda:3"))
        ptr = store._hbm_segment.data_ptr()
        self.assertTrue(ptr <= base and base + size <= ptr + store._hbm_segment.numel())
        store.close()
        self.assertEqual(fake.unmounts, [["seg-0"]])
        self.assertIsNone(store._hbm_segment)

    def test_missing_api_raises(self, *_):
        store = make_store(FakeStore(has_api=False))
        with envs.MOONCAKE_HBM_SEGMENT_SIZE.override("1gb"):
            with self.assertRaisesRegex(RuntimeError, "mount_external_segment"):
                store._mount_hbm_segment()

    def test_mount_failure_raises(self, *_):
        store = make_store(FakeStore(ret=-1))
        with envs.MOONCAKE_HBM_SEGMENT_SIZE.override("1gb"):
            with self.assertRaisesRegex(RuntimeError, "Failed to mount"):
                store._mount_hbm_segment()


if __name__ == "__main__":
    unittest.main()
