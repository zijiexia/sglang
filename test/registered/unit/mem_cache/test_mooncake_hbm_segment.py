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

    def setup(self, *args):
        self.setup_args = args
        return self.ret

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


class TestResolveDeviceName(CustomTestCase):
    def test_resolve(self):
        mapping = '{"0": "mlx5_0", "1": "mlx5_1"}'
        self.assertEqual(ms._resolve_device_name("mlx5_0,mlx5_1", 1), "mlx5_0,mlx5_1")
        self.assertEqual(ms._resolve_device_name(mapping, 1), "mlx5_1")
        self.assertEqual(ms._resolve_device_name(mapping, None), "")
        self.assertEqual(ms._resolve_device_name("{bad", 0), "")


def make_donor(fake):
    config = SimpleNamespace(
        local_hostname="host",
        metadata_server="P2PHANDSHAKE",
        protocol="rdma",
        device_name='{"2": "mlx5_2"}',
        master_server_address="master:50051",
    )
    with (
        patch.object(
            ms.MooncakeHbmDonor, "_import_mooncake_store", return_value=lambda: fake
        ),
        patch.object(ms.MooncakeHbmDonor, "_load_config", return_value=config),
    ):
        return ms.MooncakeHbmDonor(tp_rank=2)


@patch.object(torch.cuda, "current_device", return_value=2)
@patch.object(ms.torch, "empty", side_effect=cpu_empty)
class TestHbmDonor(CustomTestCase):
    def test_donor_mounts_without_host_segment(self, *_):
        fake = FakeStore()
        with envs.MOONCAKE_HBM_SEGMENT_SIZE.override("1gb"):
            donor = make_donor(fake)
        # No global segment and no local buffer: the donor only lends HBM.
        self.assertEqual(
            fake.setup_args,
            ("host", "P2PHANDSHAKE", 0, 0, "rdma", "mlx5_2", "master:50051"),
        )
        ((_, size, protocol, location),) = fake.mounts
        self.assertEqual((size, protocol, location), (1 << 30, "rdma", "cuda:2"))
        donor.close()
        self.assertEqual(fake.unmounts, [["seg-0"]])
        self.assertIsNone(donor._hbm_segment)

    def test_donor_setup_failure_raises(self, *_):
        with envs.MOONCAKE_HBM_SEGMENT_SIZE.override("1gb"):
            with self.assertRaisesRegex(RuntimeError, "HBM donor"):
                make_donor(FakeStore(ret=-1))


class TestSchedulerDonorGate(CustomTestCase):
    """The donor starts only when no store client on the worker mounts HBM."""

    def _run(self, size, linker=False, linker_backend=None, storage_backend=None):
        from sglang.srt.managers import scheduler as sched

        memory = SimpleNamespace(
            hicache_storage_backend=storage_backend,
            unified_cache_external_linker_backend=linker_backend,
        )
        fake_self = SimpleNamespace(enable_unified_cache_external_linker=linker)
        parallel = SimpleNamespace(tp_rank=5)
        with (
            envs.MOONCAKE_HBM_SEGMENT_SIZE.override(size),
            patch.object(sched, "get_memory", return_value=memory),
            patch.object(sched, "get_parallel", return_value=parallel),
            patch.object(ms, "MooncakeHbmDonor") as donor_cls,
            patch.object(sched.atexit, "register"),
        ):
            sched.Scheduler.maybe_init_hbm_donor(fake_self)
        return donor_cls, fake_self.hbm_donor

    def test_off(self):
        donor_cls, donor = self._run("0")
        donor_cls.assert_not_called()
        self.assertIsNone(donor)

    def test_decode_without_store_starts_donor(self):
        donor_cls, donor = self._run("20gb")
        donor_cls.assert_called_once_with(tp_rank=5)
        self.assertIs(donor, donor_cls.return_value)

    def test_store_clients_mount_themselves(self):
        for kwargs in (
            dict(linker=True, linker_backend="mooncake"),
            dict(storage_backend="mooncake"),
        ):
            donor_cls, donor = self._run("20gb", **kwargs)
            donor_cls.assert_not_called()
            self.assertIsNone(donor)

    def test_non_mooncake_store_still_donates(self):
        donor_cls, _ = self._run("20gb", storage_backend="file")
        donor_cls.assert_called_once()


if __name__ == "__main__":
    unittest.main()
