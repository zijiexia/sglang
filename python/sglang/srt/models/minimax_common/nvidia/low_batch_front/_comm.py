# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vLLM vllm/models/minimax_m3/nvidia/specialized_decode/kernels/_comm.py
# (symmetric-memory and loopback transports only).
"""Peer-memory transport for the cross-rank "push" kernels.

Every cross-rank exchange of the fused front is a push: a producer kernel
stores its partial (Lamport-tagged words) straight into each peer's receive
buffer through a device pointer, and the consumer polls its own buffer. The
kernels only need, per exchange buffer, the local base pointer and every
rank's base pointer (`peer_ptrs[rank] == local`). This module provides them:

- `SymmMemPeerTransport`: torch symmetric memory over the TP device group
  (TP <= 16 inside one NVLink domain, one or more OS instances). The arena
  also gets an NVLS multicast address when the fabric supports it.
- `LoopbackTransport`: world 1 (single-GPU kernel tests).

`MAX_PEERS` is the peer table of the device structs (`int* p[16]`). Buffers
are carved from one arena per transport, in the same order on every rank
(the allocation calls are collective by construction: every rank builds the
same layers in the same order).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

MAX_PEERS = 16  # the device structs hold ``int* p[16]`` (TP16: four NVL72 trays)
MAX_TRANSPORT_PEERS = MAX_PEERS
TRANSPORT_KINDS = ("symm_mem", "loopback")


@dataclass
class PeerBuffer:
    name: str
    nbytes: int
    rank: int
    world: int
    local: torch.Tensor  # torch.uint8 CUDA tensor over this rank's buffer
    peer_ptrs: tuple[int, ...]  # every rank's base address, [rank] is local
    _dev: torch.Tensor | None = field(default=None, repr=False)
    mc_ptr: int = 0  # NVLS multicast address of the buffer (0: none)

    def peer_ptrs_tensor(self, table: int = MAX_PEERS):
        """int64 [table] on the device (zero padded), for kernels that take
        the peer table by pointer."""
        import torch

        if self._dev is None or self._dev.numel() != table:
            v = self.peer_ptrs_padded(table)
            self._dev = torch.tensor(v, dtype=torch.int64, device=self.local.device)
        return self._dev

    def peer_ptrs_padded(self, table: int = MAX_PEERS) -> list[int]:
        """Every rank's address, zero padded to a device peer table of
        `table` entries; raises if the world does not fit."""
        if len(self.peer_ptrs) > table:
            raise ValueError(
                f"{self.name}: world {len(self.peer_ptrs)} exceeds the device "
                f"peer table ({table})"
            )
        return list(self.peer_ptrs) + [0] * (table - len(self.peer_ptrs))


def _align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


class PeerMemoryError(RuntimeError):
    """Peer memory could not be set up on some rank; raised on every rank
    of the group at the same point (`PeerTransport.agree`)."""


class PeerTransport(ABC):
    """Hands out peer-addressable buffers for one TP group."""

    kind: str = "abstract"
    max_world: int = MAX_TRANSPORT_PEERS

    def __init__(self, rank: int, world: int, capacity: int):
        if not 1 <= world <= self.max_world:
            raise ValueError(f"{self.kind}: world {world} outside 1..{self.max_world}")
        self.rank, self.world, self.capacity = rank, world, capacity
        self._used = 0
        self._bufs: dict[str, PeerBuffer] = {}

    @abstractmethod
    def _base_ptrs(self) -> tuple[int, ...]:
        """Every rank's arena base (collective on first call)."""

    def agree(self, ok: bool, what: str) -> None:
        """Collective: raise `PeerMemoryError` on every rank if `ok` is
        false on any rank. Every rank calls it at the same point, so a local
        failure never leaves the peers waiting in a later collective."""
        if not ok:
            raise PeerMemoryError(f"{what} failed")

    @abstractmethod
    def _local_tensor(self):
        """This rank's arena as a uint8 CUDA tensor."""

    def alloc(self, name: str, nbytes: int, align: int = 256) -> PeerBuffer:
        """A zero-initialized buffer at the same arena offset on every rank."""
        if name in self._bufs:
            b = self._bufs[name]
            if b.nbytes != nbytes:
                raise ValueError(
                    f"{name}: re-requested with {nbytes} B, had {b.nbytes}"
                )
            return b
        off = _align(self._used, align)
        if off + nbytes > self.capacity:
            raise MemoryError(
                f"peer arena ({self.kind}) exhausted: {name} needs {nbytes} B at "
                f"{off}, capacity {self.capacity} B"
            )
        self._used = off + nbytes
        bases = self._base_ptrs()
        local = self._local_tensor().narrow(0, off, nbytes)
        local.zero_()
        mc = self.multicast_ptr
        b = PeerBuffer(
            name,
            nbytes,
            self.rank,
            self.world,
            local,
            tuple(p + off for p in bases),
            mc_ptr=mc + off if mc else 0,
        )
        self._bufs[name] = b
        return b

    @property
    def multicast_ptr(self) -> int:
        """The arena's multicast base address (0: the transport has none)."""
        return 0

    def barrier(self) -> None:
        """Wait for every rank (a no-op for a one-rank transport)."""
        return None

    def close(self) -> None:
        """Release the arena (nothing to release by default)."""
        return None


def _wrap_ptr(ptr: int, nbytes: int, device: int):
    import torch

    class _Buf:
        __cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }

    return torch.as_tensor(_Buf(), device=f"cuda:{device}")


class SymmMemPeerTransport(PeerTransport):
    """torch symmetric memory over the TP device group (single or multi node).

    One `symm_mem.empty` arena per rank, rendezvoused once over `group`
    (a `ProcessGroup` or its name). Within one OS instance torch maps the
    peers with CUDA IPC (POSIX fd) handles; across the trays of an NVL72 it
    uses fabric handles, which need the IMEX channel in the container
    (`/dev/nvidia-caps-imex-channels`). `multicast_ptr` is the arena's
    NVLS multicast address (0 when the fabric does not provide one), at the
    same offsets as `peer_ptrs`."""

    kind = "symm_mem"

    def __init__(self, group, rank: int, world: int, capacity: int):
        super().__init__(rank, world, capacity)
        self._group = group
        self._buf: Any = None
        self._handle: Any = None

    def _ensure(self):
        if self._handle is not None:
            return
        import torch
        import torch.distributed._symmetric_memory as symm_mem

        group = self._group
        name = group if isinstance(group, str) else group.group_name
        enable = getattr(symm_mem, "enable_symm_mem_for_group", None)
        if enable is not None:  # a no-op (or absent) on newer torch
            enable(name)
        # Every rank agrees before and after the rendezvous (a collective), so a local
        # failure raises on every rank instead of leaving peers in it.
        buf = None
        try:
            buf = symm_mem.empty(
                max(self.capacity, 1),
                dtype=torch.uint8,
                device=torch.accelerator.current_device_index(),
            )
            buf.zero_()
        except Exception:
            logger.exception("symm_mem arena: allocation failed on rank %d", self.rank)
        self.agree(buf is not None, "symm_mem arena allocation")
        handle = symm_mem.rendezvous(buf, name)
        # peer_ptrs / multicast_ptr are the allocation's bases: the arena must start
        # there (a pooled sub-allocation would not).
        ok = (
            int(handle.buffer_ptrs[self.rank]) == int(buf.data_ptr())  # type: ignore[union-attr]
            and handle.world_size == self.world
            and handle.rank == self.rank
        )
        if not ok:
            logger.error(
                "symm_mem rendezvous on rank %d: base %#x vs allocation %#x, "
                "rank %d/%d, expected %d/%d",
                self.rank,
                int(handle.buffer_ptrs[self.rank]),
                int(buf.data_ptr()),  # type: ignore[union-attr]
                handle.rank,
                handle.world_size,
                self.rank,
                self.world,
            )
        self.agree(ok, "symm_mem rendezvous")
        self._buf, self._handle = buf, handle
        torch.accelerator.synchronize()
        handle.barrier()

    def _process_group(self):
        if not isinstance(self._group, str):
            return self._group
        from torch.distributed.distributed_c10d import _resolve_process_group

        return _resolve_process_group(self._group)

    def agree(self, ok: bool, what: str) -> None:
        import torch
        import torch.distributed as dist

        flag = torch.tensor(
            [1 if ok else 0],
            dtype=torch.int32,
            device=torch.accelerator.current_device_index(),
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=self._process_group())
        if not int(flag.item()):
            raise PeerMemoryError(f"{what} failed on some rank")

    def _base_ptrs(self):
        self._ensure()
        return tuple(int(p) for p in self._handle.buffer_ptrs)

    def _local_tensor(self):
        self._ensure()
        return self._buf

    @property
    def multicast_ptr(self) -> int:
        self._ensure()
        return int(self._handle.multicast_ptr or 0)

    def barrier(self) -> None:
        self._ensure()
        self._handle.barrier()

    def close(self) -> None:
        if self._handle is None:
            return
        import torch

        torch.accelerator.synchronize()
        self._buf, self._handle = None, None


class LoopbackTransport(PeerTransport):
    """world == 1 (single-GPU kernel tests): the only peer is self."""

    kind = "loopback"
    max_world = 1

    def __init__(self, capacity: int):
        super().__init__(0, 1, capacity)
        self._tensor: Any = None

    def _ensure(self):
        if self._tensor is None:
            import torch

            self._tensor = torch.zeros(
                max(self.capacity, 1), dtype=torch.uint8, device="cuda"
            )

    def _base_ptrs(self):
        self._ensure()
        return (int(self._tensor.data_ptr()),)

    def _local_tensor(self):
        self._ensure()
        return self._tensor


def make_transport(
    kind: str, *, group=None, rank: int, world: int, capacity: int
) -> PeerTransport:
    """`symm_mem` over the device process `group` (world > 1) or
    `loopback` (world 1)."""
    if kind not in TRANSPORT_KINDS:
        raise ValueError(f"unknown peer transport {kind!r}; one of {TRANSPORT_KINDS}")
    if kind == "symm_mem":
        return SymmMemPeerTransport(group, rank, world, capacity)
    return LoopbackTransport(capacity)


__all__ = [
    "PeerMemoryError",
    "MAX_PEERS",
    "MAX_TRANSPORT_PEERS",
    "TRANSPORT_KINDS",
    "LoopbackTransport",
    "PeerBuffer",
    "PeerTransport",
    "SymmMemPeerTransport",
    "make_transport",
]
