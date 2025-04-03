# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Spilling integration for Dask Distributed clusters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar, cast

import dask.sizeof
from distributed import get_worker
from distributed.protocol.cuda import (
    cuda_deserialize,
    cuda_dumps,
    cuda_loads,
    cuda_serialize,
)
from distributed.protocol.serialize import dask_dumps, dask_loads
from distributed.utils import log_errors

from rapidsmp.buffer.buffer import MemoryType

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pylibcudf import gpumemoryview

    from rapidsmp.buffer.spill_collection import SpillCollection

WrappedType = TypeVar("WrappedType")


class SpillableWrapper(Generic[WrappedType]):
    """
    A wrapper that allows objects to be spillable.

    Implements the `rapidsmp.buffer.spill_collection.Spillable` protocol.

    Notes
    -----
    The wrapper is lockfree, which is possible because of two properties:
     1) We only move data from the unspilled state (device memory) to lower
        states (e.g. host memory). On unspill, data is copied (not moved)
        to device memory and a copy of the data is retained in the spilled
        state (e.g. host memory).
     2) No in-place modifications of the wrapped object.

    Parameters
    ----------
    on_device
        The object currently stored on the device. Must be serializable by
        Dask Distributed.
    on_host
        The serialized representation of the object stored in host memory. If
        on_host is provided, it will never become None (not even when unspilled).
    """

    def __init__(
        self,
        *,
        on_device: WrappedType | None = None,
        on_host: tuple[dict, Iterable[memoryview | gpumemoryview]] | None = None,
    ):
        self._on_device = on_device
        self._on_host = on_host
        if on_device is not None:
            # If running on a Worker, add this wrapper to the worker's spill collection,
            # which makes it available for spilling on demand.
            try:
                spill_collection: SpillCollection = get_worker()._rmp_spill_collection
            except ValueError:
                pass
            else:
                spill_collection.add_spillable(self)

    def mem_type(self) -> MemoryType:
        """
        Determine the memory type where the object is currently stored.

        Returns
        -------
        The memory type e.g. `MemoryType.DEVICE` or `MemoryType.HOST`.
        """
        if self._on_device is not None:
            return MemoryType.DEVICE
        else:
            assert self._on_host is not None
            return MemoryType.HOST

    def approx_spillable_amount(self) -> int:
        """
        Estimate the amount of memory that can be spilled.

        Returns
        -------
        The amount of memory in bytes.
        """
        on_device = self._on_device
        if on_device is not None:
            return cast(int, dask.sizeof.sizeof(on_device))
        else:
            return 0

    def spill(self, amount: int) -> int:
        """
        Spill the object from the device to the host.

        Parameters
        ----------
        amount
            The amount of memory in bytes requested to be spilled.

        Returns
        -------
        The actual amount of memory spilled in bytes, which may be more, less,
        or equal to the requested amount.
        """
        if amount > 0:
            on_device = self._on_device
            if on_device is not None:
                ret = self.approx_spillable_amount()
                self._on_host = dask_dumps(on_device)
                self._on_device = None
                return ret
        return 0

    def unspill(self) -> WrappedType:
        """
        Unspill the object, loading it back into device memory.

        The spilled object **will not** be freed. E.g., unspilling a dataframe
        from host to device memory, will not free up host memory. Delete the
        `SpillableWrapper` itself to free up memory.

        Returns
        -------
        The unspilled object now in device memory.
        """
        on_device = self._on_device
        if on_device is not None:
            return on_device
        assert self._on_host is not None
        self._on_device = cast(WrappedType, dask_loads(*self._on_host))
        return self._on_device


def register_dask_serialize() -> None:
    """
    Register dask serialization routines for DataFrames.

    This need to called before Dask can serialize SpillableWrapper objects.
    """

    @cuda_serialize.register(SpillableWrapper)
    def _(x: SpillableWrapper) -> tuple[dict, Iterable[memoryview | gpumemoryview]]:
        with log_errors():
            on_device = x._on_device
            is_on_device = on_device is not None
            if is_on_device:
                sub_header, frames = cuda_dumps(on_device)
            else:
                assert x._on_host is not None
                sub_header, frames = x._on_host
            header = {
                "is_on_device": is_on_device,
                "sub_header": sub_header,
            }
            return header, list(frames)

    @cuda_deserialize.register(SpillableWrapper)
    def _(
        header: dict, frames: Iterable[memoryview | gpumemoryview]
    ) -> SpillableWrapper:
        with log_errors():
            if header["is_on_device"]:
                return SpillableWrapper(
                    on_device=cuda_loads(header["sub_header"], frames)
                )
            else:
                return SpillableWrapper(on_host=(header["sub_header"], frames))
