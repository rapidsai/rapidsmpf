# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Spilling integration for Dask Distributed clusters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar, cast

import dask.sizeof
from distributed.protocol.cuda import (
    cuda_deserialize,
    cuda_dumps,
    cuda_loads,
    cuda_serialize,
)
from distributed.protocol.serialize import dask_dumps, dask_loads
from distributed.utils import log_errors

from rapidsmpf.buffer.buffer import MemoryType
from rapidsmpf.integrations.dask.core import get_worker_context

if TYPE_CHECKING:
    from collections.abc import Iterable

    import rmm
    from pylibcudf import gpumemoryview
    from rmm.pylibrmm.memory_resource import DeviceMemoryResource
    from rmm.pylibrmm.stream import Stream

    from rapidsmpf.buffer.spill_collection import SpillCollection

WrappedType = TypeVar("WrappedType")


class SpillableWrapper(Generic[WrappedType]):
    """
    A lockfree wrapper that allows objects to be spillable.

    The wrapped object is immutable and to allow spilling, no external
    references should exist to the object. If such references do exist,
    modifying the object results in undefined behavior.

    Implements the `rapidsmpf.buffer.spill_collection.Spillable` protocol.

    Notes
    -----
    The following properties are maintained:
    1. No modifications of the wrapped object are allowed.
    2. Data transitions only from a higher to a lower state (e.g., device
       to host memory). During unspill, data is copied (not moved) back to
       device memory while retaining a copy in the spilled state.
    3. A copy of the wrapped object (spilled or unspilled) is always retained
       before deletion from a higher state.

    These properties ensure the wrapper remains lock-free:
    - Property (1) allows multiple copies without coherency concerns.
    - Properties (2) and (3) guarantee that a valid copy can always be retrieved
      by searching lower states, eliminating the need for locking.

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
                spill_collection: SpillCollection = (
                    get_worker_context().spill_collection
                )
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

    def spill(
        self,
        amount: int,
        *,
        stream: Stream,
        device_mr: DeviceMemoryResource,
        staging_device_buffer: rmm.DeviceBuffer | None = None,
    ) -> int:
        """
        Spill the object from the device to the host.

        Parameters
        ----------
        amount
            The amount of memory in bytes requested to be spilled.
        stream
            The CUDA stream on which to perform the spill operation.
        device_mr
            The memory resource used for device memory allocation and deallocation.
        staging_device_buffer
            An optional preallocated device buffer that can be used as temporary
            staging space during the spill operation. If not provided, a new buffer
            may be allocated internally.

        Returns
        -------
        The actual amount of memory spilled in bytes, which may be more, less,
        or equal to the requested amount.
        """
        if amount > 0:
            on_device = self._on_device
            if on_device is not None and self._on_host is None:
                ret = self.approx_spillable_amount()
                self._on_host = dask_dumps(
                    on_device,
                    context={
                        "stream": stream,
                        "device_mr": device_mr,
                        "staging_device_buffer": staging_device_buffer,
                    },
                )
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
