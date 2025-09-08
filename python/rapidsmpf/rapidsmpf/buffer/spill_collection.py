# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Collection of objects to spill."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from weakref import WeakValueDictionary

from rapidsmpf.buffer.buffer import MemoryType

if TYPE_CHECKING:
    import rmm
    from rmm.pylibrmm.memory_resource import DeviceMemoryResource
    from rmm.pylibrmm.stream import Stream


@runtime_checkable
class Spillable(Protocol):
    """An interface for spillable objects."""

    def mem_type(self) -> MemoryType:
        """
        Get the memory type of the spillable object.

        Returns
        -------
        The memory type of the object.
        """

    def approx_spillable_amount(self) -> int:
        """
        Get the approximate size of the spillable amount.

        Returns
        -------
        The amount of memory in bytes.
        """

    def spill(
        self,
        amount: int,
        *,
        stream: Stream,
        device_mr: DeviceMemoryResource,
        staging_device_buffer: rmm.DeviceBuffer | None = None,
    ) -> int:
        """
        Spill a specified amount of memory.

        Parameters
        ----------
        amount
            The amount of memory to spill in bytes.
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

    def unspill(self) -> Any:
        """
        Return the unspilled object in device memory.

        The spilled object **will not** be freed. E.g., unspilling a dataframe
        from host to device memory, will not free up host memory. Delete the
        `Spillable` object itself to free up memory.

        Returns
        -------
        The unspilled object now in device memory.
        """


class SpillCollection:
    """A collection of spillable objects that facilitates memory spilling."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._key_counter = 0
        self._spillables: WeakValueDictionary[int, Spillable] = WeakValueDictionary()

    def add_spillable(self, obj: Spillable) -> None:
        """
        Add a spillable object to the collection.

        Parameters
        ----------
        obj
            The spillable object to be added.

        Raises
        ------
        ValueError
            If the object isn't in device memory.
        """
        with self._lock:
            self._key_counter += 1
            self._spillables[self._key_counter] = obj

    def spill(
        self,
        amount: int,
        *,
        stream: Stream,
        device_mr: DeviceMemoryResource,
        staging_device_buffer: rmm.DeviceBuffer | None = None,
    ) -> int:
        """
        Spill memory from device to host until the requested amount is reached.

        This method iterates through spillables and spill them until at least
        the requested amount of memory has been spilled or no more spilling is
        possible.

        Parameters
        ----------
        amount
            The amount of memory to spill in bytes.
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
        The actual amount of memory spilled (in bytes), which may be more, less
        or equal to the requested.

        Raises
        ------
        ValueError
            If the requested spill amount is negative.
        """
        if amount < 0:
            raise ValueError("cannot spill a negative amount")

        # Search through spillables (FIFO ordered) on device and extract objects to spill.
        to_spill: list[Spillable] = []
        to_spill_amount = 0
        with self._lock:
            for _, obj in sorted(self._spillables.items()):
                if obj.mem_type() == MemoryType.DEVICE:
                    to_spill.append(obj)
                    to_spill_amount += obj.approx_spillable_amount()
                    if to_spill_amount >= amount:
                        break

        # Spill the found objects (without the lock).
        spilled_amount = 0
        for obj in to_spill:
            if (residual := amount - spilled_amount) <= 0:
                break
            spilled_amount += obj.spill(
                residual,
                stream=stream,
                device_mr=device_mr,
                staging_device_buffer=staging_device_buffer,
            )
        return spilled_amount
