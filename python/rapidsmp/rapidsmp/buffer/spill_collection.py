# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Collection of objects to spill."""

from __future__ import annotations

import threading
from typing import Protocol, runtime_checkable
from weakref import WeakValueDictionary

from rapidsmp.buffer.buffer import MemoryType


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

    def spill(self, amount: int) -> int:
        """
        Spill a specified amount of memory.

        Parameters
        ----------
        amount
            The amount of memory to spill in bytes.

        Returns
        -------
        The actual amount of memory spilled in bytes, which may be more, less,
        or equal to the requested amount.
        """


class SpillCollection:
    """A collection of spillable objects that facilitates memory spilling."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._key_counter = 0
        self._spillables: dict[MemoryType, WeakValueDictionary[int, Spillable]] = {
            MemoryType.DEVICE: WeakValueDictionary(),
            MemoryType.HOST: WeakValueDictionary(),
        }

    def add_spillable(self, obj: Spillable) -> None:
        """
        Add a spillable object to the collection.

        Parameters
        ----------
        obj
            The spillable object to be added.
        """
        memtype = obj.mem_type()
        with self._lock:
            self._key_counter += 1
            self._spillables[memtype][self._key_counter] = obj

    def spill(self, amount: int) -> int:
        """
        Spill memory from device to host until the requested amount is reached.

        This method iterates through spillables and spill them until at least
        the requested amount of memory has been spilled or no more spilling is
        possible.

        Parameters
        ----------
        amount
            The amount of memory to spill in bytes.

        Returns
        -------
        The total amount of memory successfully spilled.

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
            on_device = self._spillables[MemoryType.DEVICE]
            for k, obj in sorted(on_device.items()):
                if obj.mem_type() == MemoryType.DEVICE:
                    to_spill.append(obj)
                    to_spill_amount += obj.approx_spillable_amount()
                    del on_device[k]
                    if to_spill_amount >= amount:
                        break

        # Spill the found objects (without the lock).
        spilled: list[Spillable] = []
        spilled_amount = 0
        for obj in to_spill:
            spilled_amount += obj.spill(amount - spilled_amount)
            spilled.append(obj)
            if spilled_amount >= amount:
                break

        # Add the spilled objects to host memory spillables.
        with self._lock:
            on_host = self._spillables[MemoryType.HOST]
            for obj in spilled:
                self._key_counter += 1
                on_host[self._key_counter] = obj
        return spilled_amount
