# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ProgressThread interface for RapidsMPF."""

from __future__ import annotations

from typing import NamedTuple

from cython.operator cimport dereference as deref
from libcpp.memory cimport make_shared
from libcpp.vector cimport vector

from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.statistics cimport Statistics


class TransferEvent(NamedTuple):
    """A completed, data-bearing collective receive."""

    op_id: int
    collective_kind: CollectiveKind
    source_rank: int
    destination_rank: int
    message_id: int
    metadata_bytes: int
    payload_bytes: int
    destination_memory_type: MemoryType
    completion_timestamp_ns: int


cdef class ProgressThread:
    """
    A progress thread that can execute arbitrary functions.

    The `ProgressThread` class provides an interface for executing arbitrary
    functions in a separate thread. The functions are executed in the order they
    were registered, and a newly registered function will only execute for the
    first time in the next iteration of the progress thread.

    Parameters
    ----------
    statistics
        The statistics instance to use. If None, statistics is disabled.

    Notes
    -----
    This class is designed to handle background tasks and progress tracking in
    distributed operations. It is typically used in conjunction with the
    `Shuffler` class to track progress of data movement operations.
    """
    def __init__(
        self,
        Statistics statistics = None,
    ):
        if statistics is None:
            statistics = Statistics(enable=False)  # Disables statistics.

        with nogil:
            self._handle = make_shared[cpp_ProgressThread](statistics._handle)

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    @property
    def statistics(self):
        """
        The statistics object for this progress thread.

        Returns
        -------
        Statistics
           The statistics.
        """
        return Statistics.from_handle(deref(self._handle).statistics())

    def enable_transfer_events(self, size_t capacity=65536):
        """Enable bounded collective receive-event recording.

        This starts a new recording interval, clearing queued events and the dropped
        event counter.
        """
        with nogil:
            deref(self._handle).enable_transfer_events(capacity)

    def disable_transfer_events(self):
        """Disable recording while preserving queued events for draining."""
        with nogil:
            deref(self._handle).disable_transfer_events()

    def drain_transfer_events(self):
        """Move all currently queued receive events into a Python list."""
        cdef vector[cpp_TransferEvent] events
        cdef cpp_TransferEvent event
        cdef list result = []
        with nogil:
            events = deref(self._handle).drain_transfer_events()
        for event in events:
            result.append(
                TransferEvent(
                    event.op_id,
                    event.collective_kind,
                    event.source_rank,
                    event.destination_rank,
                    event.message_id,
                    event.metadata_bytes,
                    event.payload_bytes,
                    event.destination_memory_type,
                    event.completion_timestamp_ns,
                )
            )
        return result

    @property
    def dropped_transfer_events(self):
        """Number of receive events dropped in the current recording interval."""
        return deref(self._handle).dropped_transfer_events()
