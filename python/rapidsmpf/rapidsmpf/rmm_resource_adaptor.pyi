# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource import DeviceMemoryResource
from rmm.pylibrmm.stream import Stream

from rapidsmpf.memory.scoped_memory_record import ScopedMemoryRecord

class RmmResourceAdaptor(DeviceMemoryResource):
    """
    A RMM memory resource adaptor tailored to RapidsMPF.

    .. rubric:: Construction

    This class cannot be constructed directly. Obtain a usable, copyable adaptor
    from a :class:`~rapidsmpf.memory.buffer_resource.BufferResource` via
    ``BufferResource.device_mr_adaptor()``. The returned adaptor holds shared
    ownership of its owning ``BufferResource``.
    """

    def __init__(self, *args: object, **kwargs: object) -> None: ...
    def allocate(self, nbytes: int, stream: Stream = ...) -> int: ...
    def deallocate(self, ptr: int, nbytes: int, stream: Stream = ...) -> None: ...
    def get_main_record(self) -> ScopedMemoryRecord: ...
    @property
    def current_allocated(self) -> int: ...
