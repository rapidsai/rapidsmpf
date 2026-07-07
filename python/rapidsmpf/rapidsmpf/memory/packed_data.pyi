# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from rmm.pylibrmm.device_buffer import DeviceBuffer
from rmm.pylibrmm.stream import Stream

from rapidsmpf.memory.buffer_resource import BufferResource

class PackedData:
    def __init__(self) -> None: ...
    @classmethod
    def from_host_bytes(
        cls, data: bytes | bytearray, br: BufferResource
    ) -> PackedData: ...
    @classmethod
    def from_device_buffer(
        cls,
        gpu_data: DeviceBuffer,
        metadata: bytes | bytearray,
        stream: Stream,
        br: BufferResource,
    ) -> PackedData: ...
    def to_host_bytes(self) -> bytes: ...
