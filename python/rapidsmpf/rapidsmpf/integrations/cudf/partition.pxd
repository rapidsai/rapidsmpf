# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0


from pylibcudf.table cimport Table
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from rapidsmpf.buffer.resource cimport BufferResource


cpdef dict partition_and_pack(
    Table table,
    columns_to_hash,
    int num_partitions,
    BufferResource br,
    stream,
)

cpdef Table unpack_and_concat(
    partitions,
    BufferResource br,
    stream,
)
