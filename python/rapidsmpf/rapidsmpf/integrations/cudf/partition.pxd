# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0


from libcpp cimport bool as bool_t
from pylibcudf.table cimport Table
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from rapidsmpf.buffer.resource cimport BufferResource
from rapidsmpf.statistics cimport Statistics


cpdef dict partition_and_pack(
    Table table,
    columns_to_hash,
    int num_partitions,
    Stream stream,
    BufferResource br,
)

cpdef Table unpack_and_concat(
    partitions,
    Stream stream,
    BufferResource br,
)

cpdef list spill_partitions(
    partitions,
    Stream stream,
    BufferResource br,
    Statistics statistics = *,
)

cpdef list unspill_partitions(
    partitions,
    Stream stream,
    BufferResource br,
    bool_t allow_overbooking,
    Statistics statistics = *,
)
