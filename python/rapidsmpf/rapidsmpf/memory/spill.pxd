# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint64_t

from rapidsmpf.memory.buffer_resource cimport BufferResource


cpdef object spill_partitions(
    object partitions,
    BufferResource br,
    uint64_t reservation_context=*,
)
cpdef object unspill_partitions(
    object partitions,
    BufferResource br,
    object allow_overbooking,
    uint64_t reservation_context=*,
)
