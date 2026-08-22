# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from rapidsmpf.memory.buffer_resource cimport BufferResource


cpdef object spill_partitions(object partitions, BufferResource br)
cpdef object unspill_partitions(
    object partitions,
    BufferResource br,
    object allow_overbooking,
)
