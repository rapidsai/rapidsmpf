# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libc.stdint cimport int32_t, uint64_t
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pylibcudf.libcudf.types cimport size_type

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.streaming.core.channel cimport cpp_Channel
from rapidsmpf.streaming.core.context cimport cpp_Context
from rapidsmpf.streaming.core.node cimport cpp_Node


cdef extern from "<rapidsmpf/streaming/cudf/bloom_filter.hpp>" nogil:
    cdef cppclass cpp_BloomFilter "rapidsmpf::streaming::BloomFilter":
        cpp_BloomFilter(
            shared_ptr[cpp_Context] ctx,
            uint64_t seed,
            size_t num_filter_blocks,
        ) noexcept
        cpp_Node build(
            shared_ptr[cpp_Channel] ch_in,
            shared_ptr[cpp_Channel] ch_out,
            int32_t tag,
        ) except +ex_handler
        cpp_Node apply(
            shared_ptr[cpp_Channel] bloom_filter,
            shared_ptr[cpp_Channel] ch_in,
            shared_ptr[cpp_Channel] ch_out,
            vector[size_type] keys,
        ) except +ex_handler


cdef extern from "<rapidsmpf/integrations/cudf/bloom_filter.hpp>" nogil:
    size_t cpp_fitting_num_blocks \
        "rapidsmpf::BloomFilter::fitting_num_blocks"(size_t l2size) noexcept


cdef class BloomFilter:
    """
    Streaming bloom filter construction and application.

    Parameters
    ----------
    ctx
        Streaming context; filter construction is collective over the
        communicator associated with this context.
    seed
        Seed used for hashing values into the bloom filter.
    num_filter_blocks
        Number of blocks used to size the filter.
    """
    cdef unique_ptr[cpp_BloomFilter] _handle
