# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t

from dataclasses import dataclass

from rapidsmpf.buffer.buffer import MemoryType


cdef extern from * nogil:
    """
    #include <rapidsmpf/buffer/content_description.hpp>
    rapidsmpf::ContentDescription cpp_create_empty_content_description(
        bool spillable
    ) {
        return rapidsmpf::ContentDescription{
            spillable?
                rapidsmpf::ContentDescription::Spillable::YES:
                rapidsmpf::ContentDescription::Spillable::NO
        };
    }
    void cpp_content_description_set_size(
        rapidsmpf::ContentDescription cd,
        rapidsmpf::MemoryType mem_type,
        std::size_t size
    ) {
        cd.content_size(mem_type) = size;
    }
    """
    cpp_ContentDescription cpp_create_empty_content_description(
        bool_t spillable
    ) except +
    void cpp_content_description_set_size(
        cpp_ContentDescription cd, cpp_MemoryType mem_type, size_t size
    ) noexcept


cdef cpp_ContentDescription content_description_to_cpp(object cd):
    assert isinstance(cd, ContentDescription)
    cdef cpp_ContentDescription ret = cpp_create_empty_content_description(
        cd.spillable()
    )
    for mem_type, size in cd.content_sizes:
        cpp_content_description_set_size(ret, <cpp_MemoryType?>mem_type, size)
    return ret


cdef content_description_from_cpp(cpp_ContentDescription cd):
    cdef dict content_sizes = {}
    for mem_type in MemoryType:
        content_sizes[mem_type] = cd.content_size(mem_type)
    return ContentDescription(content_sizes, cd.spillable())


@dataclass
class ContentDescription:
    content_sizes: dict[MemoryType, int]
    spillable: bool
