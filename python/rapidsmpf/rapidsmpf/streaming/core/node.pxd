# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

cdef extern from "<rapidsmpf/streaming/core/node.hpp>" nogil:
    cdef cppclass cpp_Node "rapidsmpf::streaming::Node":
        pass


cdef class Node:
    cdef cpp_Node _handle

    @staticmethod
    cdef Node from_handle(cpp_Node handle)
