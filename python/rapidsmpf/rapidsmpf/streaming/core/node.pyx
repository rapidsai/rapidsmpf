# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp.utility cimport move


cdef class Node:
    @staticmethod
    cdef Node from_handle(cpp_Node handle):
        cdef Node ret = Node.__new__(Node)
        ret._handle = move(handle)
        return ret
