# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference as deref
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move


cdef class Node:
    def __init__(self):
        raise ValueError("use the `from_*` factory functions")

    @staticmethod
    cdef Node from_handle(unique_ptr[cpp_Node] handle, object owner):
        cdef Node ret = Node.__new__(Node)
        ret._handle = move(handle)
        ret._owner = owner
        return ret

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    cdef unique_ptr[cpp_Node] handle_release(self):
        if not self._handle:
            raise ValueError("Node is uninitialized, has it been consumed?")
        return move(self._handle)


def run_streaming_pipeline(nodes):
    # Warning, nodes are consumed.
    cdef vector[cpp_Node] _nodes
    for node in nodes:
        _nodes.push_back(move(deref((<Node?>node).handle_release())))

    with nogil:
        cpp_run_streaming_pipeline(move(_nodes))
