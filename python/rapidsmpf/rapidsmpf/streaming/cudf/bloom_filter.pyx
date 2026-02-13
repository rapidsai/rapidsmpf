# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference as deref
from libc.stddef cimport size_t
from libc.stdint cimport int32_t, uint64_t
from libcpp.memory cimport make_unique
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pylibcudf.libcudf.types cimport size_type

from rapidsmpf.streaming.core.channel cimport Channel
from rapidsmpf.streaming.core.context cimport Context
from rapidsmpf.streaming.core.node cimport CppNode, cpp_Node


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

    def __init__(
        self,
        Context ctx not None,
        uint64_t seed,
        size_t num_filter_blocks,
    ):
        with nogil:
            self._handle = make_unique[cpp_BloomFilter](
                ctx._handle,
                seed,
                num_filter_blocks,
            )

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    @staticmethod
    def fitting_num_blocks(size_t l2size):
        """
        Return the number of blocks needed to fit within an L2 cache size.

        Parameters
        ----------
        l2size
            Size of the L2 cache in bytes.

        Returns
        -------
        Number of blocks to use in the filter.
        """
        cdef size_t ret
        with nogil:
            ret = cpp_fitting_num_blocks(l2size)
        return ret

    def build(
        self,
        Channel ch_in not None,
        Channel ch_out not None,
        int32_t tag,
    ):
        """
        Build a bloom filter from input table chunks.

        Parameters
        ----------
        ch_in
            Input channel of ``TableChunk`` objects.
        ch_out
            Output channel receiving a single bloom filter message.
        tag
            Disambiguating tag to combine filters across ranks.

        Returns
        -------
        A streaming node representing the asynchronous filter construction.
        """
        cdef cpp_Node _ret
        with nogil:
            _ret = deref(self._handle).build(
                ch_in._handle,
                ch_out._handle,
                tag,
            )
        return CppNode.from_handle(
            make_unique[cpp_Node](move(_ret)), owner=None
        )

    def apply(
        self,
        Channel bloom_filter not None,
        Channel ch_in not None,
        Channel ch_out not None,
        keys,
    ):
        """
        Apply a bloom filter to incoming table chunks.

        Parameters
        ----------
        bloom_filter
            Channel containing the bloom filter (a single message).
        ch_in
            Input channel of ``TableChunk`` objects to filter.
        ch_out
            Output channel receiving filtered ``TableChunk`` objects.
        keys
            Indices selecting the key columns for hash fingerprints.

        Returns
        -------
        A streaming node representing the asynchronous filter application.
        """
        cdef vector[size_type] c_keys = tuple(keys)
        cdef cpp_Node c_ret
        with nogil:
            c_ret = deref(self._handle).apply(
                bloom_filter._handle,
                ch_in._handle,
                ch_out._handle,
                c_keys,
            )
        return CppNode.from_handle(
            make_unique[cpp_Node](move(c_ret)), owner=None
        )
