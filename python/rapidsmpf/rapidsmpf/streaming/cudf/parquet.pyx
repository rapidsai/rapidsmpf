# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libcpp.memory cimport make_unique, shared_ptr
from libcpp.utility cimport move
from pylibcudf.io.parquet cimport ParquetReaderOptions
from pylibcudf.libcudf.io.parquet cimport parquet_reader_options
from pylibcudf.libcudf.types cimport size_type

from rapidsmpf.streaming.core.channel cimport Channel, cpp_Channel
from rapidsmpf.streaming.core.context cimport Context, cpp_Context
from rapidsmpf.streaming.core.node cimport CppNode, cpp_Node


cdef extern from "<rapidsmpf/streaming/cudf/parquet.hpp>" nogil:
    cdef cpp_Node cpp_read_parquet \
        "rapidsmpf::streaming::node::read_parquet"(
            shared_ptr[cpp_Context] ctx,
            shared_ptr[cpp_Channel] ch_out,
            size_t num_producers,
            parquet_reader_options options,
            size_type num_rows_per_chunk,
        ) except +


def read_parquet(
    Context ctx not None,
    Channel ch_out not None,
    size_t num_producers,
    ParquetReaderOptions options not None,
    size_type num_rows_per_chunk
):
    """
    Create a streaming node to read from parquet.

    Parameters
    ----------
    ctx
        Streaming execution context
    ch_out
        Output channel to receive the TableChunks.
    num_producers
        Number of concurrent producers of output chunks.
    options
        Reader options
    num_rows_per_chunk
        Target (maximum) number of rows per output chunk.

    Notes
    -----
    This is a collective operation, all ranks participating via the
    execution context's communicator must call it with the same options.
    """
    cdef cpp_Node _ret
    with nogil:
        _ret = cpp_read_parquet(
            ctx._handle,
            ch_out._handle,
            num_producers,
            options.c_obj,
            num_rows_per_chunk,
        )
    return CppNode.from_handle(
        make_unique[cpp_Node](move(_ret)), owner=None
    )
