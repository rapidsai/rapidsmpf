/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cstddef>
#include <memory>

#include <cudf/io/parquet.hpp>
#include <cudf/types.hpp>

#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/node.hpp>

namespace rapidsmpf::streaming::node {

/**
 * @brief Asynchronously read parquet files into an output channel.
 *
 * @note This is a collective operation, all ranks named by the execution context's
 * communicator will participate. All ranks must specify the same set of options.
 * Behaviour is undefined if a `read_parquet` node appears only on a subset of the ranks
 * named by the communicator, or the options differ between ranks.
 *
 * @param ctx The execution context to use.
 * @param ch_out Channel to which `TableChunk`s are sent.
 * @param num_producers Number of concurrent producer tasks.
 * @param options Template reader options. The files within will be picked apart and used
 * to reconstruct new options for each read chunk. The options should therefore specify
 * the read options "as-if" one were reading the whole input in one go.
 * @param num_rows_per_chunk Target (maximum) number of rows any sent `TableChunk` should
 * have.
 *
 * @warning If the options contain a filter then any stream-ordered operations to create
 * scalars must be synchronised before calling this function.
 *
 * @return Streaming node representing the asynchronous read.
 */
Node read_parquet(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Channel> ch_out,
    std::size_t num_producers,
    cudf::io::parquet_reader_options options,
    // TODO: use byte count, not row count?
    cudf::size_type num_rows_per_chunk
);

}  // namespace rapidsmpf::streaming::node
