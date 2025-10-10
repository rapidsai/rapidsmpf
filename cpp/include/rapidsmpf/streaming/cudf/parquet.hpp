/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cstddef>
#include <memory>

#include <cudf/ast/expressions.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/types.hpp>

#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/node.hpp>

namespace rapidsmpf::streaming::node {

/**
 * @brief Asynchronously read parquet files into an output channel.
 *
 * @param ctx The execution context to use.
 * @param ch_out Channel to which `TableChunk`s are sent.
 * @param max_tickets Maximum number of tickets to throttle production of chunks. Up to
 * this many tasks can start producing data simultaneously.
 * @param options Template reader options. The files within will be picked apart and used
 * to reconstruct new options for each read chunk.
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
    std::ptrdiff_t max_tickets,
    cudf::io::parquet_reader_options options,
    // TODO: use byte count, not row count?
    cudf::size_type num_rows_per_chunk
);

}  // namespace rapidsmpf::streaming::node
