/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <cudf/ast/expressions.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/types.hpp>

#include <rapidsmpf/owning_wrapper.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/node.hpp>

namespace rapidsmpf::streaming {

/**
 * @brief Filter ast expression with lifetime/stream management.
 */
struct Filter {
    rmm::cuda_stream_view stream;  ///< Stream the filter's scalars are valid on.
    cudf::ast::expression& filter;  ///< Filter expression.
    OwningWrapper owner{};  ///< Owner of all objects in the filter.
};

namespace node {
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
 * @param filter Optional filter expression to apply to the read.
 *
 * @return Streaming node representing the asynchronous read.
 */
Node read_parquet(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Channel> ch_out,
    std::size_t num_producers,
    cudf::io::parquet_reader_options options,
    // TODO: use byte count, not row count?
    cudf::size_type num_rows_per_chunk,
    std::unique_ptr<Filter> filter = nullptr
);

/**
 * @brief Asynchronously read parquet files with uniform chunk distribution.
 *
 * Unlike read_parquet which targets a specific number of rows per chunk, this function
 * targets a specific total number of chunks and distributes them uniformly across ranks.
 *
 * When target_num_chunks <= num_files: Files are grouped and read completely
 * (file-aligned). When target_num_chunks > num_files: Files are split into slices,
 * aligned to row groups.
 *
 * @note This is a collective operation, all ranks must participate with identical
 * parameters.
 *
 * @param ctx The execution context to use.
 * @param ch_out Channel to which `TableChunk`s are sent.
 * @param num_producers Number of concurrent producer tasks.
 * @param options Template reader options (same as read_parquet).
 * @param target_num_chunks Target total number of chunks to create across all ranks.
 * @param filter Optional filter expression to apply to the read.
 *
 * @return Streaming node representing the asynchronous read.
 */
Node read_parquet_uniform(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Channel> ch_out,
    std::size_t num_producers,
    cudf::io::parquet_reader_options options,
    std::size_t target_num_chunks,
    std::unique_ptr<Filter> filter = nullptr
);

/**
 * @brief Estimate target chunk count from parquet file metadata.
 *
 * Samples metadata from up to `max_samples` files to estimate total rows,
 * then calculates how many chunks are needed to achieve the target rows per chunk.
 *
 * This is useful for computing the `target_num_chunks` parameter for
 * `read_parquet_uniform` when you have a target `num_rows_per_chunk` instead.
 *
 * @param files List of parquet file paths.
 * @param num_rows_per_chunk Target number of rows per output chunk.
 * @param max_samples Maximum number of files to sample for row estimation.
 *
 * @return Estimated target number of chunks.
 */
std::size_t estimate_target_num_chunks(
    std::vector<std::string> const& files,
    cudf::size_type num_rows_per_chunk,
    std::size_t max_samples = 3
);
}  // namespace node
}  // namespace rapidsmpf::streaming
