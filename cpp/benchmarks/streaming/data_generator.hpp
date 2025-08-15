/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once


#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/node.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>
#include <rapidsmpf/streaming/cudf/utils.hpp>

#include "../utils/random_data.hpp"

namespace rapidsmpf::streaming::node {

/**
 * @brief Asynchronously generates and sends a sequence of random numeric tables.
 *
 * This is a streaming version of `rapidsmpf::random_table_generator` that operates on
 * table chunks using channels.
 *
 * It creates a specified number of cuDF tables with random `int32_t` values, each
 * consisting of `ncolumns` columns and `nrows` rows. The values are uniformly
 * distributed in the range [`min_val`, `max_val`]. Each generated table is wrapped
 * in a `TableChunk` and sent to the provided output channel in streaming fashion.
 *
 * @param ctx The context to use.
 * @param ch_out Output channel to which generated `TableChunk` objects are sent.
 * @param num_blocks Number of tables (chunks) to generate and send.
 * @param ncolumns Number of columns per generated table.
 * @param nrows Number of rows per column in each table.
 * @param min_val Minimum inclusive value for the generated random integers.
 * @param max_val Maximum inclusive value for the generated random integers.
 *
 * @return A streaming node that completes once all random tables have been generated
 *         and sent, and the channel has been drained.
 */
Node random_table_generator(
    std::shared_ptr<Context> ctx,
    SharedChannel<TableChunk> ch_out,
    std::uint64_t num_blocks,
    cudf::size_type ncolumns,
    cudf::size_type nrows,
    std::int32_t min_val,
    std::int32_t max_val
) {
    ShutdownAtExit c{ch_out};
    co_await ctx->executor()->schedule();
    auto nbytes = static_cast<std::size_t>(ncolumns * nrows) * sizeof(std::int32_t);
    for (std::uint64_t seq = 0; seq < num_blocks; ++seq) {
        auto reservation = ctx->reserve_and_spill(nbytes);
        co_await ch_out->send(
            std::make_unique<TableChunk>(
                seq,
                std::make_unique<cudf::table>(random_table(
                    ncolumns,
                    nrows,
                    min_val,
                    max_val,
                    ctx->stream(),
                    ctx->br()->device_mr()
                ))
            )
        );
    }
    co_await ch_out->drain(ctx->executor());
}


}  // namespace rapidsmpf::streaming::node
