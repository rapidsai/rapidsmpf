/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <cudf/ast/expressions.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/io/types.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/coro_utils.hpp>
#include <rapidsmpf/streaming/core/node.hpp>
#include <rapidsmpf/streaming/cudf/parquet.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

namespace rapidsmpf::streaming::node {
namespace {

/**
 * @brief Read a single chunk from a parquet source and send it to an output channel.
 *
 * @param ctx The execution context to use.
 * @param ch_out Channel to which `TableChunk`s are sent.
 * @param stream The stream on which to read the chunk.
 * @param options The parquet reader options describing the data to read.
 * @param sequence_number The ordered chunk id to reconstruct original ordering of the
 * data.
 *
 * @note The caller is responsible for scheduling this coroutine onto a thread pool for
 * execution.
 *
 * @return Streaming node representing the asynchronous read of a chunk and send to the
 * output channel.
 */
Node read_parquet_chunk(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<ThrottlingAdaptor> ch_out,
    rmm::cuda_stream_view stream,
    cudf::io::parquet_reader_options options,
    std::uint64_t sequence_number
) {
    auto ticket = co_await ch_out->acquire();
    auto result = std::make_unique<TableChunk>(
        sequence_number,
        cudf::io::read_parquet(options, stream, ctx->br()->device_mr()).tbl,
        stream
    );
    auto [_, receipt] = co_await ticket.send(std::move(result));
    // Move this coroutine to the back of the queue so that when we release the semaphore
    // it is likely to occur on a different thread, releasing the semaphore resumes any
    // waiters on the current thread which is not what we typically want for throttled
    // reads, we want the next waiting read task to run on a different thread.
    co_await ctx->executor()->yield();
    co_await receipt;
}
}  // namespace

Node read_parquet(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Channel> ch_out,
    std::ptrdiff_t max_tickets,
    cudf::io::parquet_reader_options options,
    cudf::size_type num_rows_per_chunk
) {
    ShutdownAtExit c{ch_out};
    auto throttle = std::make_shared<ThrottlingAdaptor>(ch_out, max_tickets);
    co_await ctx->executor()->schedule();
    auto size = static_cast<std::size_t>(ctx->comm()->nranks());
    auto rank = static_cast<std::size_t>(ctx->comm()->rank());
    auto source = options.get_source();
    RAPIDSMPF_EXPECTS(
        source.type() == cudf::io::io_type::FILEPATH, "Only implemented for file sources"
    );
    // TODO: To handle this we need a prefix scan across all the ranks of the total number
    // of rows that would be read by previous ranks.
    RAPIDSMPF_EXPECTS(
        size == 1 || !options.get_num_rows().has_value(),
        "Reading subset of rows not yet supported in multi-rank execution"
    );
    // TODO: To handle this we need a prefix scan across all the ranks of the total number
    // of rows that would be read by previous ranks.
    RAPIDSMPF_EXPECTS(
        size == 1 || options.get_skip_rows() == 0,
        "Skipping rows not yet supported in multi-rank execution"
    );
    auto files = source.filepaths();
    RAPIDSMPF_EXPECTS(files.size() > 0, "Must have at least one file to read");
    RAPIDSMPF_EXPECTS(
        files.size() < std::numeric_limits<int>::max(), "Trying to read too many files"
    );
    // TODO: Handle case where multiple ranks are reading from a single file.
    int files_per_rank =
        static_cast<int>(files.size() / size + (rank < (files.size() % size)));
    int file_offset = rank * (files.size() / size) + std::min(rank, files.size() % size);
    auto local_files = std::vector(
        files.begin() + file_offset, files.begin() + file_offset + files_per_rank
    );
    cudf::io::parquet_reader_options local_options{options};
    local_options.set_source(cudf::io::source_info(local_files));
    auto metadata = cudf::io::read_parquet_metadata(local_options.get_source());
    auto const local_num_rows = metadata.num_rows();
    auto skip_rows = options.get_skip_rows();
    auto num_rows_to_read =
        options.get_num_rows().value_or(std::numeric_limits<int64_t>::max());
    if ((num_rows_to_read == 0 && rank == 0)
        || (skip_rows >= local_num_rows && rank == size - 1))
    {
        // If we're reading nothing, rank zero sends an empty table of correct
        // shape/schema and everyone else sends nothing. Similarly, if we skipped
        // everything in the file and we're the last rank, send an empty table, otherwise
        // send nothing.
        cudf::io::parquet_reader_options empty_opts{options};
        empty_opts.set_source(cudf::io::source_info{options.get_source().filepaths()[0]});
        empty_opts.set_skip_rows(0);
        empty_opts.set_num_rows(0);
        co_await ctx->executor()->schedule(read_parquet_chunk(
            ctx, throttle, ctx->br()->stream_pool().get_stream(), std::move(empty_opts), 0
        ));
    } else {
        std::uint64_t sequence_number = 0;
        std::vector<Node> read_tasks;
        while (skip_rows < local_num_rows && num_rows_to_read > 0) {
            cudf::size_type chunk_num_rows = std::min(
                {static_cast<std::int64_t>(num_rows_per_chunk),
                 local_num_rows - skip_rows,
                 num_rows_to_read}
            );
            num_rows_to_read -= chunk_num_rows;
            cudf::io::parquet_reader_options chunk_options{local_options};
            chunk_options.set_skip_rows(skip_rows);
            chunk_options.set_num_rows(chunk_num_rows);
            // TODO: This reads the metadata ntasks times.
            // See https://github.com/rapidsai/cudf/issues/20311
            read_tasks.push_back(ctx->executor()->schedule(read_parquet_chunk(
                ctx,
                throttle,
                ctx->br()->stream_pool().get_stream(),
                std::move(chunk_options),
                // TODO: sequence number being correct relies on read_parquet_chunk
                // sending only one chunk.
                sequence_number++
            )));
            skip_rows += chunk_num_rows;
        }
        coro_results(co_await coro::when_all(std::move(read_tasks)));
    }
    co_await ch_out->drain(ctx->executor());
}
}  // namespace rapidsmpf::streaming::node
