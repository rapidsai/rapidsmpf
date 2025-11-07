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
#include <rapidsmpf/streaming/core/lineariser.hpp>
#include <rapidsmpf/streaming/core/node.hpp>
#include <rapidsmpf/streaming/cudf/parquet.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

namespace rapidsmpf::streaming::node {
namespace {

/**
 * @brief Read a single chunk from a parquet source.
 *
 * @param ctx The execution context to use.
 * @param stream The stream on which to read the chunk.
 * @param options The parquet reader options describing the data to read.
 * @param sequence_number The ordered chunk id to reconstruct original ordering of the
 * data.
 *
 * @return Message representing the read chunk.
 */
Message read_parquet_chunk(
    std::shared_ptr<Context> ctx,
    rmm::cuda_stream_view stream,
    cudf::io::parquet_reader_options options,
    std::uint64_t sequence_number
) {
    auto result = std::make_unique<TableChunk>(
        cudf::io::read_parquet(options, stream, ctx->br()->device_mr()).tbl, stream
    );
    return to_message(sequence_number, std::move(result));
}

struct ChunkDesc {
    std::uint64_t sequence_number;
    std::vector<std::string> files;
};

/**
 * @brief Read chunks and send them to an output channel.
 *
 * @param ctx Execution context to use.
 * @param ch_out Channel to send output to.
 * @param options Template reader options.
 * @param chunks List of chunks from the input files to read. Processed in order.
 * @param idx Index of the next chunk to process.
 *
 * @return Coroutine representing the processing of all chunks.
 */
Node produce_chunks(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Channel> ch_out,
    Semaphore& ticket,
    cudf::io::parquet_reader_options options,
    std::vector<ChunkDesc>& chunks
) {
    ShutdownAtExit c{ch_out};
    for (auto& chunk : chunks) {
        cudf::io::parquet_reader_options chunk_options{options};
        // chunk_options.set_skip_rows(chunk.skip_rows);
        // chunk_options.set_num_rows(chunk.num_rows);
        chunk_options.set_source(cudf::io::source_info(std::move(chunk.files)));
        auto stream = ctx->br()->stream_pool().get_stream();
        // TODO: This reads the metadata ntasks times.
        // See https://github.com/rapidsai/cudf/issues/20311
        co_await ticket.acquire();
        co_await ctx->executor()->schedule();
        co_await ch_out->send(
            read_parquet_chunk(ctx, stream, chunk_options, chunk.sequence_number)
        );
    }
    co_await ch_out->drain(ctx->executor());
}
}  // namespace

Node read_parquet(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Channel> ch_out,
    std::size_t num_producers,
    cudf::io::parquet_reader_options options,
    [[maybe_unused]] cudf::size_type num_rows_per_chunk
) {
    ShutdownAtExit c{ch_out};
    co_await ctx->executor()->schedule();
    auto size = static_cast<std::size_t>(ctx->comm()->nranks());
    auto rank = static_cast<std::size_t>(ctx->comm()->rank());
    auto source = options.get_source();
    RAPIDSMPF_EXPECTS(
        source.type() == cudf::io::io_type::FILEPATH, "Only implemented for file sources"
    );
    // TODO: To handle this we need a prefix scan across all the ranks of the total
    // number of rows that would be read by previous ranks.
    RAPIDSMPF_EXPECTS(
        size == 1 || !options.get_num_rows().has_value(),
        "Reading subset of rows not yet supported in multi-rank execution"
    );
    // TODO: To handle this we need a prefix scan across all the ranks of the total
    // number of rows that would be read by previous ranks.
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
    if (local_files.empty()) {
        co_return;
    }

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
        // everything in the file and we're the last rank, send an empty table,
        // otherwise send nothing.
        cudf::io::parquet_reader_options empty_opts{options};
        empty_opts.set_source(cudf::io::source_info{options.get_source().filepaths()[0]});
        empty_opts.set_skip_rows(0);
        empty_opts.set_num_rows(0);
        co_await ctx->executor()->schedule(ch_out->send(read_parquet_chunk(
            ctx, ctx->br()->stream_pool().get_stream(), std::move(empty_opts), 0
        )));
    } else {
        std::uint64_t sequence_number = 0;
        std::vector<std::vector<ChunkDesc>> chunks_per_producer(num_producers);
        std::size_t nfiles_done = 0;
        while (nfiles_done < local_files.size()) {
            // TODO: this is a total hack.
            std::vector<std::string> chunk_files;
            auto to_do = std::min(local_files.size() - nfiles_done, std::size_t{6});
            std::ranges::copy_n(
                local_files.begin() + static_cast<std::ptrdiff_t>(nfiles_done),
                static_cast<std::ptrdiff_t>(to_do),
                std::back_inserter(chunk_files)
            );
            chunks_per_producer[sequence_number % num_producers].emplace_back(
                sequence_number, std::move(chunk_files)
            );
            nfiles_done += to_do;
            sequence_number++;
        }
        // while (skip_rows < local_num_rows && num_rows_to_read > 0) {
        //     auto chunk_num_rows = std::min(
        //         {static_cast<std::int64_t>(num_rows_per_chunk),
        //          local_num_rows - skip_rows,
        //          num_rows_to_read}
        //     );
        //     num_rows_to_read -= chunk_num_rows;
        //     chunks_per_producer[sequence_number % num_producers].emplace_back(
        //         sequence_number, skip_rows, chunk_num_rows
        //     );
        //     skip_rows += chunk_num_rows;
        //     sequence_number++;
        // }
        std::vector<Node> read_tasks;
        read_tasks.reserve(1 + num_producers);
        auto lineariser = Lineariser(ctx, ch_out, num_producers);
        auto& inputs = lineariser.get_inputs();
        for (std::size_t i = 0; i < num_producers; i++) {
            read_tasks.push_back(produce_chunks(
                ctx,
                inputs[i].second,
                *inputs[i].first,
                local_options,
                chunks_per_producer[i]
            ));
        }
        read_tasks.push_back(lineariser.drain());
        coro_results(co_await coro::when_all(std::move(read_tasks)));
    }
    co_await ch_out->drain(ctx->executor());
}
}  // namespace rapidsmpf::streaming::node
