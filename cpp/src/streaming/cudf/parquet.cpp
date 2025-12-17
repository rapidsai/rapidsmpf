/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>
#include <ranges>

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
    std::int64_t skip_rows;
    std::int64_t num_rows;
    cudf::io::source_info source;
};

/**
 * @brief Estimate total rows by sampling a subset of files.
 *
 * @param files List of all file paths.
 * @param max_samples Maximum number of files to sample.
 *
 * @return Estimated total rows across all files.
 */
std::int64_t estimate_total_rows(
    std::vector<std::string> const& files, std::size_t max_samples = 10
) {
    RAPIDSMPF_EXPECTS(!files.empty(), "Must have at least one file");

    // Sample files with a stride to spread samples evenly across the file list
    std::size_t stride = std::max(std::size_t{1}, files.size() / max_samples);
    std::vector<std::string> sample_files;
    for (std::size_t i = 0; i < files.size() && sample_files.size() < max_samples;
         i += stride)
    {
        sample_files.push_back(files[i]);
    }

    // Read metadata from sampled files to get row count
    auto metadata = cudf::io::read_parquet_metadata(cudf::io::source_info(sample_files));
    std::int64_t sampled_rows = metadata.num_rows();

    // Extrapolate to estimate total rows across all files
    return (sampled_rows * static_cast<std::int64_t>(files.size()))
           / static_cast<std::int64_t>(sample_files.size());
}

/**
 * @brief Structure to hold row group boundary information for a file.
 */
struct FileRowGroupInfo {
    std::vector<std::int64_t> rg_offsets;  ///< Cumulative row offsets for row groups.
    std::int64_t total_rows;  ///< Total rows in the file.
};

/**
 * @brief Read row group metadata for a file and compute cumulative offsets.
 *
 * @param filepath Path to the parquet file.
 *
 * @return FileRowGroupInfo with row group offsets.
 */
FileRowGroupInfo get_file_row_group_info(std::string const& filepath) {
    auto metadata = cudf::io::read_parquet_metadata(cudf::io::source_info(filepath));
    auto rg_metadata = metadata.rowgroup_metadata();
    std::vector<std::int64_t> rg_offsets;
    rg_offsets.reserve(rg_metadata.size() + 1);
    rg_offsets.push_back(0);
    for (auto const& rg : rg_metadata) {
        rg_offsets.push_back(rg_offsets.back() + rg.at("num_rows"));
    }
    auto total_rows = rg_offsets.back();
    return FileRowGroupInfo{
        .rg_offsets = std::move(rg_offsets), .total_rows = total_rows
    };
}

/**
 * @brief Compute row-group-aligned skip_rows and num_rows for a split of a file.
 *
 * Given a file's row group offsets, compute the skip_rows and num_rows for a specific
 * split, aligning to row group boundaries when possible.
 *
 * @param rg_info Row group information for the file.
 * @param split_idx Index of this split (0-based).
 * @param total_splits Total number of splits for this file.
 *
 * @return Pair of (skip_rows, num_rows).
 */
std::pair<std::int64_t, std::int64_t> compute_split_range(
    FileRowGroupInfo const& rg_info, std::size_t split_idx, std::size_t total_splits
) {
    auto const& rg_offsets = rg_info.rg_offsets;
    auto num_row_groups = rg_offsets.size() - 1;
    auto total_rows = rg_info.total_rows;

    std::int64_t skip_rows, num_rows;

    if (total_splits <= num_row_groups) {
        // Align to row groups - distribute row groups evenly across splits
        std::size_t rg_per_split = num_row_groups / total_splits;
        std::size_t extra_rg = num_row_groups % total_splits;
        std::size_t rg_start = split_idx * rg_per_split + std::min(split_idx, extra_rg);
        std::size_t rg_end = rg_start + rg_per_split + (split_idx < extra_rg ? 1 : 0);
        skip_rows = rg_offsets[rg_start];
        num_rows = rg_offsets[rg_end] - skip_rows;
    } else {
        // More splits than row groups - split by rows
        std::int64_t rows_per_split =
            total_rows / static_cast<std::int64_t>(total_splits);
        std::int64_t extra_rows = total_rows % static_cast<std::int64_t>(total_splits);
        auto split_idx_signed = static_cast<std::int64_t>(split_idx);

        skip_rows =
            split_idx_signed * rows_per_split + std::min(split_idx_signed, extra_rows);
        num_rows = rows_per_split + (split_idx_signed < extra_rows ? 1 : 0);
    }

    return {skip_rows, num_rows};
}

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
    std::shared_ptr<BoundedQueue> ch_out,
    std::vector<ChunkDesc>& chunks,
    cudf::io::parquet_reader_options options
) {
    // ShutdownAtExit c{ch_out};
    co_await ctx->executor()->schedule();
    for (auto& chunk : chunks) {
        cudf::io::parquet_reader_options chunk_options{options};
        chunk_options.set_skip_rows(chunk.skip_rows);
        chunk_options.set_num_rows(chunk.num_rows);
        chunk_options.set_source(chunk.source);
        auto stream = ctx->br()->stream_pool().get_stream();
        auto ticket = co_await ch_out->acquire();
        if (!ticket.has_value()) {
            // Semaphore (and hence output channel) shutdown
            break;
        }
        // Having acquire a ticket, let's move to a new thread.
        co_await ctx->executor()->schedule();
        // TODO: This reads the metadata ntasks times.
        // See https://github.com/rapidsai/cudf/issues/20311
        auto [msg, exception] = [&]() -> std::pair<Message, std::exception_ptr> {
            try {
                return {
                    read_parquet_chunk(ctx, stream, chunk_options, chunk.sequence_number),
                    nullptr
                };
            } catch (...) {
                return {Message{}, std::current_exception()};
            }
        }();
        if (exception != nullptr) {
            co_await ch_out->shutdown();
            std::rethrow_exception(exception);
        }
        auto sent = co_await ticket->send(std::move(msg));
        if (!sent) {
            // Output channel is shutdown, no need for more reads.
            break;
        }
    }
    co_await ch_out->drain(ctx->executor());
}
}  // namespace

Node read_parquet(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Channel> ch_out,
    std::size_t num_producers,
    cudf::io::parquet_reader_options options,
    cudf::size_type num_rows_per_chunk,
    std::unique_ptr<Filter> filter
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
    RAPIDSMPF_EXPECTS(!files.empty(), "Must have at least one file to read");
    RAPIDSMPF_EXPECTS(
        files.size() < std::numeric_limits<int>::max(), "Trying to read too many files"
    );
    RAPIDSMPF_EXPECTS(
        !options.get_filter().has_value(),
        "Do not set filter on options, use the filter argument"
    );
    if (filter != nullptr) {
        options.set_filter(filter->filter);
        // Let's just join all the possible streams here rather than inducing cross-stream
        // deps in the tasks
        cuda_stream_join(
            std::ranges::transform_view(
                std::ranges::iota_view(
                    std::size_t{0}, ctx->br()->stream_pool().get_pool_size()
                ),
                [&](auto i) { return ctx->br()->stream_pool().get_stream(i); }
            ),
            std::ranges::single_view(filter->stream)
        );
    }
    std::uint64_t sequence_number = 0;
    std::vector<std::vector<ChunkDesc>> chunks_per_producer(num_producers);
    bool rank_has_assigned_work = false;  // Track if this rank was assigned work

    if (files.size() >= size) {
        // Standard case: at least one file per rank
        // Distribute files evenly across ranks
        std::size_t files_per_rank =
            files.size() / size + ((rank < (files.size() % size)) ? 1 : 0);
        std::size_t file_offset =
            rank * (files.size() / size) + std::min(rank, files.size() % size);
        auto local_files = std::vector(
            files.begin() + static_cast<std::ptrdiff_t>(file_offset),
            files.begin() + static_cast<std::ptrdiff_t>(file_offset + files_per_rank)
        );
        rank_has_assigned_work = !local_files.empty();
        auto const num_files = local_files.size();
        // Estimate number of rows per file
        std::size_t files_per_chunk = 1;
        if (files.size() > 1) {
            auto nrows =
                cudf::io::read_parquet_metadata(cudf::io::source_info(local_files[0]))
                    .num_rows();
            files_per_chunk = static_cast<std::size_t>(
                std::max(num_rows_per_chunk / nrows, std::int64_t{1})
            );
        }
        auto to_skip = options.get_skip_rows();
        auto to_read =
            options.get_num_rows().value_or(std::numeric_limits<int64_t>::max());
        for (std::size_t file_offset = 0; file_offset < num_files;
             file_offset += files_per_chunk)
        {
            std::vector<std::string> chunk_files;
            auto const nchunk_files = std::min(num_files - file_offset, files_per_chunk);
            std::ranges::copy_n(
                local_files.begin() + static_cast<std::int64_t>(file_offset),
                static_cast<std::int64_t>(nchunk_files),
                std::back_inserter(chunk_files)
            );
            auto source = cudf::io::source_info(chunk_files);
            // Must read [skip_rows, skip_rows + num_rows) from full fileset
            auto chunk_rows =
                cudf::io::read_parquet_metadata(source).num_rows() - to_skip;
            auto chunk_skip_rows = to_skip;
            // If the chunk is larger than the number rows we need to skip, on the next
            // iteration we don't need to skip any more rows, otherwise we must skip the
            // remainder.
            to_skip = std::max(std::int64_t{0}, -chunk_rows);
            while (chunk_rows > 0 && to_read > 0) {
                auto rows_read = std::min(
                    {static_cast<std::int64_t>(num_rows_per_chunk), chunk_rows, to_read}
                );
                chunks_per_producer[sequence_number % num_producers].emplace_back(
                    sequence_number, chunk_skip_rows, rows_read, source
                );
                sequence_number++;
                to_read = std::max(std::int64_t{0}, to_read - rows_read);
                chunk_skip_rows += rows_read;
                chunk_rows -= rows_read;
            }
        }
    } else {
        // Multi-rank single-file case: fewer files than ranks
        // Use sampling to estimate chunks and distribute work across ranks
        auto const num_files = files.size();

        // For single file, read metadata once and reuse; otherwise sample
        std::optional<FileRowGroupInfo> single_file_info = std::nullopt;
        std::int64_t estimated_total_rows = 0;
        if (num_files == 1) {
            // Single file: read metadata once, use for both estimation and splits
            single_file_info = get_file_row_group_info(files[0]);
            estimated_total_rows = single_file_info->total_rows;
        } else {
            // Multiple files: sample to estimate
            estimated_total_rows = estimate_total_rows(files);
        }

        // Estimate total chunks and splits per file
        auto estimated_total_chunks = std::max(
            std::size_t{1},
            static_cast<std::size_t>(estimated_total_rows / num_rows_per_chunk)
        );
        auto splits_per_file =
            (estimated_total_chunks + num_files - 1) / num_files;  // Round up
        auto total_splits = num_files * splits_per_file;

        // Distribute split indices across ranks (only use as many ranks as we have
        // splits)
        auto active_ranks = std::min(size, total_splits);
        rank_has_assigned_work = (rank < active_ranks);
        if (rank_has_assigned_work) {
            // Distribute splits evenly across active ranks
            auto splits_per_rank = total_splits / active_ranks;
            auto extra_splits = total_splits % active_ranks;
            auto split_start = rank * splits_per_rank + std::min(rank, extra_splits);
            auto split_end =
                split_start + splits_per_rank + (rank < extra_splits ? 1 : 0);

            // Process each split assigned to this rank
            // Track which file we're currently working on to avoid re-reading metadata
            std::size_t current_file_idx = std::numeric_limits<std::size_t>::max();
            FileRowGroupInfo current_file_info;

            for (auto split_idx = split_start; split_idx < split_end; ++split_idx) {
                auto file_idx = split_idx / splits_per_file;
                auto local_split_idx = split_idx % splits_per_file;

                if (file_idx >= num_files) {
                    // Past the end of files (can happen with rounding)
                    break;
                }

                // Read file metadata if we haven't already for this file
                if (file_idx != current_file_idx) {
                    current_file_idx = file_idx;
                    // Reuse cached metadata for single-file case
                    if (single_file_info.has_value() && file_idx == 0) {
                        current_file_info = *single_file_info;
                    } else {
                        current_file_info = get_file_row_group_info(files[file_idx]);
                    }
                }

                // Compute row-group-aligned range for this split
                auto [skip_rows, total_rows_for_split] = compute_split_range(
                    current_file_info, local_split_idx, splits_per_file
                );

                if (total_rows_for_split <= 0) {
                    continue;
                }

                // Produce chunks of num_rows_per_chunk from this split's row range
                auto source = cudf::io::source_info(files[file_idx]);
                auto chunk_skip = skip_rows;
                auto remaining = total_rows_for_split;

                while (remaining > 0) {
                    auto chunk_rows = std::min(
                        static_cast<std::int64_t>(num_rows_per_chunk), remaining
                    );
                    chunks_per_producer[sequence_number % num_producers].emplace_back(
                        ChunkDesc{
                            .sequence_number = sequence_number,
                            .skip_rows = chunk_skip,
                            .num_rows = chunk_rows,
                            .source = source
                        }
                    );
                    sequence_number++;
                    chunk_skip += chunk_rows;
                    remaining -= chunk_rows;
                }
            }
        }
    }
    if (std::ranges::all_of(chunks_per_producer, [](auto&& v) { return v.empty(); })) {
        if (rank_has_assigned_work) {
            // If we're on the hook to read some files, but the skip_rows/num_rows setup
            // meant our slice was empty, send an empty table of correct shape.
            // Use the first file to get the schema for the empty table.
            auto empty_opts = options;
            empty_opts.set_source(cudf::io::source_info(files[0]));
            empty_opts.set_skip_rows(0);
            empty_opts.set_num_rows(0);
            co_await ctx->executor()->schedule(ch_out->send(read_parquet_chunk(
                ctx, ctx->br()->stream_pool().get_stream(), std::move(empty_opts), 0
            )));
        }
        // Ranks without assigned work just close their output channel without sending
    } else {
        std::vector<Node> read_tasks;
        read_tasks.reserve(1 + num_producers);
        auto lineariser = Lineariser(ctx, ch_out, num_producers);
        auto queues = lineariser.get_queues();
        for (std::size_t i = 0; i < num_producers; i++) {
            read_tasks.push_back(
                produce_chunks(ctx, queues[i], chunks_per_producer[i], options)
            );
        }
        read_tasks.push_back(lineariser.drain());
        coro_results(co_await coro::when_all(std::move(read_tasks)));
    }
    co_await ch_out->drain(ctx->executor());
    if (filter != nullptr) {
        // Let's just join all the possible streams here rather than inducing cross-stream
        // deps in the tasks
        cuda_stream_join(
            std::ranges::single_view(filter->stream),
            std::ranges::transform_view(
                std::ranges::iota_view(
                    std::size_t{0}, ctx->br()->stream_pool().get_pool_size()
                ),
                [&](auto i) { return ctx->br()->stream_pool().get_stream(i); }
            )
        );
    }
}
}  // namespace rapidsmpf::streaming::node
