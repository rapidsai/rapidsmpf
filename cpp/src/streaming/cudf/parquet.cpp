/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <optional>
#include <ranges>
#include <unordered_map>
#include <utility>

#include <cudf/ast/expressions.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/io/types.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/memory/memory_type.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/coro_utils.hpp>
#include <rapidsmpf/streaming/core/lineariser.hpp>
#include <rapidsmpf/streaming/core/message.hpp>
#include <rapidsmpf/streaming/core/node.hpp>
#include <rapidsmpf/streaming/core/spillable_messages.hpp>
#include <rapidsmpf/streaming/cudf/parquet.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

namespace rapidsmpf::streaming::node {
namespace {

/**
 * @brief Per-context cache for file-backed messages.
 *
 * FileCache caches file read results by storing message copies in the associated
 * Context's SpillableMessages instance. By tying cached data to the Context, the
 * lifetime of cached entries matches the lifetime of the Context itself.
 *
 * Each cache instance is scoped to a single Context and is shared across callers
 * using that Context.
 *
 * The cache is thread-safe.
 */
class FileCache {
  public:
    struct Key {
        std::vector<std::string> filepaths;
        std::int64_t skip_rows;
        std::size_t skip_bytes;
        std::optional<int64_t> num_rows;
        std::optional<int64_t> num_bytes;
        std::optional<std::vector<std::string>> column_names;
        std::optional<std::vector<cudf::size_type>> column_indices;
        std::vector<std::vector<cudf::size_type>> row_groups;

        // Lexicographical comparison of all data members.
        auto operator<=>(Key const&) const = default;
    };

    /**
     * @brief Construct a FileCache.
     *
     * @param mem_type Memory type used for cache storage.
     */
    FileCache(MemoryType mem_type = MemoryType::HOST) : mem_type_{mem_type} {}

    /**
     * @brief Insert a message into the cache.
     *
     * The message is copied into the memory type configured for this cache
     * and stored in the associated Context's SpillableMessages instance.
     *
     * @param ctx Streaming context.
     * @param key Cache key identifying the message.
     * @param msg Message to cache.
     * @return True if the message was inserted, false if the key already existed.
     */
    bool insert(std::shared_ptr<Context> ctx, Key key, Message const& msg) {
        auto reservation = ctx->br()->reserve_or_fail(msg.copy_cost(), mem_type_);
        auto msg_copy = msg.copy(reservation);

        std::lock_guard lock(mutex_);
        if (cache_.contains(key)) {
            return false;
        }
        cache_.emplace(
            std::move(key), ctx->spillable_messages()->insert(std::move(msg_copy))
        );
        return true;
    }

    /**
     * @brief Retrieve a cached message.
     *
     * If the key exists, the cached message is copied out of spillable storage
     * using newly reserved memory, prioritizing memory types in `MEMORY_TYPES`
     * order.
     *
     * @param ctx Streaming context.
     * @param key Cache key to look up.
     * @return The cached message, or std::nullopt if the key is not present.
     */
    std::optional<Message> get(std::shared_ptr<Context> ctx, Key const& key) const {
        auto& stats = *ctx->statistics();

        auto formatter = [](std::ostream& os, std::size_t count, double val) {
            os << val << "/" << count << " (hits/lookups)";
        };

        SpillableMessages::MessageId mid;
        {
            std::lock_guard lock(mutex_);
            auto it = cache_.find(key);
            if (it == cache_.end()) {
                stats.add_stat("unbounded_file_read_cache hits", 0, formatter);
                return std::nullopt;
            }
            mid = it->second;
        }
        auto const size =
            ctx->spillable_messages()->get_content_description(mid).content_size();

        stats.add_stat("unbounded_file_read_cache hits", 1, formatter);
        stats.add_bytes_stat("unbounded_file_read_cache saved", size);
        auto reservation = ctx->br()->reserve_or_fail(size, MEMORY_TYPES);
        return ctx->spillable_messages()->copy(mid, reservation);
    }

    /**
     * @brief Get the FileCache instance for a Context.
     *
     * Each Context has exactly one FileCache instance for the lifetime of the
     * process. If the `unbounded_file_read_cache` option is disabled, this
     * function returns nullptr.
     *
     * @param ctx Context used to identify the cache instance. The same Context must
     * be used for all subsequent insert and get operations.
     * @return Shared pointer to the per-context FileCache, or nullptr if the cache
     * is disabled.
     */
    static std::shared_ptr<FileCache> instance(std::shared_ptr<Context> ctx) {
        static std::mutex mutex;
        static std::unordered_map<std::size_t, std::shared_ptr<FileCache>> instances;

        std::lock_guard lock(mutex);
        auto const id = ctx->uid();
        auto it = instances.find(id);
        if (it != instances.end()) {
            return it->second;
        }

        // Get the memory type of the file cache, if enabled.
        auto const mem_type = ctx->options().get<std::optional<MemoryType>>(
            "unbounded_file_read_cache", [](auto const& s) -> std::optional<MemoryType> {
                auto val = parse_optional(s);
                if (!val.has_value() || val->empty()) {
                    return std::nullopt;
                }
                return parse_string<MemoryType>(s);
            }
        );

        if (mem_type.has_value()) {
            auto ret = std::make_shared<FileCache>(*mem_type);
            instances.emplace(id, ret);
            return ret;
        }
        return nullptr;
    }

  private:
    mutable std::mutex mutex_;
    std::map<Key, SpillableMessages::MessageId> cache_;
    MemoryType mem_type_;
};

/**
 * @brief Read a single chunk from a parquet source.
 *
 * @param ctx The execution context to use.
 * @param stream The stream on which to read the chunk.
 * @param options The parquet reader options describing the data to read.
 * @param sequence_number The ordered chunk id to reconstruct original ordering of the
 * data.
 * @return Message representing the read chunk.
 */
Message read_parquet_chunk(
    std::shared_ptr<Context> ctx,
    rmm::cuda_stream_view stream,
    cudf::io::parquet_reader_options options,
    std::uint64_t sequence_number
) {
    auto do_read_parquet = [&]() -> Message {
        return to_message(
            sequence_number,
            std::make_unique<TableChunk>(
                cudf::io::read_parquet(options, stream, ctx->br()->device_mr()).tbl,
                stream
            )
        );
    };

    auto file_cache = FileCache::instance(ctx);
    if (file_cache == nullptr) {
        return do_read_parquet();
    }

    FileCache::Key key{
        .filepaths = options.get_source().filepaths(),
        .skip_rows = options.get_skip_rows(),
        .skip_bytes = options.get_skip_bytes(),
        .num_rows = options.get_num_rows(),
        .num_bytes = options.get_num_bytes(),
        .column_names = options.get_column_names(),
        .column_indices = options.get_column_indices(),
        .row_groups = options.get_row_groups()
    };

    auto msg = file_cache->get(ctx, key);
    if (msg.has_value()) {
        return std::move(*msg);
    }

    auto ret = do_read_parquet();
    file_cache->insert(ctx, key, ret);
    return ret;
}

struct ChunkDesc {
    std::uint64_t sequence_number;
    std::int64_t skip_rows;
    std::int64_t num_rows;
    cudf::io::source_info source;
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
    std::shared_ptr<BoundedQueue> ch_out,
    std::vector<ChunkDesc>& chunks,
    cudf::io::parquet_reader_options options
) {
    // ShutdownAtExit c{ch_out};
    co_await ctx->executor()->schedule();
    for (auto& chunk : chunks) {
        cudf::io::parquet_reader_options chunk_options{options};
        chunk_options.set_skip_rows(chunk.skip_rows);
        if (chunk.num_rows >= 0) {
            chunk_options.set_num_rows(chunk.num_rows);
        }
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
    auto const size = safe_cast<std::size_t>(ctx->comm()->nranks());
    auto const rank = safe_cast<std::size_t>(ctx->comm()->rank());
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
    // TODO: Handle case where multiple ranks are reading from a single file.
    auto const files_per_rank =
        safe_cast<int>(files.size() / size + (rank < (files.size() % size)));
    auto const file_offset = safe_cast<int>(
        rank * (files.size() / size) + std::min(rank, files.size() % size)
    );
    auto local_files = std::vector(
        files.begin() + file_offset, files.begin() + file_offset + files_per_rank
    );
    std::uint64_t sequence_number = 0;
    std::vector<std::vector<ChunkDesc>> chunks_per_producer(num_producers);
    auto const num_files = local_files.size();
    // Estimate number of rows per file
    std::size_t files_per_chunk = 1;
    if (num_files > 1) {
        auto nrows =
            cudf::io::read_parquet_metadata(cudf::io::source_info(local_files[0]))
                .num_rows();
        files_per_chunk =
            safe_cast<std::size_t>(std::max(num_rows_per_chunk / nrows, 1l));
    }
    auto to_skip = options.get_skip_rows();
    auto to_read = options.get_num_rows().value_or(std::numeric_limits<int64_t>::max());
    for (std::size_t file_offset = 0; file_offset < num_files;
         file_offset += files_per_chunk)
    {
        std::vector<std::string> chunk_files;
        auto const nchunk_files = std::min(num_files - file_offset, files_per_chunk);
        std::ranges::copy_n(
            local_files.begin() + safe_cast<std::int64_t>(file_offset),
            safe_cast<std::int64_t>(nchunk_files),
            std::back_inserter(chunk_files)
        );
        auto source = cudf::io::source_info(chunk_files);
        // Must read [skip_rows, skip_rows + num_rows) from full fileset
        auto chunk_rows = cudf::io::read_parquet_metadata(source).num_rows() - to_skip;
        auto chunk_skip_rows = to_skip;
        // If the chunk is larger than the number rows we need to skip, on the next
        // iteration we don't need to skip any more rows, otherwise we must skip the
        // remainder.
        to_skip = std::max(0l, -chunk_rows);
        while (chunk_rows > 0 && to_read > 0) {
            auto rows_read =
                std::min({safe_cast<int64_t>(num_rows_per_chunk), chunk_rows, to_read});
            chunks_per_producer[sequence_number % num_producers].emplace_back(
                sequence_number, chunk_skip_rows, rows_read, source
            );
            sequence_number++;
            to_read = std::max(0l, to_read - rows_read);
            chunk_skip_rows += rows_read;
            chunk_rows -= rows_read;
        }
    }
    if (std::ranges::all_of(chunks_per_producer, [](auto&& v) { return v.empty(); })) {
        if (local_files.size() > 0) {
            // If we're on the hook to read some files, but the skip_rows/num_rows setup
            // meant our slice was empty, send an empty table of correct shape.
            // Anyone with no files will just immediately close their output channel.
            auto empty_opts = options;
            empty_opts.set_source(cudf::io::source_info(local_files[0]));
            empty_opts.set_skip_rows(0);
            empty_opts.set_num_rows(0);
            co_await ctx->executor()->schedule(ch_out->send(read_parquet_chunk(
                ctx, ctx->br()->stream_pool().get_stream(), std::move(empty_opts), 0
            )));
        }
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

Node read_parquet_uniform(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Channel> ch_out,
    std::size_t num_producers,
    cudf::io::parquet_reader_options options,
    std::size_t target_num_chunks,
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
    RAPIDSMPF_EXPECTS(
        size == 1 || !options.get_num_rows().has_value(),
        "Reading subset of rows not yet supported in multi-rank execution"
    );
    RAPIDSMPF_EXPECTS(
        size == 1 || options.get_skip_rows() == 0,
        "Skipping rows not yet supported in multi-rank execution"
    );

    auto files = source.filepaths();
    RAPIDSMPF_EXPECTS(files.size() > 0, "Must have at least one file to read");
    RAPIDSMPF_EXPECTS(
        !options.get_filter().has_value(),
        "Do not set filter on options, use the filter argument"
    );

    if (filter != nullptr) {
        options.set_filter(filter->filter);
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
    auto const num_files = files.size();

    // Determine mode: split files or group files
    bool split_mode = (target_num_chunks > num_files);

    if (split_mode) {
        // SPLIT MODE: Each file divided into multiple slices
        // Total chunks = splits_per_file * num_files
        std::size_t splits_per_file = (target_num_chunks + num_files - 1) / num_files;
        std::size_t total_chunks = splits_per_file * num_files;

        // Calculate which chunk IDs this rank owns
        std::size_t chunks_per_rank = (total_chunks + size - 1) / size;
        std::size_t chunk_start = rank * chunks_per_rank;
        std::size_t chunk_end = std::min(chunk_start + chunks_per_rank, total_chunks);

        // Read metadata once per unique file this rank needs and compute row group
        // offsets
        std::unordered_map<std::size_t, std::vector<std::int64_t>> file_rg_offsets_cache;
        for (std::size_t chunk_id = chunk_start; chunk_id < chunk_end; ++chunk_id) {
            std::size_t file_idx = chunk_id / splits_per_file;
            if (file_idx >= num_files)
                continue;

            if (file_rg_offsets_cache.find(file_idx) == file_rg_offsets_cache.end()) {
                auto metadata = cudf::io::read_parquet_metadata(
                    cudf::io::source_info(files[file_idx])
                );
                auto rg_metadata = metadata.rowgroup_metadata();
                std::vector<std::int64_t> rg_offsets;
                rg_offsets.reserve(rg_metadata.size() + 1);
                rg_offsets.push_back(0);
                for (auto const& rg : rg_metadata) {
                    rg_offsets.push_back(rg_offsets.back() + rg.at("num_rows"));
                }
                file_rg_offsets_cache[file_idx] = std::move(rg_offsets);
            }
        }

        // Map chunk IDs to (file_idx, split_idx) with row-group-aligned splits
        for (std::size_t chunk_id = chunk_start; chunk_id < chunk_end; ++chunk_id) {
            std::size_t file_idx = chunk_id / splits_per_file;
            std::size_t split_idx = chunk_id % splits_per_file;

            if (file_idx >= num_files)
                continue;  // Past the end

            const auto& filepath = files[file_idx];
            auto const& rg_offsets = file_rg_offsets_cache[file_idx];
            auto num_row_groups = rg_offsets.size() - 1;
            auto total_rows = rg_offsets.back();

            // Determine slice boundaries aligned to row group boundaries
            std::int64_t skip_rows, num_rows;

            if (splits_per_file <= num_row_groups) {
                // Align to row groups
                std::size_t rg_per_split = num_row_groups / splits_per_file;
                std::size_t extra_rg = num_row_groups % splits_per_file;
                std::size_t rg_start =
                    split_idx * rg_per_split + std::min(split_idx, extra_rg);
                std::size_t rg_end =
                    rg_start + rg_per_split + (split_idx < extra_rg ? 1 : 0);
                skip_rows = rg_offsets[rg_start];
                num_rows = rg_offsets[rg_end] - skip_rows;
            } else {
                // More splits than row groups - split by rows
                std::int64_t rows_per_split =
                    total_rows / static_cast<std::int64_t>(splits_per_file);
                std::int64_t extra_rows =
                    total_rows % static_cast<std::int64_t>(splits_per_file);

                skip_rows = static_cast<std::int64_t>(split_idx) * rows_per_split
                            + std::min(static_cast<std::int64_t>(split_idx), extra_rows);
                num_rows =
                    rows_per_split + (std::cmp_less(split_idx, extra_rows) ? 1 : 0);
            }

            chunks_per_producer[sequence_number % num_producers].emplace_back(
                ChunkDesc{
                    .sequence_number = sequence_number,
                    .skip_rows = skip_rows,
                    .num_rows = num_rows,
                    .source = cudf::io::source_info(filepath)
                }
            );
            sequence_number++;
        }
    } else {
        // GROUP MODE: Multiple files per chunk (file-aligned)
        // Read entire files without needing metadata
        std::size_t files_per_chunk =
            (num_files + target_num_chunks - 1) / target_num_chunks;

        // Calculate which chunk IDs this rank owns
        std::size_t chunks_per_rank = (target_num_chunks + size - 1) / size;
        std::size_t chunk_start = rank * chunks_per_rank;
        std::size_t chunk_end =
            std::min(chunk_start + chunks_per_rank, target_num_chunks);

        // Map chunk IDs to file ranges
        for (std::size_t chunk_id = chunk_start; chunk_id < chunk_end; ++chunk_id) {
            std::size_t file_start = chunk_id * files_per_chunk;
            std::size_t file_end = std::min(file_start + files_per_chunk, num_files);

            if (file_start >= num_files)
                continue;  // Past the end

            // Collect files for this chunk
            std::vector<std::string> chunk_files;
            for (std::size_t file_idx = file_start; file_idx < file_end; ++file_idx) {
                chunk_files.push_back(files[file_idx]);
            }

            // Read entire files - no need for metadata
            // Use -1 for num_rows to read all rows
            chunks_per_producer[sequence_number % num_producers].emplace_back(
                ChunkDesc{
                    .sequence_number = sequence_number,
                    .skip_rows = 0,
                    .num_rows = -1,
                    .source = cudf::io::source_info(chunk_files)
                }
            );
            sequence_number++;
        }
    }

    // Handle empty case
    if (std::ranges::all_of(chunks_per_producer, [](auto&& v) { return v.empty(); })) {
        // No chunks to read - drain and return
        co_await ch_out->drain(ctx->executor());
        if (filter != nullptr) {
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
        co_return;
    }

    // Launch producers
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

    co_await ch_out->drain(ctx->executor());
    if (filter != nullptr) {
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

std::size_t estimate_target_num_chunks(
    std::vector<std::string> const& files,
    cudf::size_type num_rows_per_chunk,
    std::size_t max_samples
) {
    RAPIDSMPF_EXPECTS(files.size() > 0, "Must have at least one file");
    RAPIDSMPF_EXPECTS(num_rows_per_chunk > 0, "num_rows_per_chunk must be positive");

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
    std::int64_t estimated_total_rows =
        (sampled_rows * static_cast<std::int64_t>(files.size()))
        / static_cast<std::int64_t>(sample_files.size());

    // Calculate target chunks (at least 1)
    return std::max(
        std::size_t{1},
        static_cast<std::size_t>(estimated_total_rows / num_rows_per_chunk)
    );
}
}  // namespace rapidsmpf::streaming::node
