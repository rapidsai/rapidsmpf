/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 *
 * TPC-H Query 18 - Pre-filter Optimization
 *
 * Usage:
 *   # Single GPU or small scale factors (SF1-SF100):
 *   q18 --input-directory /path/to/tpch/data --output-file result.parquet
 *
 *   # Multi-GPU with large scale factors (SF1000+):
 *   mpirun -np 4 q18 --input-directory /path/to/tpch/data \
 *       --output-file result.parquet --use-shuffle
 *
 * Key Options:
 *   --use-shuffle    Use shuffle-based distributed joins instead of all-gather.
 *                    REQUIRED for large scale factors (SF1000+) on multi-GPU to
 *                    avoid memory pressure from all-gathering large intermediate
 *                    tables to every rank.
 *
 *   --spill-device-limit <ratio>
 *                    Fraction of GPU memory before spilling to host (default: 0.8).
 *                    Use lower values (e.g., 0.5) for memory-constrained systems.
 *
 * Algorithm:
 *   This benchmark implements Q18 with a two-phase approach that exploits the
 *   high selectivity of the "sum(l_quantity) > 300" filter (~0.004% of orders).
 *
 *   Phase 1 (blocking): Compute qualifying orderkeys
 *     - Read lineitem -> groupby(l_orderkey, sum(l_quantity)) -> filter(sum > 300)
 *     - With --use-shuffle: shuffle by orderkey for parallel aggregation
 *     - Without --use-shuffle: all-gather partial aggregates (redundant work)
 *     - Result: ~57 orderkeys at SF1, ~57K at SF1000, ~171K at SF3000
 *
 *   Phase 2 (streaming): Pre-filter and join
 *     - Read lineitem/orders -> semi-join filter using qualifying orderkeys
 *     - With --use-shuffle: shuffle filtered data for parallel joins
 *     - Without --use-shuffle: all-gather filtered data (works for small results)
 *     - Join with customer -> groupby -> sort -> write top 100
 *
 * When to use --use-shuffle:
 *   - Multi-GPU runs at SF1000+: The intermediate filtered tables (~228K lineitem
 *     rows, ~57K orders rows at SF1000) become too large to all-gather efficiently.
 *   - Memory-constrained systems: Shuffle distributes memory pressure across ranks.
 *
 * When NOT to use --use-shuffle:
 *   - Single GPU: No benefit from shuffle overhead.
 *   - Small scale factors (SF1-SF100): All-gather is faster for tiny results.
 *
 * Disclaimers:
 *   - The two-phase approach corresponds to "advanced" query optimization.
 *   - Re-reading lineitem may not be optimal with slow remote storage.
 */

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>

#include <cuda_runtime_api.h>
#include <getopt.h>
#include <mpi.h>

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/context.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/join/filtered_join.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/communicator/mpi.hpp>
#include <rapidsmpf/communicator/ucxx.hpp>
#include <rapidsmpf/communicator/ucxx_utils.hpp>
#include <rapidsmpf/config.hpp>
#include <rapidsmpf/cuda_event.hpp>
#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/integrations/cudf/partition.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/nvtx.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/streaming/coll/allgather.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/leaf_node.hpp>
#include <rapidsmpf/streaming/core/node.hpp>
#include <rapidsmpf/streaming/cudf/parquet.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

#include "concatenate.hpp"
#include "join.hpp"
#include "utils.hpp"

namespace {

// ============================================================================
// Utility Functions
// ============================================================================

// NOTE: This is added to ndsh::detail in https://github.com/rapidsai/rapidsmpf/pull/710
std::string get_table_path(
    std::string const& input_directory, std::string const& table_name
) {
    auto dir = input_directory.empty() ? "." : input_directory;
    auto file_path = dir + "/" + table_name + ".parquet";
    if (std::filesystem::exists(file_path)) {
        return file_path;
    }
    return dir + "/" + table_name + "/";
}

// ============================================================================
// Table Readers
// ============================================================================

rapidsmpf::streaming::Node read_lineitem(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    std::size_t num_producers,
    cudf::size_type num_rows_per_chunk,
    std::string const& input_directory
) {
    auto files = rapidsmpf::ndsh::detail::list_parquet_files(
        get_table_path(input_directory, "lineitem")
    );
    auto options = cudf::io::parquet_reader_options::builder(cudf::io::source_info(files))
                       .columns({"l_orderkey", "l_quantity"})
                       .build();
    return rapidsmpf::streaming::node::read_parquet(
        ctx, ch_out, num_producers, options, num_rows_per_chunk
    );
}

rapidsmpf::streaming::Node read_orders(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    std::size_t num_producers,
    cudf::size_type num_rows_per_chunk,
    std::string const& input_directory
) {
    auto files = rapidsmpf::ndsh::detail::list_parquet_files(
        get_table_path(input_directory, "orders")
    );
    auto options =
        cudf::io::parquet_reader_options::builder(cudf::io::source_info(files))
            .columns({"o_orderkey", "o_custkey", "o_orderdate", "o_totalprice"})
            .build();
    return rapidsmpf::streaming::node::read_parquet(
        ctx, ch_out, num_producers, options, num_rows_per_chunk
    );
}

rapidsmpf::streaming::Node read_customer(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    std::size_t num_producers,
    cudf::size_type num_rows_per_chunk,
    std::string const& input_directory
) {
    auto files = rapidsmpf::ndsh::detail::list_parquet_files(
        get_table_path(input_directory, "customer")
    );
    auto options = cudf::io::parquet_reader_options::builder(cudf::io::source_info(files))
                       .columns({"c_custkey", "c_name"})
                       .build();
    return rapidsmpf::streaming::node::read_parquet(
        ctx, ch_out, num_producers, options, num_rows_per_chunk
    );
}

// ============================================================================
// Phase 1: Compute Qualifying Orderkeys
// ============================================================================

/**
 * @brief Stage 1: Chunk-wise groupby (NO filter yet!)
 *
 * Computes partial aggregates: groupby(l_orderkey, sum(l_quantity))
 * The same orderkey may appear in multiple chunks, so we can't filter here.
 */
rapidsmpf::streaming::Node chunkwise_groupby_lineitem(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
    auto mr = ctx->br()->device_mr();
    std::uint64_t sequence = 0;
    std::size_t total_input_rows = 0;
    std::size_t total_output_rows = 0;
    std::size_t chunk_count = 0;

    while (true) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        co_await ctx->executor()->schedule();
        auto chunk = rapidsmpf::ndsh::to_device(
            ctx, msg.release<rapidsmpf::streaming::TableChunk>()
        );
        auto chunk_stream = chunk.stream();
        auto table = chunk.table_view();
        total_input_rows += static_cast<std::size_t>(table.num_rows());
        chunk_count++;

        // Groupby l_orderkey, sum(l_quantity) - NO FILTER
        auto grouper = cudf::groupby::groupby(
            table.select({0}), cudf::null_policy::EXCLUDE, cudf::sorted::NO
        );
        std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
        aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        std::vector<cudf::groupby::aggregation_request> requests;
        requests.push_back(
            cudf::groupby::aggregation_request(table.column(1), std::move(aggs))
        );
        auto [keys, results] = grouper.aggregate(requests, chunk_stream, mr);

        auto result_columns = keys->release();
        for (auto&& r : results) {
            std::ranges::move(r.results, std::back_inserter(result_columns));
        }
        auto grouped_table = std::make_unique<cudf::table>(std::move(result_columns));
        total_output_rows += static_cast<std::size_t>(grouped_table->num_rows());

        if (grouped_table->num_rows() > 0) {
            co_await ch_out->send(
                rapidsmpf::streaming::to_message(
                    sequence++,
                    std::make_unique<rapidsmpf::streaming::TableChunk>(
                        std::move(grouped_table), chunk_stream
                    )
                )
            );
        }
    }

    ctx->comm()->logger().print(
        "chunkwise_groupby: rank processed ",
        chunk_count,
        " chunks, ",
        total_input_rows,
        " -> ",
        total_output_rows,
        " rows"
    );
    co_await ch_out->drain(ctx->executor());
}

/**
 * @brief Stage 3: Final groupby + filter (after concatenate + all-gather)
 *
 * Merges partial aggregates and filters for sum > threshold.
 * Input: concatenated partial aggregates (l_orderkey, partial_sum)
 * Output: qualifying orderkeys (l_orderkey, total_sum where total_sum > threshold)
 */
rapidsmpf::streaming::Node final_groupby_filter_lineitem(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    double quantity_threshold
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
    auto mr = ctx->br()->device_mr();

    auto msg = co_await ch_in->receive();
    if (msg.empty()) {
        ctx->comm()->logger().print("final_groupby_filter: rank received EMPTY input!");
        co_await ch_out->drain(ctx->executor());
        co_return;
    } else {
        auto next = co_await ch_in->receive();
        RAPIDSMPF_EXPECTS(
            next.empty(), "final_groupby_filter: Unexpected second message."
        );
    }

    auto chunk =
        rapidsmpf::ndsh::to_device(ctx, msg.release<rapidsmpf::streaming::TableChunk>());
    auto chunk_stream = chunk.stream();
    auto table = chunk.table_view();

    ctx->comm()->logger().print(
        "final_groupby_filter: rank processing ", table.num_rows(), " partial aggregates"
    );

    // Final groupby to merge partial sums for same orderkey
    auto grouper = cudf::groupby::groupby(
        table.select({0}), cudf::null_policy::EXCLUDE, cudf::sorted::NO
    );
    std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
    aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
    std::vector<cudf::groupby::aggregation_request> requests;
    requests.push_back(
        cudf::groupby::aggregation_request(table.column(1), std::move(aggs))
    );
    auto [keys, results] = grouper.aggregate(requests, chunk_stream, mr);

    auto result_columns = keys->release();
    for (auto&& r : results) {
        std::ranges::move(r.results, std::back_inserter(result_columns));
    }
    auto merged_table = std::make_unique<cudf::table>(std::move(result_columns));

    ctx->comm()->logger().print(
        "final_groupby_filter: rank merged to ",
        merged_table->num_rows(),
        " unique orderkeys"
    );

    // NOW filter for sum > threshold
    auto sum_col = merged_table->view().column(1);
    auto threshold_scalar = cudf::make_numeric_scalar(
        cudf::data_type(cudf::type_id::FLOAT64), chunk_stream, mr
    );
    static_cast<cudf::numeric_scalar<double>*>(threshold_scalar.get())
        ->set_value(quantity_threshold, chunk_stream);

    auto mask = cudf::binary_operation(
        sum_col,
        *threshold_scalar,
        cudf::binary_operator::GREATER,
        cudf::data_type(cudf::type_id::BOOL8),
        chunk_stream,
        mr
    );

    auto filtered_table =
        cudf::apply_boolean_mask(merged_table->view(), mask->view(), chunk_stream, mr);

    ctx->comm()->logger().print(
        "final_groupby_filter: ",
        filtered_table->num_rows(),
        " qualifying orderkeys (sum > ",
        quantity_threshold,
        ")"
    );

    if (filtered_table->num_rows() > 0) {
        co_await ch_out->send(
            rapidsmpf::streaming::to_message(
                0,
                std::make_unique<rapidsmpf::streaming::TableChunk>(
                    std::move(filtered_table), chunk_stream
                )
            )
        );
    }
    co_await ch_out->drain(ctx->executor());
}

/**
 * @brief All-gather node: collect from ch_in, all-gather across ranks, send to ch_out.
 *
 * For single-rank, this is a simple pass-through.
 */
rapidsmpf::streaming::Node allgather_table(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    rapidsmpf::OpID tag
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};

    auto msg = co_await ch_in->receive();
    if (!msg.empty()) {
        auto next = co_await ch_in->receive();
        RAPIDSMPF_EXPECTS(next.empty(), "allgather_table: Unexpected second message.");
    }

    std::unique_ptr<cudf::table> result;
    rmm::cuda_stream_view stream;

    if (ctx->comm()->nranks() > 1) {
        rapidsmpf::streaming::AllGather gatherer{ctx, tag};

        if (!msg.empty()) {
            auto chunk = rapidsmpf::ndsh::to_device(
                ctx, msg.release<rapidsmpf::streaming::TableChunk>()
            );
            stream = chunk.stream();
            auto pack = cudf::pack(chunk.table_view(), stream, ctx->br()->device_mr());
            gatherer.insert(
                0,
                {rapidsmpf::PackedData(
                    std::move(pack.metadata),
                    ctx->br()->move(std::move(pack.gpu_data), stream)
                )}
            );
        }
        gatherer.insert_finished();

        auto packed_data =
            co_await gatherer.extract_all(rapidsmpf::streaming::AllGather::Ordered::NO);

        result = rapidsmpf::unpack_and_concat(
            rapidsmpf::unspill_partitions(
                std::move(packed_data), ctx->br(), true, ctx->statistics()
            ),
            stream,
            ctx->br(),
            ctx->statistics()
        );
    } else {
        // Single rank - just forward the message as-is
        if (!msg.empty()) {
            co_await ch_out->send(std::move(msg));
        }
        ctx->comm()->logger().print(
            "allgather_table: single rank finished forwarding message"
        );
        co_await ch_out->drain(ctx->executor());
        co_return;
    }

    ctx->comm()->logger().debug(
        "allgather_table: ", result ? result->num_rows() : 0, " rows gathered"
    );

    if (result && result->num_rows() > 0) {
        co_await ch_out->send(
            rapidsmpf::streaming::to_message(
                0,
                std::make_unique<rapidsmpf::streaming::TableChunk>(
                    std::move(result), stream
                )
            )
        );
    }
    co_await ch_out->drain(ctx->executor());
}

/**
 * @brief Phase 1: Compute qualifying orderkeys.
 *
 * Pipeline strategy depends on number of ranks:
 *
 * Single-rank (all-gather approach):
 * 1. Read lineitem → chunkwise_groupby (partial aggregates)
 * 2. Concatenate → final groupby + filter
 *
 * Multi-rank (shuffle approach - required at scale!):
 * 1. Read lineitem → chunkwise_groupby (partial aggregates)
 * 2. SHUFFLE by orderkey (distributes work across ranks)
 * 3. Per-partition: concatenate → groupby → filter
 * 4. All-gather ONLY the tiny filtered result (~57K orderkeys at SF1000)
 *
 * The shuffle approach is required for multi-rank because:
 * - Each rank produces partial sums for MOST orderkeys (not just 1/N)
 * - All-gathering these partials would be O(orderkeys) per rank = OOM at scale
 * - Shuffle ensures each rank only merges 1/N of the orderkeys
 *
 * @return Table with single column (l_orderkey) of qualifying orders, or nullptr if
 * empty.
 */
std::unique_ptr<cudf::table> compute_qualifying_orderkeys(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    cudf::size_type num_rows_per_chunk,
    std::string const& input_directory,
    double quantity_threshold,
    std::uint32_t num_partitions,
    rapidsmpf::OpID& base_tag
) {
    bool const single_rank = ctx->comm()->nranks() == 1;
    ctx->comm()->logger().print(
        "Phase 1: Computing qualifying orderkeys",
        single_rank ? " (single-rank: local groupby)" : " (multi-rank: shuffle-based)"
    );

    // Build Phase 1 pipeline
    std::vector<rapidsmpf::streaming::Node> nodes;

    // Stage 1: Read lineitem → chunk-wise groupby (partial aggregates)
    auto lineitem = ctx->create_channel();
    nodes.push_back(read_lineitem(ctx, lineitem, 4, num_rows_per_chunk, input_directory));

    auto partial_aggs = ctx->create_channel();
    nodes.push_back(chunkwise_groupby_lineitem(ctx, lineitem, partial_aggs));

    std::shared_ptr<rapidsmpf::streaming::Channel> to_collect;

    if (single_rank) {
        // Single rank: simple local pipeline (no shuffle needed)
        auto to_concat = partial_aggs;

        auto concatenated = ctx->create_channel();
        nodes.push_back(
            rapidsmpf::ndsh::concatenate(
                ctx, to_concat, concatenated, rapidsmpf::ndsh::ConcatOrder::DONT_CARE
            )
        );

        to_collect = ctx->create_channel();
        nodes.push_back(final_groupby_filter_lineitem(
            ctx, concatenated, to_collect, quantity_threshold
        ));
    } else {
        // Multi-rank: SHUFFLE partial aggregates by orderkey (column 0)
        // This distributes work across ranks - required at scale!
        auto partial_aggs_shuffled = ctx->create_channel();
        nodes.push_back(
            rapidsmpf::ndsh::shuffle(
                ctx,
                partial_aggs,
                partial_aggs_shuffled,
                {0},  // l_orderkey
                num_partitions,
                base_tag++
            )
        );
        auto to_concat = partial_aggs_shuffled;

        // Per-partition: concatenate → groupby → filter
        auto concatenated = ctx->create_channel();
        nodes.push_back(
            rapidsmpf::ndsh::concatenate(
                ctx, to_concat, concatenated, rapidsmpf::ndsh::ConcatOrder::DONT_CARE
            )
        );

        auto filtered_local = ctx->create_channel();
        nodes.push_back(final_groupby_filter_lineitem(
            ctx, concatenated, filtered_local, quantity_threshold
        ));

        // All-gather the TINY filtered result (~57K orderkeys at SF1000)
        auto gathered = ctx->create_channel();
        nodes.push_back(allgather_table(ctx, filtered_local, gathered, base_tag++));
        to_collect = gathered;
    }

    // Collect result using pull_from_channel (safe coroutine pattern)
    std::vector<rapidsmpf::streaming::Message> result_messages;
    nodes.push_back(
        rapidsmpf::streaming::node::pull_from_channel(ctx, to_collect, result_messages)
    );

    // Run pipeline
    rapidsmpf::streaming::run_streaming_pipeline(std::move(nodes));

    // Extract result from collected messages
    std::unique_ptr<cudf::table> result;
    if (!result_messages.empty()) {
        auto chunk = rapidsmpf::ndsh::to_device(
            ctx, result_messages[0].release<rapidsmpf::streaming::TableChunk>()
        );
        auto stream = chunk.stream();
        auto table_view = chunk.table_view();

        // Extract just the orderkey column (column 0)
        // The filtered result has (l_orderkey, sum_quantity), we only need l_orderkey
        std::vector<std::unique_ptr<cudf::column>> cols;
        cols.push_back(
            std::make_unique<cudf::column>(
                table_view.column(0), stream, ctx->br()->device_mr()
            )
        );
        stream.synchronize();
        result = std::make_unique<cudf::table>(std::move(cols));
    }

    ctx->comm()->logger().print(
        "Phase 1 complete: ", result ? result->num_rows() : 0, " qualifying orderkeys"
    );

    return result;
}

// ============================================================================
// Phase 2: Pre-filter Pipeline
// ============================================================================

/**
 * @brief Pre-filter table by qualifying orderkeys using semi-join.
 *
 * @param qualifying_orderkeys Table with single column of qualifying l_orderkey values.
 * @param key_column_idx Which column in input chunks to match against orderkeys.
 */
rapidsmpf::streaming::Node prefilter_by_orderkeys(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    std::shared_ptr<cudf::table> qualifying_orderkeys,
    cudf::size_type key_column_idx
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
    std::uint64_t sequence = 0;

    // Build filtered_join for semi-join (orderkeys is the "right"/build side)
    auto joiner = cudf::filtered_join(
        qualifying_orderkeys->view(),
        cudf::null_equality::UNEQUAL,
        cudf::set_as_build_table::RIGHT,
        ctx->br()->stream_pool().get_stream()
    );

    std::size_t total_input_rows = 0;
    std::size_t total_output_rows = 0;

    while (true) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        co_await ctx->executor()->schedule();
        auto chunk = rapidsmpf::ndsh::to_device(
            ctx, msg.release<rapidsmpf::streaming::TableChunk>()
        );
        auto chunk_stream = chunk.stream();
        auto table = chunk.table_view();
        total_input_rows += static_cast<std::size_t>(table.num_rows());

        // Semi-join: get indices of matching rows
        auto match = joiner.semi_join(
            table.select({key_column_idx}), chunk_stream, ctx->br()->device_mr()
        );

        // Gather matching rows - convert device_uvector to column_view
        auto indices = cudf::column_view(
            cudf::data_type{cudf::type_id::INT32},
            static_cast<cudf::size_type>(match->size()),
            match->data(),
            nullptr,  // null mask
            0  // null count
        );
        auto filtered = cudf::gather(
            table, indices, cudf::out_of_bounds_policy::DONT_CHECK, chunk_stream
        );

        total_output_rows += static_cast<std::size_t>(filtered->num_rows());

        if (filtered->num_rows() > 0) {
            co_await ch_out->send(
                rapidsmpf::streaming::to_message(
                    sequence++,
                    std::make_unique<rapidsmpf::streaming::TableChunk>(
                        std::move(filtered), chunk_stream
                    )
                )
            );
        }
    }

    ctx->comm()->logger().print(
        "prefilter: rank processed ",
        total_input_rows,
        " -> ",
        total_output_rows,
        " rows (",
        (total_output_rows * 100.0 / std::max(total_input_rows, std::size_t{1})),
        "%)"
    );

    co_await ch_out->drain(ctx->executor());
}

/**
 * @brief Local inner join (both inputs are small after pre-filtering).
 *
 * Receives one chunk from each input, joins them, outputs result.
 */
rapidsmpf::streaming::Node local_inner_join(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_left,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_right,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    std::vector<cudf::size_type> left_on,
    std::vector<cudf::size_type> right_on
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_left, ch_right, ch_out};

    auto left_msg = co_await ch_left->receive();
    auto right_msg = co_await ch_right->receive();

    if (left_msg.empty() || right_msg.empty()) {
        co_await ch_out->drain(ctx->executor());
        co_return;
    }

    auto left_chunk = rapidsmpf::ndsh::to_device(
        ctx, left_msg.release<rapidsmpf::streaming::TableChunk>()
    );
    auto next = co_await ch_left->receive();
    RAPIDSMPF_EXPECTS(next.empty(), "Unexpected second message from left channel.");
    auto right_chunk = rapidsmpf::ndsh::to_device(
        ctx, right_msg.release<rapidsmpf::streaming::TableChunk>()
    );
    next = co_await ch_right->receive();
    RAPIDSMPF_EXPECTS(next.empty(), "Unexpected second message from right channel.");

    auto stream = left_chunk.stream();
    rapidsmpf::cuda_stream_join(stream, right_chunk.stream());

    auto left_table = left_chunk.table_view();
    auto right_table = right_chunk.table_view();

    ctx->comm()->logger().print(
        "local_inner_join: ", left_table.num_rows(), " x ", right_table.num_rows()
    );

    // Build hash table on right (smaller side typically)
    auto hash_table =
        cudf::hash_join(right_table.select(right_on), cudf::null_equality::EQUAL, stream);

    auto [left_indices, right_indices] = hash_table.inner_join(
        left_table.select(left_on), {}, stream, ctx->br()->device_mr()
    );

    // Gather from both sides using device_span for column_view
    cudf::column_view left_col = cudf::device_span<cudf::size_type const>(*left_indices);
    cudf::column_view right_col =
        cudf::device_span<cudf::size_type const>(*right_indices);

    auto left_gathered = cudf::gather(
        left_table, left_col, cudf::out_of_bounds_policy::DONT_CHECK, stream
    );
    auto right_gathered = cudf::gather(
        right_table.select({1}),  // Only l_quantity from lineitem
        right_col,
        cudf::out_of_bounds_policy::DONT_CHECK,
        stream
    );

    // Concatenate columns: all from left + l_quantity from right
    auto result_cols = left_gathered->release();
    std::ranges::move(right_gathered->release(), std::back_inserter(result_cols));

    auto result = std::make_unique<cudf::table>(std::move(result_cols));
    ctx->comm()->logger().print(
        "local_inner_join: result has ", result->num_rows(), " rows"
    );

    co_await ch_out->send(
        rapidsmpf::streaming::to_message(
            0,
            std::make_unique<rapidsmpf::streaming::TableChunk>(std::move(result), stream)
        )
    );
    co_await ch_out->drain(ctx->executor());
}

// ============================================================================
// Final Processing (reused from q18.cpp)
// ============================================================================

rapidsmpf::streaming::Node chunkwise_groupby_agg(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
    std::uint64_t sequence = 0;
    while (true) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        co_await ctx->executor()->schedule();
        auto chunk = rapidsmpf::ndsh::to_device(
            ctx, msg.release<rapidsmpf::streaming::TableChunk>()
        );
        auto chunk_stream = chunk.stream();
        // Reorder columns: swap 0 and 1 to get (c_custkey, o_orderkey, ...)
        auto table = chunk.table_view().select({1, 0, 2, 3, 4, 5});

        auto grouper = cudf::groupby::groupby(
            table.select({0, 1, 2, 3, 4}), cudf::null_policy::EXCLUDE, cudf::sorted::NO
        );
        std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
        aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        std::vector<cudf::groupby::aggregation_request> requests;
        requests.push_back(
            cudf::groupby::aggregation_request(table.column(5), std::move(aggs))
        );
        auto [keys, results] =
            grouper.aggregate(requests, chunk_stream, ctx->br()->device_mr());

        auto result_columns = keys->release();
        for (auto&& r : results) {
            std::ranges::move(r.results, std::back_inserter(result_columns));
        }

        co_await ch_out->send(
            rapidsmpf::streaming::to_message(
                sequence++,
                std::make_unique<rapidsmpf::streaming::TableChunk>(
                    std::make_unique<cudf::table>(std::move(result_columns)), chunk_stream
                )
            )
        );
    }
    co_await ch_out->drain(ctx->executor());
}

rapidsmpf::streaming::Node final_groupby_and_sort(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    rapidsmpf::OpID allgather_tag
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};

    auto msg = co_await ch_in->receive();
    if (msg.empty()) {
        co_await ch_out->drain(ctx->executor());
        co_return;
    } else {
        auto next = co_await ch_in->receive();
        RAPIDSMPF_EXPECTS(
            next.empty(), "final_groupby_and_sort: Unexpected second message."
        );
    }

    auto chunk =
        rapidsmpf::ndsh::to_device(ctx, msg.release<rapidsmpf::streaming::TableChunk>());
    auto stream = chunk.stream();
    auto table = chunk.table_view();

    // Local groupby
    std::unique_ptr<cudf::table> local_result;
    if (table.num_rows() > 0) {
        auto grouper = cudf::groupby::groupby(
            table.select({0, 1, 2, 3, 4}), cudf::null_policy::EXCLUDE, cudf::sorted::NO
        );
        std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
        aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        std::vector<cudf::groupby::aggregation_request> requests;
        requests.push_back(
            cudf::groupby::aggregation_request(table.column(5), std::move(aggs))
        );
        auto [keys, results] =
            grouper.aggregate(requests, stream, ctx->br()->device_mr());
        auto cols = keys->release();
        for (auto&& r : results) {
            std::ranges::move(r.results, std::back_inserter(cols));
        }
        local_result = std::make_unique<cudf::table>(std::move(cols));
    }
    {
        auto _ = std::move(chunk);
    }  // Release original input to free GPU memory

    // All-gather if multi-rank
    std::unique_ptr<cudf::table> global_result;
    if (ctx->comm()->nranks() > 1 && local_result) {
        rapidsmpf::streaming::AllGather gatherer{ctx, allgather_tag};
        auto pack = cudf::pack(local_result->view(), stream, ctx->br()->device_mr());
        gatherer.insert(
            0,
            {rapidsmpf::PackedData(
                std::move(pack.metadata),
                ctx->br()->move(std::move(pack.gpu_data), stream)
            )}
        );
        gatherer.insert_finished();

        auto packed_data =
            co_await gatherer.extract_all(rapidsmpf::streaming::AllGather::Ordered::NO);

        if (ctx->comm()->rank() == 0) {
            auto gathered = rapidsmpf::unpack_and_concat(
                rapidsmpf::unspill_partitions(
                    std::move(packed_data), ctx->br(), true, ctx->statistics()
                ),
                stream,
                ctx->br(),
                ctx->statistics()
            );

            // Final groupby
            auto grouper = cudf::groupby::groupby(
                gathered->view().select({0, 1, 2, 3, 4}),
                cudf::null_policy::EXCLUDE,
                cudf::sorted::NO
            );
            std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
            aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
            std::vector<cudf::groupby::aggregation_request> requests;
            requests.push_back(
                cudf::groupby::aggregation_request(
                    gathered->view().column(5), std::move(aggs)
                )
            );
            auto [keys, results] =
                grouper.aggregate(requests, stream, ctx->br()->device_mr());
            auto cols = keys->release();
            for (auto&& r : results) {
                std::ranges::move(r.results, std::back_inserter(cols));
            }
            global_result = std::make_unique<cudf::table>(std::move(cols));
        }
    } else {
        global_result = std::move(local_result);
    }

    // Sort and limit (rank 0 only in multi-rank)
    if (global_result && (ctx->comm()->nranks() == 1 || ctx->comm()->rank() == 0)) {
        auto sorted = cudf::sort_by_key(
            global_result->view(),
            global_result->view().select({4, 3}),  // o_totalprice DESC, o_orderdate ASC
            {cudf::order::DESCENDING, cudf::order::ASCENDING},
            {cudf::null_order::AFTER, cudf::null_order::AFTER},
            stream,
            ctx->br()->device_mr()
        );
        cudf::size_type limit = std::min(100, sorted->num_rows());
        auto limited = cudf::slice(sorted->view(), {0, limit})[0];

        co_await ch_out->send(
            rapidsmpf::streaming::to_message(
                0,
                std::make_unique<rapidsmpf::streaming::TableChunk>(
                    std::make_unique<cudf::table>(
                        limited, stream, ctx->br()->device_mr()
                    ),
                    stream
                )
            )
        );
    }

    co_await ch_out->drain(ctx->executor());
}

rapidsmpf::streaming::Node write_parquet(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::string output_path
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in};
    auto msg = co_await ch_in->receive();
    if (msg.empty()) {
        co_return;
    } else {
        auto next = co_await ch_in->receive();
        RAPIDSMPF_EXPECTS(next.empty(), "write_parquet: Unexpected second message.");
    }
    auto chunk =
        rapidsmpf::ndsh::to_device(ctx, msg.release<rapidsmpf::streaming::TableChunk>());
    auto sink = cudf::io::sink_info(output_path);
    auto builder = cudf::io::parquet_writer_options::builder(sink, chunk.table_view());
    auto metadata = cudf::io::table_input_metadata(chunk.table_view());
    metadata.column_metadata[0].set_name("c_name");
    metadata.column_metadata[1].set_name("c_custkey");
    metadata.column_metadata[2].set_name("o_orderkey");
    metadata.column_metadata[3].set_name("o_orderdate");
    metadata.column_metadata[4].set_name("o_totalprice");
    metadata.column_metadata[5].set_name("sum_quantity");
    builder = builder.metadata(metadata);
    cudf::io::write_parquet(builder.build(), chunk.stream());
    ctx->comm()->logger().print(
        "Wrote ", chunk.table_view().num_rows(), " rows to ", output_path
    );
}

}  // namespace

// ============================================================================
// Command Line Parsing
// ============================================================================

struct Q18Options {
    std::uint32_t num_partitions{64};
};

Q18Options parse_options(int argc, char** argv) {
    Q18Options options;
    // NOLINTBEGIN(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,modernize-use-designated-initializers)
    opterr = 0;

    static struct option long_options[] = {
        {"num-partitions", required_argument, nullptr, 7}, {nullptr, 0, nullptr, 0}
    };
    // NOLINTEND(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,modernize-use-designated-initializers)

    int opt;
    int option_index = 0;

    while ((opt = getopt_long(argc, argv, "", long_options, &option_index)) != -1) {
        switch (opt) {
        case 1:
            options.num_partitions = static_cast<std::uint32_t>(std::atoi(optarg));
            break;
        }
    }

    return options;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    rapidsmpf::ndsh::FinalizeMPI finalize{};
    cudaFree(nullptr);
    // work around https://github.com/rapidsai/cudf/issues/20849
    cudf::initialize();
    auto mr = rmm::mr::cuda_async_memory_resource{};
    auto stats_wrapper = rapidsmpf::RmmResourceAdaptor(&mr);
    auto arguments = rapidsmpf::ndsh::parse_arguments(argc, argv);
    auto q18_arguments = parse_options(argc, argv);
    auto ctx = rapidsmpf::ndsh::create_context(arguments, &stats_wrapper);
    std::string output_path = arguments.output_file;

    ctx->comm()->logger().print("Q18 Pre-filter Benchmark");
    ctx->comm()->logger().print(
        "Executor has ", ctx->executor()->thread_count(), " threads"
    );
    ctx->comm()->logger().print("Executor has ", ctx->comm()->nranks(), " ranks");
    ctx->comm()->logger().print(
        "Phase 1 mode: ",
        ctx->comm()->nranks() > 1 ? "shuffle (multi-rank)" : "local (single-rank)",
        ", Phase 2 mode: ",
        arguments.use_shuffle_join ? "shuffle joins" : "local joins",
        ", partitions: ",
        // arguments.num_partitions
        q18_arguments.num_partitions
    );

    for (int i = 0; i < arguments.num_iterations; i++) {
        auto start = std::chrono::steady_clock::now();

        // ================================================================
        // Phase 1: Compute qualifying orderkeys (blocking)
        // Uses shuffle for multi-rank (required at scale to avoid OOM)
        // Uses simple local groupby for single-rank
        // ================================================================
        rapidsmpf::OpID phase1_op_id{static_cast<rapidsmpf::OpID>(100 + i * 10)};
        std::unique_ptr<cudf::table> qualifying_orderkeys = compute_qualifying_orderkeys(
            ctx,
            arguments.num_rows_per_chunk,
            arguments.input_directory,
            300.0,  // quantity_threshold
            q18_arguments.num_partitions,
            phase1_op_id
        );
        auto phase1_end = std::chrono::steady_clock::now();

        if (!qualifying_orderkeys || qualifying_orderkeys->num_rows() == 0) {
            ctx->comm()->logger().print("No qualifying orderkeys found - empty result");
            continue;
        }

        // Share orderkeys across nodes (they're small and identical on all ranks)
        auto shared_orderkeys =
            std::make_shared<cudf::table>(std::move(*qualifying_orderkeys));

        // ================================================================
        // Phase 2: Build pre-filtered pipeline
        // ================================================================
        std::vector<rapidsmpf::streaming::Node> nodes;
        std::uint32_t num_partitions = q18_arguments.num_partitions;
        int phase2_op_id = 0;

        // Read and pre-filter lineitem
        auto lineitem_raw = ctx->create_channel();
        nodes.push_back(read_lineitem(
            ctx, lineitem_raw, 4, arguments.num_rows_per_chunk, arguments.input_directory
        ));

        auto lineitem_filtered = ctx->create_channel();
        nodes.push_back(prefilter_by_orderkeys(
            ctx, lineitem_raw, lineitem_filtered, shared_orderkeys, 0
        ));

        // Read and pre-filter orders
        auto orders_raw = ctx->create_channel();
        nodes.push_back(read_orders(
            ctx, orders_raw, 4, arguments.num_rows_per_chunk, arguments.input_directory
        ));

        auto orders_filtered = ctx->create_channel();
        nodes.push_back(
            prefilter_by_orderkeys(ctx, orders_raw, orders_filtered, shared_orderkeys, 0)
        );

        // Read customer
        auto customer = ctx->create_channel();
        nodes.push_back(read_customer(
            ctx, customer, 4, arguments.num_rows_per_chunk, arguments.input_directory
        ));

        auto all_joined = ctx->create_channel();

        bool const single_rank = ctx->comm()->nranks() == 1;

        if (arguments.use_shuffle_join && !single_rank) {
            // ============================================================
            // SHUFFLE MODE: Proper parallel scaling (multi-rank only)
            // ============================================================

            // Shuffle filtered lineitem by orderkey
            auto lineitem_shuffled = ctx->create_channel();
            nodes.push_back(
                rapidsmpf::ndsh::shuffle(
                    ctx,
                    lineitem_filtered,
                    lineitem_shuffled,
                    {0},  // l_orderkey
                    num_partitions,
                    rapidsmpf::OpID{
                        static_cast<rapidsmpf::OpID>(200 + i * 10 + phase2_op_id++)
                    }
                )
            );

            // Shuffle filtered orders by orderkey
            auto orders_shuffled = ctx->create_channel();
            nodes.push_back(
                rapidsmpf::ndsh::shuffle(
                    ctx,
                    orders_filtered,
                    orders_shuffled,
                    {0},  // o_orderkey
                    num_partitions,
                    rapidsmpf::OpID{
                        static_cast<rapidsmpf::OpID>(200 + i * 10 + phase2_op_id++)
                    }
                )
            );

            // Shuffle-based join: orders x lineitem
            // Output: o_orderkey, o_custkey, o_orderdate, o_totalprice, l_quantity
            auto orders_x_lineitem = ctx->create_channel();
            nodes.push_back(
                rapidsmpf::ndsh::inner_join_shuffle(
                    ctx,
                    orders_shuffled,
                    lineitem_shuffled,
                    orders_x_lineitem,
                    {0},  // o_orderkey
                    {0},  // l_orderkey
                    rapidsmpf::ndsh::KeepKeys::YES
                )
            );

            // Shuffle orders_x_lineitem by custkey for customer join
            auto orders_x_lineitem_shuffled = ctx->create_channel();
            nodes.push_back(
                rapidsmpf::ndsh::shuffle(
                    ctx,
                    orders_x_lineitem,
                    orders_x_lineitem_shuffled,
                    {1},  // o_custkey
                    num_partitions,
                    rapidsmpf::OpID{
                        static_cast<rapidsmpf::OpID>(200 + i * 10 + phase2_op_id++)
                    }
                )
            );

            // Shuffle customer by custkey
            auto customer_shuffled = ctx->create_channel();
            nodes.push_back(
                rapidsmpf::ndsh::shuffle(
                    ctx,
                    customer,
                    customer_shuffled,
                    {0},  // c_custkey
                    num_partitions,
                    rapidsmpf::OpID{
                        static_cast<rapidsmpf::OpID>(200 + i * 10 + phase2_op_id++)
                    }
                )
            );

            // Shuffle-based join: customer x orders_x_lineitem
            nodes.push_back(
                rapidsmpf::ndsh::inner_join_shuffle(
                    ctx,
                    customer_shuffled,
                    orders_x_lineitem_shuffled,
                    all_joined,
                    {0},  // c_custkey
                    {1},  // o_custkey
                    rapidsmpf::ndsh::KeepKeys::YES
                )
            );

        } else {
            // ============================================================
            // ALL-GATHER MODE: Simple but doesn't scale
            // ============================================================

            auto lineitem_concat = ctx->create_channel();
            nodes.push_back(
                rapidsmpf::ndsh::concatenate(
                    ctx,
                    lineitem_filtered,
                    lineitem_concat,
                    rapidsmpf::ndsh::ConcatOrder::DONT_CARE
                )
            );

            auto lineitem_gathered = ctx->create_channel();
            nodes.push_back(allgather_table(
                ctx,
                lineitem_concat,
                lineitem_gathered,
                rapidsmpf::OpID{static_cast<rapidsmpf::OpID>(200 + i)}
            ));

            auto orders_concat = ctx->create_channel();
            nodes.push_back(
                rapidsmpf::ndsh::concatenate(
                    ctx,
                    orders_filtered,
                    orders_concat,
                    rapidsmpf::ndsh::ConcatOrder::DONT_CARE
                )
            );

            auto orders_gathered = ctx->create_channel();
            nodes.push_back(allgather_table(
                ctx,
                orders_concat,
                orders_gathered,
                rapidsmpf::OpID{static_cast<rapidsmpf::OpID>(201 + i)}
            ));

            // Local join: orders x lineitem (both small after pre-filtering)
            auto orders_x_lineitem = ctx->create_channel();
            nodes.push_back(local_inner_join(
                ctx,
                orders_gathered,
                lineitem_gathered,
                orders_x_lineitem,
                {0},  // o_orderkey
                {0}  // l_orderkey
            ));

            // Join with customer (broadcast - orders_x_lineitem is small)
            nodes.push_back(
                rapidsmpf::ndsh::inner_join_broadcast(
                    ctx,
                    customer,
                    orders_x_lineitem,
                    all_joined,
                    {0},  // c_custkey
                    {1},  // o_custkey
                    rapidsmpf::OpID{static_cast<rapidsmpf::OpID>(202 + i)},
                    rapidsmpf::ndsh::KeepKeys::YES
                )
            );
        }

        // Groupby aggregation (includes column reordering)
        auto groupby_output = ctx->create_channel();
        nodes.push_back(chunkwise_groupby_agg(ctx, all_joined, groupby_output));

        auto concat_groupby = ctx->create_channel();
        nodes.push_back(
            rapidsmpf::ndsh::concatenate(
                ctx,
                groupby_output,
                concat_groupby,
                rapidsmpf::ndsh::ConcatOrder::DONT_CARE
            )
        );

        // Final groupby, all-gather, sort, limit
        auto final_output = ctx->create_channel();
        nodes.push_back(final_groupby_and_sort(
            ctx,
            concat_groupby,
            final_output,
            rapidsmpf::OpID{static_cast<rapidsmpf::OpID>(200 + i * 10 + phase2_op_id++)}
        ));

        // Write output
        nodes.push_back(write_parquet(ctx, final_output, output_path));

        // Run pipeline
        auto phase2_build_end = std::chrono::steady_clock::now();
        rapidsmpf::streaming::run_streaming_pipeline(std::move(nodes));
        auto phase2_run_end = std::chrono::steady_clock::now();

        std::chrono::duration<double> phase1_time = phase1_end - start;
        std::chrono::duration<double> phase2_build_time = phase2_build_end - phase1_end;
        std::chrono::duration<double> phase2_run_time = phase2_run_end - phase2_build_end;
        std::chrono::duration<double> total_time = phase2_run_end - start;

        ctx->comm()->logger().print(
            "Iteration ", i, " Phase 1 (groupby+filter) [s]: ", phase1_time.count()
        );
        ctx->comm()->logger().print(
            "Iteration ", i, " Phase 2 build [s]: ", phase2_build_time.count()
        );
        ctx->comm()->logger().print(
            "Iteration ", i, " Phase 2 run [s]: ", phase2_run_time.count()
        );
        ctx->comm()->logger().print("Iteration ", i, " TOTAL [s]: ", total_time.count());
        ctx->comm()->logger().print(ctx->statistics()->report());
    }

    return 0;
}
