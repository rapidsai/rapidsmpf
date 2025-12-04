/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>

#include <cuda_runtime_api.h>
#include <getopt.h>
#include <mpi.h>

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
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
#include <rapidsmpf/communicator/single.hpp>
#include <rapidsmpf/communicator/ucxx.hpp>
#include <rapidsmpf/communicator/ucxx_utils.hpp>
#include <rapidsmpf/config.hpp>
#include <rapidsmpf/cuda_event.hpp>
#include <rapidsmpf/integrations/cudf/partition.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/nvtx.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/streaming/coll/allgather.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/fanout.hpp>
#include <rapidsmpf/streaming/core/node.hpp>
#include <rapidsmpf/streaming/cudf/parquet.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

#include "concatenate.hpp"
#include "join.hpp"
#include "utils.hpp"

namespace {

rapidsmpf::streaming::Node read_lineitem(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    std::size_t num_producers,
    cudf::size_type num_rows_per_chunk,
    std::string const& input_directory
) {
    auto files = rapidsmpf::ndsh::detail::list_parquet_files(
        rapidsmpf::ndsh::detail::get_table_path(input_directory, "lineitem")
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
        rapidsmpf::ndsh::detail::get_table_path(input_directory, "orders")
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
        rapidsmpf::ndsh::detail::get_table_path(input_directory, "customer")
    );
    auto options = cudf::io::parquet_reader_options::builder(cudf::io::source_info(files))
                       .columns({"c_custkey", "c_name"})
                       .build();
    return rapidsmpf::streaming::node::read_parquet(
        ctx, ch_out, num_producers, options, num_rows_per_chunk
    );
}

/**
 * @brief Perform a chunk-wise groupby aggregation to compute sum(l_quantity) per
 * l_orderkey, then filter for sum > 300.
 *
 * Input columns: l_orderkey (0), l_quantity (1)
 * Output columns: l_orderkey (0), sum_quantity (1)
 */
rapidsmpf::streaming::Node groupby_filter_lineitem(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    double quantity_threshold
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
    auto mr = ctx->br()->device_mr();
    while (true) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        co_await ctx->executor()->schedule();
        // Preserve the input sequence number (partition ID for shuffle joins)
        auto sequence = msg.sequence_number();
        auto chunk = rapidsmpf::ndsh::to_device(
            ctx, msg.release<rapidsmpf::streaming::TableChunk>()
        );
        auto chunk_stream = chunk.stream();
        auto table = chunk.table_view();

        // Groupby l_orderkey, sum(l_quantity)
        auto grouper = cudf::groupby::groupby(
            table.select({0}), cudf::null_policy::EXCLUDE, cudf::sorted::NO
        );
        auto requests = std::vector<cudf::groupby::aggregation_request>();
        std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
        aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        requests.push_back(
            cudf::groupby::aggregation_request(table.column(1), std::move(aggs))
        );
        auto [keys, results] = grouper.aggregate(requests, chunk_stream, mr);

        auto result_columns = keys->release();
        for (auto&& r : results) {
            std::ranges::move(r.results, std::back_inserter(result_columns));
        }
        auto grouped_table = std::make_unique<cudf::table>(std::move(result_columns));

        // Filter for sum_quantity > quantity_threshold
        auto sum_quantity_col = grouped_table->view().column(1);
        auto threshold_scalar = cudf::make_numeric_scalar(
            cudf::data_type(cudf::type_id::FLOAT64), chunk_stream, mr
        );
        static_cast<cudf::numeric_scalar<double>*>(threshold_scalar.get())
            ->set_value(quantity_threshold, chunk_stream);

        // Create mask: sum_quantity > threshold using binary_operation
        auto mask = cudf::binary_operation(
            sum_quantity_col,
            *threshold_scalar,
            cudf::binary_operator::GREATER,
            cudf::data_type(cudf::type_id::BOOL8),
            chunk_stream,
            mr
        );

        auto filtered_table = cudf::apply_boolean_mask(
            grouped_table->view(), mask->view(), chunk_stream, mr
        );

        co_await ch_out->send(
            rapidsmpf::streaming::to_message(
                sequence,
                std::make_unique<rapidsmpf::streaming::TableChunk>(
                    std::move(filtered_table), chunk_stream
                )
            )
        );
    }
    co_await ch_out->drain(ctx->executor());
}

/**
 * @brief Perform the final groupby aggregation.
 *
 * Input columns: c_name (0), c_custkey (1), o_orderkey (2), o_orderdate (3),
 *                o_totalprice (4), l_quantity (5)
 * Output columns: c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, col6 (sum of
 * l_quantity)
 */
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
        auto table = chunk.table_view();

        // Groupby (c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice)
        auto grouper = cudf::groupby::groupby(
            table.select({0, 1, 2, 3, 4}), cudf::null_policy::EXCLUDE, cudf::sorted::NO
        );
        auto requests = std::vector<cudf::groupby::aggregation_request>();
        std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
        aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
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

/**
 * @brief Final groupby aggregation across all chunks and ranks.
 */
rapidsmpf::streaming::Node final_groupby_agg(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    rapidsmpf::OpID tag
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
    co_await ctx->executor()->schedule();
    auto msg = co_await ch_in->receive();
    auto next = co_await ch_in->receive();
    ctx->comm()->logger().debug("Final groupby");
    RAPIDSMPF_EXPECTS(next.empty(), "Expecting concatenated input at this point");
    auto chunk =
        rapidsmpf::ndsh::to_device(ctx, msg.release<rapidsmpf::streaming::TableChunk>());
    auto chunk_stream = chunk.stream();
    auto table = chunk.table_view();
    ctx->comm()->logger().debug(
        "Final groupby input: ", table.num_rows(), " rows, ", table.num_columns(), " cols"
    );
    std::unique_ptr<cudf::table> local_result{nullptr};
    if (!table.is_empty()) {
        auto grouper = cudf::groupby::groupby(
            table.select({0, 1, 2, 3, 4}), cudf::null_policy::EXCLUDE, cudf::sorted::NO
        );
        auto requests = std::vector<cudf::groupby::aggregation_request>();
        std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
        aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        requests.push_back(
            cudf::groupby::aggregation_request(table.column(5), std::move(aggs))
        );
        auto [keys, results] =
            grouper.aggregate(requests, chunk_stream, ctx->br()->device_mr());
        std::ignore = std::move(chunk);
        auto result = keys->release();
        for (auto&& r : results) {
            std::ranges::move(r.results, std::back_inserter(result));
        }
        local_result = std::make_unique<cudf::table>(std::move(result));
        ctx->comm()->logger().debug(
            "Final groupby output: ", local_result->num_rows(), " rows"
        );
    }
    if (ctx->comm()->nranks() > 1) {
        rapidsmpf::streaming::AllGather gatherer{ctx, tag};
        if (local_result) {
            auto pack =
                cudf::pack(local_result->view(), chunk_stream, ctx->br()->device_mr());
            gatherer.insert(
                0,
                {rapidsmpf::PackedData(
                    std::move(pack.metadata),
                    ctx->br()->move(std::move(pack.gpu_data), chunk_stream)
                )}
            );
        }
        gatherer.insert_finished();
        auto packed_data =
            co_await gatherer.extract_all(rapidsmpf::streaming::AllGather::Ordered::NO);
        if (ctx->comm()->rank() == 0) {
            auto global_result = rapidsmpf::unpack_and_concat(
                rapidsmpf::unspill_partitions(
                    std::move(packed_data), ctx->br(), true, ctx->statistics()
                ),
                chunk_stream,
                ctx->br(),
                ctx->statistics()
            );
            if (ctx->comm()->rank() == 0) {
                auto result_view = global_result->view();
                auto grouper = cudf::groupby::groupby(
                    result_view.select({0, 1, 2, 3, 4}),
                    cudf::null_policy::EXCLUDE,
                    cudf::sorted::NO
                );
                auto requests = std::vector<cudf::groupby::aggregation_request>();
                std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
                aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
                requests.push_back(
                    cudf::groupby::aggregation_request(
                        result_view.column(5), std::move(aggs)
                    )
                );
                auto [keys, results] =
                    grouper.aggregate(requests, chunk_stream, ctx->br()->device_mr());
                global_result.reset();
                auto result = keys->release();
                for (auto&& r : results) {
                    std::ranges::move(r.results, std::back_inserter(result));
                }
                co_await ch_out->send(
                    rapidsmpf::streaming::to_message(
                        0,
                        std::make_unique<rapidsmpf::streaming::TableChunk>(
                            std::make_unique<cudf::table>(std::move(result)), chunk_stream
                        )
                    )
                );
            }
        } else {
            std::ignore = std::move(packed_data);
        }
    } else {
        co_await ch_out->send(
            rapidsmpf::streaming::to_message(
                0,
                std::make_unique<rapidsmpf::streaming::TableChunk>(
                    std::move(local_result), chunk_stream
                )
            )
        );
    }
    co_await ch_out->drain(ctx->executor());
}

/**
 * @brief Sort by o_totalprice DESC, o_orderdate ASC and take the top 100 rows.
 *
 * Input columns: c_name (0), c_custkey (1), o_orderkey (2), o_orderdate (3),
 *                o_totalprice (4), col6 (5)
 */
rapidsmpf::streaming::Node sort_and_limit(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
    co_await ctx->executor()->schedule();
    auto msg = co_await ch_in->receive();
    if (msg.empty()) {
        co_return;
    }
    ctx->comm()->logger().debug("Sort and limit");
    auto chunk =
        rapidsmpf::ndsh::to_device(ctx, msg.release<rapidsmpf::streaming::TableChunk>());
    auto table = chunk.table_view();

    // Sort by o_totalprice (4) DESC, o_orderdate (3) ASC
    auto sorted_table = cudf::sort_by_key(
        table,
        table.select({4, 3}),
        {cudf::order::DESCENDING, cudf::order::ASCENDING},
        {cudf::null_order::AFTER, cudf::null_order::AFTER},
        chunk.stream(),
        ctx->br()->device_mr()
    );

    // Take top 100 rows
    cudf::size_type limit = std::min(100, sorted_table->num_rows());
    auto result = cudf::slice(sorted_table->view(), {0, limit})[0];
    auto result_table =
        std::make_unique<cudf::table>(result, chunk.stream(), ctx->br()->device_mr());

    co_await ch_out->send(
        rapidsmpf::streaming::to_message(
            0,
            std::make_unique<rapidsmpf::streaming::TableChunk>(
                std::move(result_table), chunk.stream()
            )
        )
    );
    co_await ch_out->drain(ctx->executor());
}

/**
 * @brief Buffer that holds messages in a spillable container.
 *
 * This node collects all incoming messages and stores them in the context's
 * SpillableMessages container. The spill manager can then spill these messages
 * to host memory if device memory pressure occurs. Messages are forwarded
 * to the output channel only after all input is received.
 *
 * This is a workaround for rapidsai/rapidsmpf#675: the unbounded fanout's
 * internal cache is not yet spillable. Once that issue is resolved, this
 * node can be removed.
 *
 * @param ctx Streaming context.
 * @param ch_in Input channel.
 * @param ch_out Output channel.
 * @return Coroutine node.
 */
rapidsmpf::streaming::Node spillable_buffer(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
    ctx->comm()->logger().debug("spillable_buffer: starting");

    // Store message IDs - messages are held in the context's SpillableMessages
    // container where the spill manager can spill them if needed
    auto spillable = ctx->spillable_messages();
    std::vector<rapidsmpf::streaming::SpillableMessages::MessageId> message_ids;

    while (true) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        co_await ctx->executor()->schedule();

        // Insert into spillable container - spill manager can now spill if needed
        auto id = spillable->insert(std::move(msg));
        message_ids.push_back(id);
        ctx->comm()->logger().debug("spillable_buffer: stored message id ", id);
    }

    ctx->comm()->logger().debug(
        "spillable_buffer: forwarding ", message_ids.size(), " messages"
    );

    // Extract and forward all messages
    for (auto id : message_ids) {
        auto msg = spillable->extract(id);
        co_await ch_out->send(std::move(msg));
    }

    co_await ch_out->drain(ctx->executor());
}

/**
 * @brief Reorder columns from join output to match expected query output.
 *
 * Input:  c_custkey(0), c_name(1), o_orderkey(2), o_orderdate(3), o_totalprice(4),
 * l_quantity(5) Output: c_name(0), c_custkey(1), o_orderkey(2), o_orderdate(3),
 * o_totalprice(4), l_quantity(5)
 */
rapidsmpf::streaming::Node reorder_columns(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
    std::uint64_t seq = 0;
    while (true) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        co_await ctx->executor()->schedule();
        auto chunk = rapidsmpf::ndsh::to_device(
            ctx, msg.release<rapidsmpf::streaming::TableChunk>()
        );
        auto table = chunk.table_view();
        auto reordered_table = std::make_unique<cudf::table>(
            table.select({1, 0, 2, 3, 4, 5}), chunk.stream(), ctx->br()->device_mr()
        );
        co_await ch_out->send(
            rapidsmpf::streaming::to_message(
                seq++,
                std::make_unique<rapidsmpf::streaming::TableChunk>(
                    std::move(reordered_table), chunk.stream()
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
    co_await ctx->executor()->schedule();
    auto msg = co_await ch_in->receive();
    if (msg.empty()) {
        co_return;
    }
    ctx->comm()->logger().debug("write parquet");
    auto chunk =
        rapidsmpf::ndsh::to_device(ctx, msg.release<rapidsmpf::streaming::TableChunk>());
    auto sink = cudf::io::sink_info(output_path);
    auto builder = cudf::io::parquet_writer_options::builder(sink, chunk.table_view());
    auto metadata = cudf::io::table_input_metadata(chunk.table_view());
    metadata.column_metadata[0].set_name("c_name");
    metadata.column_metadata[1].set_name("c_custkey");
    metadata.column_metadata[2].set_name("o_orderkey");
    metadata.column_metadata[3].set_name("o_orderdat");
    metadata.column_metadata[4].set_name("o_totalprice");
    metadata.column_metadata[5].set_name("col6");
    builder = builder.metadata(metadata);
    auto options = builder.build();
    cudf::io::write_parquet(options, chunk.stream());
    ctx->comm()->logger().debug(
        "Wrote ", chunk.table_view().num_rows(), " rows to ", output_path
    );
}

}  // namespace

struct ProgramOptions {
    int num_streaming_threads{1};
    cudf::size_type num_rows_per_chunk{100'000'000};
    std::uint32_t num_partitions{16};
    std::optional<double> spill_device_limit{std::nullopt};
    bool use_shuffle_join = false;
    bool shuffle_customer = false;
    bool spillable_fanout = false;
    std::string output_file;
    std::string input_directory;
};

ProgramOptions parse_options(int argc, char** argv) {
    ProgramOptions options;

    auto print_usage = [&argv]() {
        std::cerr
            << "Usage: " << argv[0] << " [options]\n"
            << "Options:\n"
            << "  --num-streaming-threads <n>  Number of streaming threads (default: 1)\n"
            << "  --num-rows-per-chunk <n>     Number of rows per chunk (default: "
               "100000000)\n"
            << "  --num-partitions <n>         Number of shuffle partitions (default: "
               "16, "
               "use 64+ for SF1000+)\n"
            << "  --spill-device-limit <n>     Fractional spill device limit (default: "
               "None)\n"
            << "  --use-shuffle-join           Use shuffle join for lineitem/orders "
               "(default: false)\n"
            << "  --shuffle-customer           Also shuffle customer join (for SF3000+, "
               "default: false)\n"
            << "  --spillable-fanout           Buffer fanout data in spillable "
               "container. "
               "Workaround for rapidsai/rapidsmpf#675 (default: false)\n"
            << "  --output-file <path>         Output file path (required)\n"
            << "  --input-directory <path>     Input directory path (required)\n"
            << "  --help                       Show this help message\n";
    };

    static struct option long_options[] = {
        {"num-streaming-threads", required_argument, nullptr, 1},
        {"num-rows-per-chunk", required_argument, nullptr, 2},
        {"use-shuffle-join", no_argument, nullptr, 3},
        {"output-file", required_argument, nullptr, 4},
        {"input-directory", required_argument, nullptr, 5},
        {"help", no_argument, nullptr, 6},
        {"spill-device-limit", required_argument, nullptr, 7},
        {"spillable-fanout", no_argument, nullptr, 8},
        {"shuffle-customer", no_argument, nullptr, 9},
        {"num-partitions", required_argument, nullptr, 10},
        {nullptr, 0, nullptr, 0}
    };

    int opt;
    int option_index = 0;

    bool saw_output_file = false;
    bool saw_input_directory = false;

    while ((opt = getopt_long(argc, argv, "", long_options, &option_index)) != -1) {
        switch (opt) {
        case 1:
            options.num_streaming_threads = std::atoi(optarg);
            break;
        case 2:
            options.num_rows_per_chunk = std::atoi(optarg);
            break;
        case 3:
            options.use_shuffle_join = true;
            break;
        case 4:
            options.output_file = optarg;
            saw_output_file = true;
            break;
        case 5:
            options.input_directory = optarg;
            saw_input_directory = true;
            break;
        case 6:
            print_usage();
            std::exit(0);
        case 7:
            options.spill_device_limit = std::stod(optarg);
            break;
        case 8:
            options.spillable_fanout = true;
            break;
        case 9:
            options.shuffle_customer = true;
            break;
        case 10:
            options.num_partitions = static_cast<std::uint32_t>(std::atoi(optarg));
            break;
        case '?':
            if (optopt == 0 && optind > 1) {
                std::cerr << "Error: Unknown option '" << argv[optind - 1] << "'\n\n";
            }
            print_usage();
            std::exit(1);
        default:
            print_usage();
            std::exit(1);
        }
    }

    // Check if required options were provided
    if (!saw_output_file || !saw_input_directory) {
        if (!saw_output_file) {
            std::cerr << "Error: --output-file is required\n";
        }
        if (!saw_input_directory) {
            std::cerr << "Error: --input-directory is required\n";
        }
        std::cerr << std::endl;
        print_usage();
        std::exit(1);
    }

    return options;
}

/**
 * @brief Run a derived version of TPC-H query 18.
 *
 * The SQL form of the query is:
 * @code{.sql}
 * select
 *     c_name,
 *     c_custkey,
 *     o_orderkey,
 *     o_orderdate as o_orderdat,
 *     o_totalprice,
 *     sum(l_quantity) as col6
 * from
 *     customer,
 *     orders,
 *     lineitem
 * where
 *     o_orderkey in (
 *         select
 *             l_orderkey
 *         from
 *             lineitem
 *         group by
 *             l_orderkey having
 *                 sum(l_quantity) > 300
 *     )
 *     and c_custkey = o_custkey
 *     and o_orderkey = l_orderkey
 * group by
 *     c_name,
 *     c_custkey,
 *     o_orderkey,
 *     o_orderdate,
 *     o_totalprice
 * order by
 *     o_totalprice desc,
 *     o_orderdate
 * limit 100
 * @endcode{}
 */
int main(int argc, char** argv) {
    cudaFree(nullptr);
    rapidsmpf::mpi::init(&argc, &argv);
    MPI_Comm mpi_comm;
    RAPIDSMPF_MPI(MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm));
    auto cmd_options = parse_options(argc, argv);
    auto limit_size = rmm::percent_of_free_device_memory(
        static_cast<std::size_t>(cmd_options.spill_device_limit.value_or(1) * 100)
    );
    rmm::mr::cuda_async_memory_resource mr{};
    auto stats_mr = rapidsmpf::RmmResourceAdaptor(&mr);
    rmm::device_async_resource_ref mr_ref(stats_mr);
    rmm::mr::set_current_device_resource(&stats_mr);
    rmm::mr::set_current_device_resource_ref(mr_ref);
    std::unordered_map<rapidsmpf::MemoryType, rapidsmpf::BufferResource::MemoryAvailable>
        memory_available{};
    if (cmd_options.spill_device_limit.has_value()) {
        memory_available[rapidsmpf::MemoryType::DEVICE] = rapidsmpf::LimitAvailableMemory{
            &stats_mr, static_cast<std::int64_t>(limit_size)
        };
    }
    auto br = std::make_shared<rapidsmpf::BufferResource>(
        stats_mr, std::move(memory_available)
    );
    auto envvars = rapidsmpf::config::get_environment_variables();
    envvars["num_streaming_threads"] = std::to_string(cmd_options.num_streaming_threads);
    auto options = rapidsmpf::config::Options(envvars);
    auto stats = std::make_shared<rapidsmpf::Statistics>(&stats_mr);
    {
        auto comm = rapidsmpf::ucxx::init_using_mpi(mpi_comm, options);
        auto progress =
            std::make_shared<rapidsmpf::ProgressThread>(comm->logger(), stats);
        auto ctx =
            std::make_shared<rapidsmpf::streaming::Context>(options, comm, br, stats);
        comm->logger().print(
            "Executor has ", ctx->executor()->thread_count(), " threads"
        );
        comm->logger().print("Executor has ", ctx->comm()->nranks(), " ranks");

        std::string output_path = cmd_options.output_file;
        std::vector<double> timings;
        for (int i = 0; i < 2; i++) {
            rapidsmpf::OpID op_id{0};
            std::vector<rapidsmpf::streaming::Node> nodes;
            auto start = std::chrono::steady_clock::now();
            {
                RAPIDSMPF_NVTX_SCOPED_RANGE("Constructing Q18 pipeline");

                // Number of partitions for shuffle operations
                std::uint32_t num_partitions = cmd_options.num_partitions;

                // Read orders
                auto orders = ctx->create_channel();
                nodes.push_back(read_orders(
                    ctx,
                    orders,
                    /* num_tickets */ 4,
                    cmd_options.num_rows_per_chunk,
                    cmd_options.input_directory
                ));  // o_orderkey, o_custkey, o_orderdate, o_totalprice

                // Read customer
                auto customer = ctx->create_channel();
                nodes.push_back(read_customer(
                    ctx,
                    customer,
                    /* num_tickets */ 4,
                    cmd_options.num_rows_per_chunk,
                    cmd_options.input_directory
                ));  // c_custkey, c_name

                auto orders_x_lineitem = ctx->create_channel();

                if (cmd_options.use_shuffle_join) {
                    // ========== SHUFFLE-BASED PIPELINE ==========
                    // For large scale factors (SF300+), use shuffle-based operations
                    // Shuffle lineitem once, then fanout to both consumers.

                    // 1. Read lineitem once
                    auto lineitem = ctx->create_channel();
                    nodes.push_back(read_lineitem(
                        ctx,
                        lineitem,
                        /* num_tickets */ 4,
                        cmd_options.num_rows_per_chunk,
                        cmd_options.input_directory
                    ));

                    // 2. Shuffle lineitem by l_orderkey (single shuffle for both uses)
                    auto lineitem_shuffled = ctx->create_channel();
                    nodes.push_back(
                        rapidsmpf::ndsh::shuffle(
                            ctx,
                            lineitem,
                            lineitem_shuffled,
                            {0},  // l_orderkey
                            num_partitions,
                            rapidsmpf::OpID{
                                static_cast<rapidsmpf::OpID>(10 * i + op_id++)
                            }
                        )
                    );

                    // 3. Fanout shuffled lineitem to both consumers (agg and join)
                    // Must use UNBOUNDED: the groupby path can block (waiting on
                    // semi-join), and BOUNDED would then block spill_buffer too.
                    auto lineitem_for_groupby = ctx->create_channel();
                    auto lineitem_for_join_raw = ctx->create_channel();
                    nodes.push_back(
                        rapidsmpf::streaming::node::fanout(
                            ctx,
                            lineitem_shuffled,
                            {lineitem_for_groupby, lineitem_for_join_raw},
                            rapidsmpf::streaming::node::FanoutPolicy::UNBOUNDED
                        )
                    );

                    // 3b. Optionally buffer in spillable container
                    // This allows the spill manager to spill if memory pressure occurs
                    auto lineitem_for_join = ctx->create_channel();
                    if (cmd_options.spillable_fanout) {
                        nodes.push_back(spillable_buffer(
                            ctx, lineitem_for_join_raw, lineitem_for_join
                        ));
                    } else {
                        // Pass through: just forward messages unchanged
                        nodes.push_back(
                            [](
                                std::shared_ptr<rapidsmpf::streaming::Context> ctx_,
                                std::shared_ptr<rapidsmpf::streaming::Channel> ch_in_,
                                std::shared_ptr<rapidsmpf::streaming::Channel> ch_out_
                            ) -> rapidsmpf::streaming::Node {
                                rapidsmpf::streaming::ShutdownAtExit c{ch_in_, ch_out_};
                                while (true) {
                                    auto msg = co_await ch_in_->receive();
                                    if (msg.empty()) {
                                        break;
                                    }
                                    co_await ch_out_->send(std::move(msg));
                                }
                                co_await ch_out_->drain(ctx_->executor());
                            }(ctx, lineitem_for_join_raw, lineitem_for_join)
                        );
                    }

                    // 4. Groupby+filter per partition (output stays partitioned)
                    auto lineitem_agg_filtered = ctx->create_channel();
                    nodes.push_back(groupby_filter_lineitem(
                        ctx, lineitem_for_groupby, lineitem_agg_filtered, 300.0
                    ));  // l_orderkey, sum_quantity (filtered, partitioned)

                    // 5. Shuffle orders by o_orderkey
                    auto orders_shuffled = ctx->create_channel();
                    nodes.push_back(
                        rapidsmpf::ndsh::shuffle(
                            ctx,
                            orders,
                            orders_shuffled,
                            {0},  // o_orderkey
                            num_partitions,
                            rapidsmpf::OpID{
                                static_cast<rapidsmpf::OpID>(10 * i + op_id++)
                            }
                        )
                    );

                    // 6. Shuffle semi-join: orders x lineitem_agg (both partitioned)
                    auto orders_filtered = ctx->create_channel();
                    nodes.push_back(
                        rapidsmpf::ndsh::left_semi_join_shuffle(
                            ctx,
                            orders_shuffled,
                            lineitem_agg_filtered,
                            orders_filtered,
                            {0},  // o_orderkey
                            {0},  // l_orderkey
                            rapidsmpf::ndsh::KeepKeys::YES
                        )
                    );  // o_orderkey, o_custkey, o_orderdate, o_totalprice (partitioned)

                    // 7. Shuffle inner join: orders_filtered x lineitem (both
                    // partitioned)
                    nodes.push_back(
                        rapidsmpf::ndsh::inner_join_shuffle(
                            ctx,
                            orders_filtered,
                            lineitem_for_join,
                            orders_x_lineitem,
                            {0},  // o_orderkey
                            {0},  // l_orderkey
                            rapidsmpf::ndsh::KeepKeys::YES
                        )
                    );  // o_orderkey, o_custkey, o_orderdate, o_totalprice, l_quantity

                } else {
                    // ========== BROADCAST-BASED PIPELINE ==========
                    // For smaller scale factors, use broadcast joins

                    // Read lineitem once, fanout to both consumers
                    auto lineitem = ctx->create_channel();
                    nodes.push_back(read_lineitem(
                        ctx,
                        lineitem,
                        /* num_tickets */ 4,
                        cmd_options.num_rows_per_chunk,
                        cmd_options.input_directory
                    ));

                    auto lineitem_for_agg = ctx->create_channel();
                    auto lineitem_for_join_raw = ctx->create_channel();
                    nodes.push_back(
                        rapidsmpf::streaming::node::fanout(
                            ctx,
                            lineitem,
                            {lineitem_for_agg, lineitem_for_join_raw},
                            rapidsmpf::streaming::node::FanoutPolicy::UNBOUNDED
                        )
                    );

                    // Optionally buffer in spillable container
                    auto lineitem_for_join = ctx->create_channel();
                    if (cmd_options.spillable_fanout) {
                        nodes.push_back(spillable_buffer(
                            ctx, lineitem_for_join_raw, lineitem_for_join
                        ));
                    } else {
                        // Pass through: just forward messages unchanged
                        nodes.push_back(
                            [](
                                std::shared_ptr<rapidsmpf::streaming::Context> ctx_,
                                std::shared_ptr<rapidsmpf::streaming::Channel> ch_in_,
                                std::shared_ptr<rapidsmpf::streaming::Channel> ch_out_
                            ) -> rapidsmpf::streaming::Node {
                                rapidsmpf::streaming::ShutdownAtExit c{ch_in_, ch_out_};
                                while (true) {
                                    auto msg = co_await ch_in_->receive();
                                    if (msg.empty()) {
                                        break;
                                    }
                                    co_await ch_out_->send(std::move(msg));
                                }
                                co_await ch_out_->drain(ctx_->executor());
                            }(ctx, lineitem_for_join_raw, lineitem_for_join)
                        );
                    }

                    // Groupby l_orderkey, sum(l_quantity), filter > 300
                    auto lineitem_agg_filtered = ctx->create_channel();
                    nodes.push_back(groupby_filter_lineitem(
                        ctx, lineitem_for_agg, lineitem_agg_filtered, 300.0
                    ));  // l_orderkey, sum_quantity (filtered)

                    // Concatenate the filtered lineitem aggregates
                    auto lineitem_agg_concat = ctx->create_channel();
                    nodes.push_back(
                        rapidsmpf::ndsh::concatenate(
                            ctx,
                            lineitem_agg_filtered,
                            lineitem_agg_concat,
                            rapidsmpf::ndsh::ConcatOrder::DONT_CARE
                        )
                    );

                    // Semi-join orders with filtered lineitem on o_orderkey = l_orderkey
                    auto orders_filtered = ctx->create_channel();
                    nodes.push_back(
                        rapidsmpf::ndsh::left_semi_join_broadcast(
                            ctx,
                            orders,
                            lineitem_agg_concat,
                            orders_filtered,
                            {0},  // o_orderkey
                            {0},  // l_orderkey
                            rapidsmpf::OpID{
                                static_cast<rapidsmpf::OpID>(10 * i + op_id++)
                            },
                            rapidsmpf::ndsh::KeepKeys::YES
                        )
                    );  // o_orderkey, o_custkey, o_orderdate, o_totalprice

                    // Inner join orders_filtered with lineitem on o_orderkey = l_orderkey
                    nodes.push_back(
                        rapidsmpf::ndsh::inner_join_broadcast(
                            ctx,
                            orders_filtered,
                            lineitem_for_join,
                            orders_x_lineitem,
                            {0},  // o_orderkey
                            {0},  // l_orderkey
                            rapidsmpf::OpID{
                                static_cast<rapidsmpf::OpID>(10 * i + op_id++)
                            },
                            rapidsmpf::ndsh::KeepKeys::YES
                        )
                    );  // o_orderkey, o_custkey, o_orderdate, o_totalprice, l_quantity
                }

                // Customer join - broadcast for small SF, shuffle for large SF (SF3000+)
                auto all_joined = ctx->create_channel();
                if (cmd_options.shuffle_customer) {
                    // Shuffle customer by c_custkey
                    auto customer_shuffled = ctx->create_channel();
                    nodes.push_back(
                        rapidsmpf::ndsh::shuffle(
                            ctx,
                            customer,
                            customer_shuffled,
                            {0},  // c_custkey
                            num_partitions,
                            rapidsmpf::OpID{
                                static_cast<rapidsmpf::OpID>(10 * i + op_id++)
                            }
                        )
                    );

                    // Shuffle orders_x_lineitem by o_custkey (column 1)
                    auto orders_x_lineitem_shuffled = ctx->create_channel();
                    nodes.push_back(
                        rapidsmpf::ndsh::shuffle(
                            ctx,
                            orders_x_lineitem,
                            orders_x_lineitem_shuffled,
                            {1},  // o_custkey
                            num_partitions,
                            rapidsmpf::OpID{
                                static_cast<rapidsmpf::OpID>(10 * i + op_id++)
                            }
                        )
                    );

                    // Shuffle inner join
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
                    // Broadcast join (customer is small at low SF)
                    nodes.push_back(
                        rapidsmpf::ndsh::inner_join_broadcast(
                            ctx,
                            customer,
                            orders_x_lineitem,
                            all_joined,
                            {0},  // c_custkey
                            {1},  // o_custkey
                            rapidsmpf::OpID{
                                static_cast<rapidsmpf::OpID>(10 * i + op_id++)
                            },
                            rapidsmpf::ndsh::KeepKeys::YES
                        )
                    );
                }  // c_custkey, c_name, o_orderkey, o_orderdate, o_totalprice,
                   // l_quantity

                // Reorder columns to match expected output
                auto reordered = ctx->create_channel();
                nodes.push_back(reorder_columns(ctx, all_joined, reordered));

                // Chunkwise groupby aggregation
                auto chunkwise_groupby_output = ctx->create_channel();
                nodes.push_back(
                    chunkwise_groupby_agg(ctx, reordered, chunkwise_groupby_output)
                );

                // Concatenate groupby results
                auto concatenated_groupby_output = ctx->create_channel();
                nodes.push_back(
                    rapidsmpf::ndsh::concatenate(
                        ctx,
                        chunkwise_groupby_output,
                        concatenated_groupby_output,
                        rapidsmpf::ndsh::ConcatOrder::DONT_CARE
                    )
                );

                // Final groupby aggregation
                auto groupby_output = ctx->create_channel();
                nodes.push_back(final_groupby_agg(
                    ctx,
                    concatenated_groupby_output,
                    groupby_output,
                    rapidsmpf::OpID{static_cast<rapidsmpf::OpID>(10 * i + op_id++)}
                ));

                // Sort and limit
                auto sorted_output = ctx->create_channel();
                nodes.push_back(sort_and_limit(ctx, groupby_output, sorted_output));

                // Write output
                nodes.push_back(write_parquet(ctx, sorted_output, output_path));
            }
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> pipeline = end - start;
            start = std::chrono::steady_clock::now();
            {
                RAPIDSMPF_NVTX_SCOPED_RANGE("Q18 Iteration");
                rapidsmpf::streaming::run_streaming_pipeline(std::move(nodes));
            }
            end = std::chrono::steady_clock::now();
            std::chrono::duration<double> compute = end - start;
            comm->logger().print(
                "Iteration ", i, " pipeline construction time [s]: ", pipeline.count()
            );
            comm->logger().print("Iteration ", i, " compute time [s]: ", compute.count());
            timings.push_back(pipeline.count());
            timings.push_back(compute.count());
            ctx->comm()->logger().print(stats->report());
            RAPIDSMPF_MPI(MPI_Barrier(mpi_comm));
        }
        if (comm->rank() == 0) {
            for (int i = 0; i < 2; i++) {
                comm->logger().print(
                    "Iteration ",
                    i,
                    " pipeline construction time [s]: ",
                    timings[size_t(2 * i)]
                );
                comm->logger().print(
                    "Iteration ", i, " compute time [s]: ", timings[size_t(2 * i + 1)]
                );
            }
        }
    }

    RAPIDSMPF_MPI(MPI_Comm_free(&mpi_comm));
    RAPIDSMPF_MPI(MPI_Finalize());
    return 0;
}
