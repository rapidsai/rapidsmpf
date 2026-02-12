/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
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
#include <cudf/context.hpp>
#include <cudf/datetime.hpp>
#include <cudf/groupby.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/round.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
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
#include <rapidsmpf/streaming/core/node.hpp>
#include <rapidsmpf/streaming/cudf/parquet.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

#include "concatenate.hpp"
#include "join.hpp"
#include "utils.hpp"

namespace {

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
                       .column_names({"c_custkey"})
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
    auto options = cudf::io::parquet_reader_options::builder(cudf::io::source_info(files))
                       .column_names({"o_comment", "o_custkey", "o_orderkey"})
                       .build();
    return rapidsmpf::streaming::node::read_parquet(
        ctx, ch_out, num_producers, options, num_rows_per_chunk
    );
}

// TODO: can we push this into the read_orders node?
rapidsmpf::streaming::Node filter_orders(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
    auto mr = ctx->br()->device_mr();
    while (true) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        co_await ctx->executor()->schedule();
        auto chunk =
            co_await msg.release<rapidsmpf::streaming::TableChunk>().make_available(ctx);
        auto chunk_stream = chunk.stream();
        auto table = chunk.table_view();
        auto o_comment = table.column(0);
        // Match rows that contain "special.*requests" and negate to get rows that don't
        auto regex_program = cudf::strings::regex_program::create("special.*requests");
        auto contains_mask =
            cudf::strings::contains_re(o_comment, *regex_program, chunk_stream, mr);
        // Negate: we want rows that do NOT match
        auto mask = cudf::unary_operation(
            contains_mask->view(), cudf::unary_operator::NOT, chunk_stream, mr
        );

        auto filtered = cudf::apply_boolean_mask(
            table.select({1, 2}), mask->view(), chunk_stream, mr
        );

        co_await ch_out->send(
            rapidsmpf::streaming::to_message(
                msg.sequence_number(),
                std::make_unique<rapidsmpf::streaming::TableChunk>(
                    std::move(filtered), chunk_stream
                )
            )
        );
    }
    co_await ch_out->drain(ctx->executor());
}

rapidsmpf::streaming::Node chunkwise_groupby_agg(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    cudf::size_type key_col_idx,
    cudf::size_type value_col_idx,
    auto&& agg_factory
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
    std::vector<cudf::table> partial_results;
    std::uint64_t sequence = 0;
    while (true) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        co_await ctx->executor()->schedule();
        auto chunk = co_await msg.template release<rapidsmpf::streaming::TableChunk>()
                         .make_available(ctx);
        auto chunk_stream = chunk.stream();
        auto table = chunk.table_view();
        auto grouper = cudf::groupby::groupby(
            table.select({key_col_idx}), cudf::null_policy::EXCLUDE, cudf::sorted::NO
        );
        std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
        aggs.emplace_back(agg_factory());

        auto requests = std::vector<cudf::groupby::aggregation_request>();
        requests.push_back(
            cudf::groupby::aggregation_request(
                table.column(value_col_idx), std::move(aggs)
            )
        );

        auto [keys, results] =
            grouper.aggregate(requests, chunk_stream, ctx->br()->device_mr());
        // Drop chunk, we don't need it.
        std::ignore = std::move(chunk);
        auto result = keys->release();
        for (auto&& r : results) {
            std::ranges::move(r.results, std::back_inserter(result));
        }
        co_await ch_out->send(
            rapidsmpf::streaming::to_message(
                sequence++,
                std::make_unique<rapidsmpf::streaming::TableChunk>(
                    std::make_unique<cudf::table>(std::move(result)), chunk_stream
                )
            )
        );
    }
    co_await ch_out->drain(ctx->executor());
}

rapidsmpf::streaming::Node all_gather_concatenated(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    rapidsmpf::OpID tag
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
    co_await ctx->executor()->schedule();

    auto msg = co_await ch_in->receive();
    auto next = co_await ch_in->receive();
    ctx->comm()->logger().print("All gather");
    RAPIDSMPF_EXPECTS(next.empty(), "Expecting concatenated input at this point");
    auto chunk =
        co_await msg.release<rapidsmpf::streaming::TableChunk>().make_available(ctx);
    auto chunk_stream = chunk.stream();
    auto local_table = chunk.table_view();

    if (ctx->comm()->nranks() > 1) {
        // Reduce across ranks...
        // Need a reduce primitive in rapidsmpf, but let's just use an allgather and
        // discard for now.
        rapidsmpf::streaming::AllGather gatherer{ctx, tag};
        if (!local_table.is_empty()) {
            auto pack = cudf::pack(local_table, chunk_stream, ctx->br()->device_mr());
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
                    std::move(packed_data),
                    ctx->br().get(),
                    rapidsmpf::AllowOverbooking::YES,
                    ctx->statistics()
                ),
                chunk_stream,
                ctx->br().get(),
                ctx->statistics()
            );

            co_await ch_out->send(
                rapidsmpf::streaming::to_message(
                    0,
                    std::make_unique<rapidsmpf::streaming::TableChunk>(
                        std::move(global_result), chunk_stream
                    )
                )
            );
        } else {
            // Drop chunk, we don't need it.
            std::ignore = std::move(packed_data);
        }
    } else {
        co_await ch_out->send(
            rapidsmpf::streaming::to_message(
                0, std::make_unique<rapidsmpf::streaming::TableChunk>(std::move(chunk))
            )
        );
    }

    co_await ch_out->drain(ctx->executor());
}

rapidsmpf::streaming::Node groupby_and_sort(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
    co_await ctx->executor()->schedule();
    ctx->comm()->logger().print("Groupby and sort");

    auto msg = co_await ch_in->receive();

    // We know we only have a single chunk from the allgather in rank 0.
    if (msg.empty()) {
        co_return;
    }

    auto next = co_await ch_in->receive();
    RAPIDSMPF_EXPECTS(next.empty(), "Expecting concatenated input at this point");
    auto chunk =
        co_await msg.release<rapidsmpf::streaming::TableChunk>().make_available(ctx);
    auto chunk_stream = chunk.stream();
    auto all_gathered = chunk.table_view();

    cudf::table_view grouped_view;
    std::unique_ptr<cudf::table> grouped;
    // if there were multiple ranks, we have a concatenated table with the groupby
    // results after allgather. This needs be grouped again.
    if (ctx->comm()->nranks() > 1) {
        auto grouper = cudf::groupby::groupby(
            all_gathered.select({0}), cudf::null_policy::EXCLUDE, cudf::sorted::NO
        );

        std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
        aggs.emplace_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());

        auto requests = std::vector<cudf::groupby::aggregation_request>();
        requests.push_back(
            cudf::groupby::aggregation_request(all_gathered.column(1), std::move(aggs))
        );

        auto [keys, results] =
            grouper.aggregate(requests, chunk_stream, ctx->br()->device_mr());
        // Drop chunk, we don't need it.
        std::ignore = std::move(chunk);

        auto result = keys->release();
        for (auto&& r : results) {
            std::ranges::move(r.results, std::back_inserter(result));
        }
        grouped = std::make_unique<cudf::table>(std::move(result));
        grouped_view = grouped->view();
    } else {
        grouped_view = all_gathered;
    }

    // We will only actually bother to do this on rank zero.
    auto sorted = cudf::sort_by_key(
        grouped_view,
        grouped_view.select({1, 0}),
        {cudf::order::DESCENDING, cudf::order::ASCENDING},
        {},
        chunk_stream,
        ctx->br()->device_mr()
    );
    grouped.reset();

    co_await ch_out->send(
        rapidsmpf::streaming::to_message(
            0,
            std::make_unique<rapidsmpf::streaming::TableChunk>(
                std::move(sorted), chunk_stream
            )
        )
    );

    co_await ch_out->drain(ctx->executor());
}

rapidsmpf::streaming::Node write_parquet(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::string output_path
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in};
    co_await ctx->executor()->schedule();
    while (true) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            co_return;
        }
        ctx->comm()->logger().print("Write parquet");
        auto chunk =
            co_await msg.release<rapidsmpf::streaming::TableChunk>().make_available(ctx);
        auto sink = cudf::io::sink_info(output_path + ".parquet");
        auto builder =
            cudf::io::parquet_writer_options::builder(sink, chunk.table_view());
        auto metadata = cudf::io::table_input_metadata(chunk.table_view());
        // Q13 output: c_custkey, o_orderkey
        metadata.column_metadata[0].set_name("count");
        metadata.column_metadata[1].set_name("custdist");
        builder = builder.metadata(metadata);
        auto options = builder.build();
        cudf::io::write_parquet(options, chunk.stream());
        ctx->comm()->logger().print(
            "Wrote chunk with ",
            chunk.table_view().num_rows(),
            " rows and ",
            chunk.table_view().num_columns(),
            " columns to ",
            output_path
        );
    }
    co_await ch_in->drain(ctx->executor());
}

}  // namespace

struct ProgramOptions {
    int num_streaming_threads{1};
    cudf::size_type num_rows_per_chunk{100'000'000};
    std::optional<double> spill_device_limit{std::nullopt};
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
            << "  --spill-device-limit <n>     Fractional spill device limit (default: "
               "None)\n"
            << "  --output-file <path>         Output file path (required)\n"
            << "  --input-directory <path>     Input directory path (required)\n"
            << "  --help                       Show this help message\n";
    };

    static struct option long_options[] = {
        {"num-streaming-threads", required_argument, nullptr, 1},
        {"num-rows-per-chunk", required_argument, nullptr, 2},
        {"output-file", required_argument, nullptr, 3},
        {"input-directory", required_argument, nullptr, 4},
        {"help", no_argument, nullptr, 5},
        {"spill-device-limit", required_argument, nullptr, 6},
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
            options.output_file = optarg;
            saw_output_file = true;
            break;
        case 4:
            options.input_directory = optarg;
            saw_input_directory = true;
            break;
        case 5:
            print_usage();
            std::exit(0);
        case 6:
            options.spill_device_limit = std::stod(optarg);
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
 * @brief Run a derived version of TPC-H query 9.
 *
 * The SQL form of the query is:
 * @code{.sql}
 * select
 *     c_count,
 *     COUNT(*) AS custdist
 * from
 *     (
 *         select
 *             c_custkey,
 *             COUNT(o_orderkey) AS c_count
 *         from
 *             customer
 *         left outer join orders on
 *             c_custkey = o_custkey
 *             and o_comment not like '%special%requests%'
 *         group by
 *             c_custkey
 *     ) as c_orders
 * group by
 *     c_count
 * order by
 *     custdist DESC,
 *     c_count DESC;
 * @endcode{}
 */
int main(int argc, char** argv) {
    rapidsmpf::ndsh::FinalizeMPI finalize{};
    cudaFree(nullptr);
    // work around https://github.com/rapidsai/cudf/issues/20849
    cudf::initialize();
    auto mr = rmm::mr::cuda_async_memory_resource{};
    auto stats_wrapper = rapidsmpf::RmmResourceAdaptor(&mr);
    auto arguments = rapidsmpf::ndsh::parse_arguments(argc, argv);
    auto ctx = rapidsmpf::ndsh::create_context(arguments, &stats_wrapper);
    std::string output_path = arguments.output_file;
    std::vector<double> timings;
    auto comm = ctx->comm();

    for (int i = 0; i < arguments.num_iterations; i++) {
        rapidsmpf::OpID op_id{0};
        std::vector<rapidsmpf::streaming::Node> nodes;
        auto start = std::chrono::steady_clock::now();
        {
            RAPIDSMPF_NVTX_SCOPED_RANGE("Constructing Q13 pipeline");

            auto customer = ctx->create_channel();
            nodes.push_back(read_customer(
                ctx,
                customer,
                /* num_tickets */ 4,
                arguments.num_rows_per_chunk,
                arguments.input_directory
            ));  // c_custkey

            auto orders = ctx->create_channel();
            nodes.push_back(read_orders(
                ctx,
                orders,
                /* num_tickets */ 4,
                arguments.num_rows_per_chunk,
                arguments.input_directory
            ));  // o_comment, o_custkey, o_orderkey


            auto filtered_orders = ctx->create_channel();
            nodes.push_back(
                filter_orders(ctx, orders, filtered_orders)  // o_custkey, o_orderkey
            );

            std::uint32_t num_partitions = 128;  // should be configurable?

            auto orders_shuffled = ctx->create_channel();
            nodes.push_back(
                rapidsmpf::ndsh::shuffle(
                    ctx,
                    filtered_orders,
                    orders_shuffled,
                    {0},
                    num_partitions,
                    rapidsmpf::OpID{static_cast<rapidsmpf::OpID>(10 * i + op_id++)}
                )
            );  // o_custkey, o_orderkey

            auto customer_shuffled = ctx->create_channel();
            nodes.push_back(
                rapidsmpf::ndsh::shuffle(
                    ctx,
                    customer,
                    customer_shuffled,
                    {0},
                    num_partitions,
                    rapidsmpf::OpID{static_cast<rapidsmpf::OpID>(10 * i + op_id++)}
                )
            );  // c_custkey

            // left join customer_shuffled and orders_shuffled
            auto customer_x_orders = ctx->create_channel();
            nodes.push_back(
                rapidsmpf::ndsh::left_join_shuffle(
                    ctx, customer_shuffled, orders_shuffled, customer_x_orders, {0}, {0}
                )
            );  // c_custkey, o_orderkey

            auto chunkwise_groupby_output = ctx->create_channel();
            nodes.push_back(chunkwise_groupby_agg(
                ctx, customer_x_orders, chunkwise_groupby_output, 0, 1, [] {
                    return cudf::make_count_aggregation<cudf::groupby_aggregation>();
                }
            ));  // c_custkey, count

            auto concatenated_groupby_output = ctx->create_channel();
            nodes.push_back(
                rapidsmpf::ndsh::concatenate(
                    ctx,
                    chunkwise_groupby_output,
                    concatenated_groupby_output,
                    rapidsmpf::ndsh::ConcatOrder::DONT_CARE
                )
            );  // c_custkey, count

            auto groupby_output = ctx->create_channel();
            nodes.push_back(chunkwise_groupby_agg(
                ctx, concatenated_groupby_output, groupby_output, 0, 1, [] {
                    return cudf::make_sum_aggregation<cudf::groupby_aggregation>();
                }
            ));  // c_custkey, count

            auto groupby_count_output = ctx->create_channel();
            nodes.push_back(chunkwise_groupby_agg(
                ctx, groupby_output, groupby_count_output, 1, 0, [] {
                    return cudf::make_count_aggregation<cudf::groupby_aggregation>();
                }
            ));  // count, len

            auto all_gather_concatenated_output = ctx->create_channel();
            nodes.push_back(all_gather_concatenated(
                ctx,
                groupby_count_output,
                all_gather_concatenated_output,
                rapidsmpf::OpID{static_cast<rapidsmpf::OpID>(10 * i + op_id++)}
            ));  // count, custdist

            auto groupby_and_sort_output = ctx->create_channel();
            nodes.push_back(groupby_and_sort(
                ctx,
                all_gather_concatenated_output,
                groupby_and_sort_output
            ));  // count, custdist

            nodes.push_back(write_parquet(
                ctx,
                groupby_and_sort_output,
                arguments.output_file + "_r" + std::to_string(ctx->comm()->rank()) + "_i"
                    + std::to_string(i)
            ));
        }

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> pipeline = end - start;
        start = std::chrono::steady_clock::now();
        {
            RAPIDSMPF_NVTX_SCOPED_RANGE("Q13 Iteration");
            rapidsmpf::streaming::run_streaming_pipeline(std::move(nodes));
        }
        end = std::chrono::steady_clock::now();
        std::chrono::duration<double> compute = end - start;
        timings.push_back(pipeline.count());
        timings.push_back(compute.count());
        comm->logger().print(ctx->statistics()->report());
        ctx->statistics()->clear();
    }

    if (comm->rank() == 0) {
        for (int i = 0; i < arguments.num_iterations; i++) {
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
    return 0;
}
